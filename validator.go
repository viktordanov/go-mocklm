package main

//go:generate go run ./cmd/specsync

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"

	jsonschema "github.com/santhosh-tekuri/jsonschema/v6"
)

// validate_responses: bodies covered by the vendored response-side
// closure of nanollm's pinned specs (spec/*.schema.json, extracted by
// cmd/specsync) are checked against it before being written — OpenAI
// chat-completion and Anthropic messages bodies (non-stream responses and
// each SSE data payload) plus provider error envelopes on every endpoint.
// Success bodies of /v1/completions, /v1/embeddings, /v1/responses, and
// /v1/models are outside the extracted closure (the closure roots mirror
// nanollm's oracle roots, which never covered those surfaces) and are NOT
// validated. A violation is a mock bug: the request fails loudly (500
// before headers, RST mid-stream, or a panic when MOCKLM_VALIDATE_PANIC
// is set) instead of leaking drift into tests. Deliberately-invalid fault
// output (malformed_chunk, emit_nonstandard_fields) bypasses validation
// by design — being invalid is those faults' purpose.

//go:embed spec/openai-responses.schema.json
var openaiResponsesSchemaJSON []byte

//go:embed spec/anthropic-responses.schema.json
var anthropicResponsesSchemaJSON []byte

// pingEventSchema is the local patch for the typed Anthropic ping event:
// the real API sends it (and the mock emits it by default) but the pinned
// MessageStreamEvent union has no ping arm, so it is validated against
// this hand-written arm instead of the vendored union.
const pingEventSchema = `{
	"type": "object",
	"properties": {"type": {"const": "ping"}},
	"required": ["type"],
	"additionalProperties": false
}`

// bodyKind names an emitted-body shape and maps to its schema root.
type bodyKind int

const (
	kindOpenAIChat bodyKind = iota
	kindOpenAIChunk
	kindOpenAIError
	kindAnthropicMessage
	kindAnthropicEvent
	kindAnthropicError
)

var compiledSchemas struct {
	once sync.Once
	err  error

	openaiChat  *jsonschema.Schema
	openaiChunk *jsonschema.Schema
	openaiError *jsonschema.Schema
	anthMessage *jsonschema.Schema
	anthEvent   *jsonschema.Schema
	anthError   *jsonschema.Schema
	anthPing    *jsonschema.Schema
}

// validationFailures counts every schema violation detected on a default
// (non-bypassed) emitted body. go-mocklm's own tests assert it stays zero.
var validationFailures atomic.Int64

func compileSchemas() error {
	compiledSchemas.once.Do(func() {
		compiledSchemas.err = func() error {
			c := jsonschema.NewCompiler()
			for url, raw := range map[string]string{
				"mocklm://spec/openai-responses.schema.json":    string(openaiResponsesSchemaJSON),
				"mocklm://spec/anthropic-responses.schema.json": string(anthropicResponsesSchemaJSON),
				"mocklm://spec/ping-event.schema.json":          pingEventSchema,
			} {
				doc, err := jsonschema.UnmarshalJSON(strings.NewReader(raw))
				if err != nil {
					return fmt.Errorf("parsing %s: %w", url, err)
				}
				if err := c.AddResource(url, doc); err != nil {
					return fmt.Errorf("adding %s: %w", url, err)
				}
			}

			compile := func(dst **jsonschema.Schema, url string) error {
				s, err := c.Compile(url)
				if err != nil {
					return fmt.Errorf("compiling %s: %w", url, err)
				}
				*dst = s
				return nil
			}
			for _, spec := range []struct {
				dst **jsonschema.Schema
				url string
			}{
				{&compiledSchemas.openaiChat, "mocklm://spec/openai-responses.schema.json#/$defs/CreateChatCompletionResponse"},
				{&compiledSchemas.openaiChunk, "mocklm://spec/openai-responses.schema.json#/$defs/CreateChatCompletionStreamResponse"},
				{&compiledSchemas.openaiError, "mocklm://spec/openai-responses.schema.json#/$defs/ErrorResponse"},
				{&compiledSchemas.anthMessage, "mocklm://spec/anthropic-responses.schema.json#/$defs/Message"},
				{&compiledSchemas.anthEvent, "mocklm://spec/anthropic-responses.schema.json#/$defs/MessageStreamEvent"},
				{&compiledSchemas.anthError, "mocklm://spec/anthropic-responses.schema.json#/$defs/ErrorResponse"},
				{&compiledSchemas.anthPing, "mocklm://spec/ping-event.schema.json"},
			} {
				if err := compile(spec.dst, spec.url); err != nil {
					return err
				}
			}
			return nil
		}()
	})
	return compiledSchemas.err
}

// envValidateResponses reports whether MOCKLM_VALIDATE_RESPONSES turns
// validation on as the default. The per-provider validate_responses knob
// (a tri-state *bool) overrides it in either direction, so a deliberate
// fault scenario can opt a request out via X-MockLM-Fault.
func envValidateResponses() bool {
	return envTruthy(os.Getenv("MOCKLM_VALIDATE_RESPONSES"))
}

// envValidatePanic makes a validation failure panic (test mode) instead of
// failing the request with a 500/RST.
func envValidatePanic() bool {
	return envTruthy(os.Getenv("MOCKLM_VALIDATE_PANIC"))
}

func envTruthy(v string) bool {
	switch strings.ToLower(v) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

// shouldValidate resolves the effective validate_responses mode for a
// request's config snapshot (nil cfg = env default only).
func shouldValidate(cfg *ProviderConfig) bool {
	if cfg != nil && cfg.ValidateResponses != nil {
		return *cfg.ValidateResponses
	}
	return envValidateResponses()
}

// validateEmittedBody checks one emitted body against its schema root.
// Anthropic stream payloads whose type is "ping" are validated against the
// local ping arm (see pingEventSchema).
func validateEmittedBody(kind bodyKind, data []byte) error {
	if err := compileSchemas(); err != nil {
		return fmt.Errorf("schema compilation failed: %w", err)
	}
	inst, err := jsonschema.UnmarshalJSON(bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("emitted body is not valid JSON: %w", err)
	}

	var sch *jsonschema.Schema
	switch kind {
	case kindOpenAIChat:
		sch = compiledSchemas.openaiChat
	case kindOpenAIChunk:
		sch = compiledSchemas.openaiChunk
	case kindOpenAIError:
		sch = compiledSchemas.openaiError
	case kindAnthropicMessage:
		sch = compiledSchemas.anthMessage
	case kindAnthropicEvent:
		sch = compiledSchemas.anthEvent
		if obj, ok := inst.(map[string]any); ok && obj["type"] == "ping" {
			sch = compiledSchemas.anthPing
		}
	case kindAnthropicError:
		sch = compiledSchemas.anthError
	}
	return sch.Validate(inst)
}

// failStreamValidation reports a mid-stream violation and severs the
// connection (RST) — headers are already written, so a 500 is impossible;
// the client sees a hard failure instead of an off-spec frame.
func failStreamValidation(w http.ResponseWriter, where string, body []byte, err error) {
	reportValidationFailure(where, body, err)
	hijackAndClose(w)
}

// reportValidationFailure records a violation and panics in test mode.
// Callers decide how to fail the request (500 before headers, RST
// mid-stream).
func reportValidationFailure(where string, body []byte, err error) {
	validationFailures.Add(1)
	msg := fmt.Sprintf("validate_responses: %s violates the pinned spec: %v\nbody: %s", where, err, body)
	if envValidatePanic() {
		panic(msg)
	}
	log.Print(msg)
}

// writeValidatedJSON marshals resp, validates it when the config snapshot
// asks for validation, and writes it as a 200. On violation the request
// fails with a plain 500 (the fallback envelope is hand-built and not
// re-validated, to avoid recursion).
func writeValidatedJSON(w http.ResponseWriter, cfg *ProviderConfig, kind bodyKind, where string, resp any) {
	data, err := marshalBody(resp)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if shouldValidate(cfg) && !bypassesValidation(cfg) {
		if verr := validateEmittedBody(kind, data); verr != nil {
			reportValidationFailure(where, data, verr)
			writeValidationFailure(w, kind, where, verr)
			return
		}
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(data)
}

// bypassesValidation reports whether this config snapshot deliberately
// emits off-spec success bodies (unknown-field probes). Error envelopes do
// not consult it — no fault knob corrupts those.
func bypassesValidation(cfg *ProviderConfig) bool {
	return cfg != nil && cfg.EmitNonstandardFields
}

// writeValidationFailure emits the loud 500 for a pre-header violation.
// The envelope is provider-shaped and static so it cannot itself fail.
func writeValidationFailure(w http.ResponseWriter, kind bodyKind, where string, err error) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusInternalServerError)
	msg := fmt.Sprintf("mocklm validate_responses: %s violates the pinned spec: %v", where, err)
	var body string
	if kind == kindAnthropicMessage || kind == kindAnthropicEvent || kind == kindAnthropicError {
		body = fmt.Sprintf(`{"type":"error","error":{"type":"api_error","message":%q},"request_id":"req_mock_validation"}`, msg)
	} else {
		body = fmt.Sprintf(`{"error":{"message":%q,"type":"server_error","param":null,"code":"server_error"}}`, msg)
	}
	fmt.Fprintln(w, body)
}

func marshalBody(resp any) ([]byte, error) {
	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(resp); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
