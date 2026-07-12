package main

import (
	"encoding/json"
	"strings"
	"testing"
)

// --- validate_responses self-validation mode ---

func boolPtr(b bool) *bool { return &b }

// validatingConfig returns a fast test config with validate_responses
// forced on for both providers.
func validatingConfig() *Config {
	cfg := defaultConfig()
	cfg.OpenAI.ValidateResponses = boolPtr(true)
	cfg.Anthropic.ValidateResponses = boolPtr(true)
	return cfg
}

// TestValidatorHasTeeth proves the compiled closure rejects the drift
// classes the loop exists to catch — if any of these pass, validation is
// silently vacuous.
func TestValidatorHasTeeth(t *testing.T) {
	valid := func(kind bodyKind, body string) error {
		t.Helper()
		return validateEmittedBody(kind, []byte(body))
	}

	okChunk := `{"id":"c","object":"chat.completion.chunk","created":1,"model":"m",` +
		`"choices":[{"index":0,"delta":{"content":"x"},"logprobs":null,"finish_reason":null}]}`
	if err := valid(kindOpenAIChunk, okChunk); err != nil {
		t.Fatalf("baseline chunk must validate (null finish_reason via the enum-null extension): %v", err)
	}

	cases := []struct {
		name string
		kind bodyKind
		body string
	}{
		{"openai chat: unknown top-level field (injected additionalProperties:false)",
			kindOpenAIChat,
			`{"id":"c","object":"chat.completion","created":1,"model":"m","choices":[],"stop_details":null}`},
		{"openai chunk: off-vocabulary finish_reason",
			kindOpenAIChunk,
			strings.Replace(okChunk, `"finish_reason":null`, `"finish_reason":"banana"`, 1)},
		{"openai error: missing required param",
			kindOpenAIError,
			`{"error":{"message":"m","type":"server_error","code":null}}`},
		{"anthropic message: missing required stop_details",
			kindAnthropicMessage,
			`{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-3-haiku-20240307",` +
				`"stop_reason":"end_turn","stop_sequence":null,"container":null,` +
				`"usage":{"input_tokens":1,"output_tokens":1,"cache_read_input_tokens":0,"cache_creation_input_tokens":0,` +
				`"cache_creation":null,"server_tool_use":null,"service_tier":"standard","inference_geo":null,"output_tokens_details":null}}`},
		{"anthropic event: unknown top-level type",
			kindAnthropicEvent,
			`{"type":"message_future"}`},
		{"anthropic ping: extra field on the local arm",
			kindAnthropicEvent,
			`{"type":"ping","extra":1}`},
		{"anthropic error: server_error is not a union arm",
			kindAnthropicError,
			`{"type":"error","error":{"type":"server_error","message":"m"},"request_id":null}`},
	}
	for _, tc := range cases {
		if err := valid(tc.kind, tc.body); err == nil {
			t.Errorf("%s: expected a validation error, got none", tc.name)
		}
	}

	if err := valid(kindAnthropicEvent, `{"type":"ping"}`); err != nil {
		t.Errorf("typed ping must validate against the local arm: %v", err)
	}
}

// TestValidateResponsesDefaultShapesPass drives every validated surface
// (OpenAI chat, Anthropic messages, error envelopes) in its default shapes
// with validate_responses on and asserts zero violations: the mock's own
// output passes the same pinned-spec closure nanollm's oracle is generated
// from. Legacy completions/embeddings/responses/models success bodies are
// outside the closure and deliberately not driven here.
func TestValidateResponsesDefaultShapesPass(t *testing.T) {
	cfg := validatingConfig()
	cfg.Anthropic.ReasoningTokens = 15
	cfg.Anthropic.CacheReadTokens = 7
	srv := testServer(cfg)
	defer srv.Close()

	before := validationFailures.Load()

	// OpenAI non-stream + stream (with and without the usage trailer).
	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
	if err != nil || resp.StatusCode != 200 {
		t.Fatalf("openai non-stream: err=%v status=%v", err, resp.StatusCode)
	}
	resp.Body.Close()
	lines := collectDataLines(t, srv.URL, openaiChatBody(true), nil)
	if lines[len(lines)-1] != "[DONE]" {
		t.Fatalf("openai stream must complete under validation, got %v", lines)
	}
	lines = collectDataLines(t, srv.URL, openaiStreamBodyWithUsage(), nil)
	if lines[len(lines)-1] != "[DONE]" {
		t.Fatalf("openai usage-trailer stream must complete under validation, got %v", lines)
	}

	// Anthropic non-stream (with a thinking block) + stream (thinking +
	// typed ping validated via the local arm).
	resp = postAnthropic(t, srv.URL, anthropicBody(false))
	if resp.StatusCode != 200 {
		t.Fatalf("anthropic non-stream: status=%d", resp.StatusCode)
	}
	resp.Body.Close()
	text := readStreamUntilClosed(t, srv.URL, anthropicBody(true))
	if !strings.Contains(text, "event: ping") || !strings.Contains(text, "event: message_stop") {
		t.Fatalf("anthropic stream must carry ping and complete under validation:\n%s", text)
	}

	// Tool-use shapes, both providers.
	toolCfg := `{"tool_use_response":true,"validate_responses":true}`
	resp, err = postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false),
		map[string]string{"X-MockLM-Fault": toolCfg})
	if err != nil || resp.StatusCode != 200 {
		t.Fatalf("openai tool-use non-stream: err=%v status=%v", err, resp.StatusCode)
	}
	resp.Body.Close()
	resp, err = postJSON(srv.URL+"/v1/messages", anthropicBody(true), map[string]string{
		"X-MockLM-Fault":    toolCfg,
		"x-api-key":         "k",
		"anthropic-version": "2023-06-01",
	})
	if err != nil || resp.StatusCode != 200 {
		t.Fatalf("anthropic tool-use stream: err=%v status=%v", err, resp.StatusCode)
	}
	resp.Body.Close()

	// Injected error envelopes are validated too.
	for _, hdr := range []string{
		`{"error_rate":1.0,"error_status":529,"validate_responses":true}`,
		`{"error_rate":1.0,"error_status":429,"validate_responses":true}`,
	} {
		resp, err = postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false),
			map[string]string{"X-MockLM-Fault": hdr})
		if err != nil {
			t.Fatalf("openai injected error: %v", err)
		}
		resp.Body.Close()
	}

	if after := validationFailures.Load(); after != before {
		t.Fatalf("default shapes tripped the validator %d time(s) — the mock violates its own pinned spec", after-before)
	}
}

// TestValidationFailureFailsLoudly forces an off-spec body (an invented
// stop_reason) with validation on: non-stream requests must 500 with a
// spec-shaped envelope, streams must be severed before the violating
// event reaches the wire.
func TestValidationFailureFailsLoudly(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	before := validationFailures.Load()
	hdr := map[string]string{
		"X-MockLM-Fault":    `{"stop_reason":"mock_invented_reason","validate_responses":true}`,
		"x-api-key":         "k",
		"anthropic-version": "2023-06-01",
	}

	// Non-stream: 500 with an Anthropic-shaped api_error envelope.
	resp, err := postJSON(srv.URL+"/v1/messages", anthropicBody(false), hdr)
	if err != nil {
		t.Fatalf("POST failed: %v", err)
	}
	if resp.StatusCode != 500 {
		t.Fatalf("expected 500 on validation failure, got %d", resp.StatusCode)
	}
	var env map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&env); err != nil {
		t.Fatalf("decoding failure envelope: %v", err)
	}
	resp.Body.Close()
	if env["type"] != "error" {
		t.Fatalf("failure envelope must be Anthropic-shaped, got %v", env)
	}
	msg, _ := env["error"].(map[string]any)["message"].(string)
	if !strings.Contains(msg, "validate_responses") {
		t.Fatalf("failure message must name validate_responses, got %q", msg)
	}

	// Stream: the violating message_delta must never be written; the
	// connection is severed after the last valid event.
	sresp, err := postJSON(srv.URL+"/v1/messages", anthropicBody(true), hdr)
	if err != nil {
		t.Fatalf("stream POST failed: %v", err)
	}
	raw := new(strings.Builder)
	buf := make([]byte, 4096)
	for {
		n, rerr := sresp.Body.Read(buf)
		raw.WriteString(string(buf[:n]))
		if rerr != nil {
			break
		}
	}
	sresp.Body.Close()
	text := raw.String()
	if !strings.Contains(text, "event: content_block_stop") {
		t.Fatalf("stream must progress up to the violation:\n%s", text)
	}
	if strings.Contains(text, "event: message_delta") || strings.Contains(text, "event: message_stop") {
		t.Fatalf("violating event leaked to the wire:\n%s", text)
	}

	if got := validationFailures.Load() - before; got != 2 {
		t.Fatalf("expected 2 recorded violations, got %d", got)
	}
}

// TestDeliberateFaultsBypassValidation: deliberately-invalid output is the
// point of malformed_chunk and emit_nonstandard_fields — validation must
// not swallow them even when switched on.
func TestDeliberateFaultsBypassValidation(t *testing.T) {
	before := validationFailures.Load()

	// emit_nonstandard_fields: probes ride an otherwise-valid response.
	cfg := validatingConfig()
	cfg.Anthropic.EmitNonstandardFields = true
	srv := testServer(cfg)
	resp := postAnthropic(t, srv.URL, anthropicBody(false))
	body := decodeBody(t, resp)
	srv.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("emit_nonstandard_fields must bypass validation, got %d", resp.StatusCode)
	}
	if _, ok := body["x_mock_unknown_field"]; !ok {
		t.Fatalf("probe field missing from bypassed response: %v", body)
	}

	// malformed_chunk: the corrupt frame reaches the wire and the stream
	// still completes.
	cfg = validatingConfig()
	cfg.OpenAI.MalformedChunk = true
	srv = testServer(cfg)
	lines := collectDataLines(t, srv.URL, openaiChatBody(true), nil)
	srv.Close()
	found := false
	for _, l := range lines {
		if strings.HasPrefix(l, "{INVALID JSON") {
			found = true
		}
	}
	if !found || lines[len(lines)-1] != "[DONE]" {
		t.Fatalf("malformed_chunk must reach the wire and the stream complete, got %v", lines)
	}

	if after := validationFailures.Load(); after != before {
		t.Fatalf("deliberate faults must not count as violations, got %d", after-before)
	}
}

// TestValidateResponsesEnvDefaultAndOptOut: MOCKLM_VALIDATE_RESPONSES
// turns validation on as the default (the nanollm-harness path); an
// explicit validate_responses:false in X-MockLM-Fault opts a deliberate
// fault scenario back out.
func TestValidateResponsesEnvDefaultAndOptOut(t *testing.T) {
	t.Setenv("MOCKLM_VALIDATE_RESPONSES", "1")

	srv := testServer(defaultConfig())
	defer srv.Close()

	hdrs := map[string]string{
		"x-api-key":         "k",
		"anthropic-version": "2023-06-01",
	}

	// Env default on: the off-spec stop_reason trips the validator.
	hdrs["X-MockLM-Fault"] = `{"stop_reason":"mock_invented_reason"}`
	resp, err := postJSON(srv.URL+"/v1/messages", anthropicBody(false), hdrs)
	if err != nil {
		t.Fatalf("POST failed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 500 {
		t.Fatalf("env default must validate, got %d", resp.StatusCode)
	}

	// Explicit opt-out wins over the env default.
	hdrs["X-MockLM-Fault"] = `{"stop_reason":"mock_invented_reason","validate_responses":false}`
	resp, err = postJSON(srv.URL+"/v1/messages", anthropicBody(false), hdrs)
	if err != nil {
		t.Fatalf("POST failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("explicit opt-out must bypass the env default, got %d", resp.StatusCode)
	}
	var out map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if out["stop_reason"] != "mock_invented_reason" {
		t.Fatalf("opted-out request must carry the off-spec value, got %v", out["stop_reason"])
	}
}
