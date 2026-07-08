// Package specsync extracts the response-side JSON-Schema closure from
// nanollm's vendored OpenAPI specs (spec/openai-openapi.json,
// spec/anthropic-openapi.json) so go-mocklm can validate the bodies those
// roots cover — chat-completion and messages responses plus provider
// error envelopes — against the same sha256-pinned contract nanollm's
// Rust oracle is generated from.
//
// The walk and the two normalization rules are ports of nanollm's
// xtask/src/main.rs:
//
//   - closure(): the transitive #/components/schemas/ $ref walk from the
//     response roots (xtask closure(), main.rs:288-340), with the same
//     dangling-ref failure.
//   - nullable-flag: the OpenAI spec declares openapi 3.1.0 but uses the
//     3.0 `nullable: true` idiom; rewrite it to a real null union
//     (main.rs:504-528). One validator-semantics addition over the xtask
//     rule: when the nullable node also carries `enum`, null is appended
//     to the enum values — `enum` is an independent JSON-Schema assertion,
//     so `type: [X, "null"]` alone would still reject null instances
//     (typify does not need this because it renders Option<Enum>).
//   - additionalProperties:false injection on exactly the two OpenAI
//     response roots (main.rs:48-51, 194-199) — without it OpenAI response
//     validation is unknown-field-blind.
//
// The Anthropic closure needs zero conversion; only refs are rewritten.
// The typify-compat rules (strip-title/pattern/default, anyOf→oneOf,
// fold-nullable) are deliberately NOT ported: they exist for Rust codegen
// and would only weaken runtime validation.
//
// Consequence, intentional: the runtime accept/reject surface is not
// byte-identical to nanollm's generated Rust deserializers. The validator
// is stricter wherever the spec carries `pattern` (retained here, stripped
// for typify) and diverges at two documented points — the enum-null
// addition above, and the local ping arm the validator (not this package)
// adds for the typed Anthropic ping event. Same pinned bytes, same roots,
// deliberate deltas.
package specsync

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"slices"
	"sort"
	"strings"
)

const refPrefix = "#/components/schemas/"

// OpenAIRoots are the response-side roots of nanollm's OpenAI oracle
// (xtask OPENAI_ROOTS minus the request root CreateChatCompletionRequest).
var OpenAIRoots = []string{
	"CreateChatCompletionResponse",
	"CreateChatCompletionStreamResponse",
	"ErrorResponse",
}

// openaiInjectStrict mirrors xtask OPENAI_INJECT_STRICT: the egress roots
// that get additionalProperties:false for deny-unknown-fields teeth.
var openaiInjectStrict = []string{
	"CreateChatCompletionResponse",
	"CreateChatCompletionStreamResponse",
}

// AnthropicRoots are the response-side roots of nanollm's Anthropic oracle
// (xtask ANTHROPIC_ROOTS minus the request root CreateMessageParams).
var AnthropicRoots = []string{
	"Message",
	"MessageStreamEvent",
	"ErrorResponse",
}

// Extraction is the vendorable output for one provider spec.
type Extraction struct {
	// Schema is a self-contained JSON Schema 2020-12 document whose $defs
	// hold the closure; roots are addressed as #/$defs/<Name>.
	Schema []byte
	// SourceSHA256 is the hex sha256 of the source spec bytes (the pin).
	SourceSHA256 string
	// Names are the sorted schema names in the closure.
	Names []string
}

// ExtractOpenAI extracts the OpenAI response closure and applies the two
// ported normalization rules.
func ExtractOpenAI(spec []byte) (*Extraction, error) {
	return extract(spec, OpenAIRoots, true)
}

// ExtractAnthropic extracts the Anthropic response closure (no
// normalization needed beyond the ref rewrite).
func ExtractAnthropic(spec []byte) (*Extraction, error) {
	return extract(spec, AnthropicRoots, false)
}

func extract(spec []byte, roots []string, openaiRules bool) (*Extraction, error) {
	dec := json.NewDecoder(bytes.NewReader(spec))
	dec.UseNumber()
	var doc map[string]any
	if err := dec.Decode(&doc); err != nil {
		return nil, fmt.Errorf("parsing spec: %w", err)
	}

	components, _ := doc["components"].(map[string]any)
	schemas, _ := components["schemas"].(map[string]any)
	if schemas == nil {
		return nil, fmt.Errorf("spec has no components.schemas")
	}

	var missing []string
	for _, r := range roots {
		if _, ok := schemas[r]; !ok {
			missing = append(missing, r)
		}
	}
	if len(missing) > 0 {
		return nil, fmt.Errorf("root schemas missing from spec: %v — spec changed; update the roots deliberately", missing)
	}

	names, err := closure(schemas, roots)
	if err != nil {
		return nil, err
	}

	defs := make(map[string]any, len(names))
	for _, name := range names {
		schema := schemas[name]
		if openaiRules {
			rewriteNullableFlag(schema)
		}
		rewriteRefs(schema)
		if openaiRules && slices.Contains(openaiInjectStrict, name) {
			if obj, ok := schema.(map[string]any); ok {
				obj["additionalProperties"] = false
			}
		}
		defs[name] = schema
	}

	out := map[string]any{
		"$schema": "https://json-schema.org/draft/2020-12/schema",
		"$comment": "@generated by `go run ./cmd/specsync` from nanollm's pinned spec. " +
			"Do not edit; see spec/pins.json for the source sha256.",
		"$defs": defs,
	}
	pretty, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return nil, err
	}
	pretty = append(pretty, '\n')

	sum := sha256.Sum256(spec)
	return &Extraction{
		Schema:       pretty,
		SourceSHA256: hex.EncodeToString(sum[:]),
		Names:        names,
	}, nil
}

// closure walks the transitive $ref closure from the roots, mirroring
// xtask's closure() including the dangling-ref bail.
func closure(schemas map[string]any, roots []string) ([]string, error) {
	seen := map[string]bool{}
	dangling := map[string]bool{}
	stack := append([]string(nil), roots...)

	for len(stack) > 0 {
		name := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if seen[name] {
			continue
		}
		if _, ok := schemas[name]; !ok {
			dangling[name] = true
			continue
		}
		seen[name] = true
		collectRefs(schemas[name], &stack)
	}

	if len(dangling) > 0 {
		var names []string
		for n := range dangling {
			names = append(names, n)
		}
		sort.Strings(names)
		return nil, fmt.Errorf("spec contains $refs to schemas that do not exist: %v", names)
	}

	names := make([]string, 0, len(seen))
	for n := range seen {
		names = append(names, n)
	}
	sort.Strings(names)
	return names, nil
}

func collectRefs(node any, out *[]string) {
	switch n := node.(type) {
	case map[string]any:
		for key, value := range n {
			if key == "$ref" {
				if s, ok := value.(string); ok && strings.HasPrefix(s, refPrefix) {
					*out = append(*out, strings.TrimPrefix(s, refPrefix))
				}
				continue
			}
			collectRefs(value, out)
		}
	case []any:
		for _, item := range n {
			collectRefs(item, out)
		}
	}
}

// rewriteRefs rewrites #/components/schemas/X pointers to #/$defs/X in
// place. xtask does this textually on the serialized document
// (main.rs:358-360), which also catches discriminator.mapping values, so
// every string value with the prefix is rewritten — not just $ref keys.
func rewriteRefs(node any) {
	switch n := node.(type) {
	case map[string]any:
		for key, value := range n {
			if s, ok := value.(string); ok && strings.HasPrefix(s, refPrefix) {
				n[key] = "#/$defs/" + strings.TrimPrefix(s, refPrefix)
				continue
			}
			rewriteRefs(value)
		}
	case []any:
		for i, item := range n {
			if s, ok := item.(string); ok && strings.HasPrefix(s, refPrefix) {
				n[i] = "#/$defs/" + strings.TrimPrefix(s, refPrefix)
				continue
			}
			rewriteRefs(item)
		}
	}
}

// rewriteNullableFlag ports the xtask nullable-flag rule (main.rs:504-528):
// honour OpenAPI-3.0 `nullable: true` by rewriting to a null union. The
// branch order matches xtask: type, then oneOf/anyOf, then $ref. The one
// addition (documented in the package comment): a co-resident `enum` gains
// a null value, since enum asserts independently of type.
func rewriteNullableFlag(node any) {
	switch n := node.(type) {
	case map[string]any:
		if b, ok := n["nullable"].(bool); ok && b {
			delete(n, "nullable")
			if t, ok := n["type"]; ok {
				types := asArray(t)
				if !slices.Contains(types, any("null")) {
					types = append(types, "null")
				}
				n["type"] = types
			} else if uni := firstKey(n, "oneOf", "anyOf"); uni != "" {
				if arr, ok := n[uni].([]any); ok {
					n[uni] = append(arr, map[string]any{"type": "null"})
				}
			} else if r, ok := n["$ref"]; ok {
				delete(n, "$ref")
				n["oneOf"] = []any{
					map[string]any{"$ref": r},
					map[string]any{"type": "null"},
				}
			}
			if e, ok := n["enum"].([]any); ok && !slices.Contains(e, nil) {
				n["enum"] = append(e, nil)
			}
		}
		for _, value := range n {
			rewriteNullableFlag(value)
		}
	case []any:
		for _, item := range n {
			rewriteNullableFlag(item)
		}
	}
}

func asArray(v any) []any {
	if a, ok := v.([]any); ok {
		return a
	}
	return []any{v}
}

func firstKey(m map[string]any, keys ...string) string {
	for _, k := range keys {
		if _, ok := m[k]; ok {
			return k
		}
	}
	return ""
}
