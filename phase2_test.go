package main

import (
	"strings"
	"testing"
)

// --- Phase 2: strict mode (contract oracle) + spec-anchored fixtures ---

func strictConfig() *Config {
	cfg := defaultConfig()
	cfg.Anthropic.Strict = true
	return cfg
}

func TestStrictRejectsUnknownTopLevelField(t *testing.T) {
	srv := testServer(strictConfig())
	defer srv.Close()

	body := `{
		"model": "claude-3-haiku-20240307",
		"max_tokens": 50,
		"messages": [{"role": "user", "content": "hi"}],
		"frequency_penalty": 0.5
	}`

	resp := postAnthropic(t, srv.URL, body)
	if resp.StatusCode != 400 {
		t.Fatalf("strict mode must reject unknown fields like the real API, got %d", resp.StatusCode)
	}
	out := decodeBody(t, resp)
	msg := out["error"].(map[string]any)["message"].(string)
	if !strings.Contains(msg, "frequency_penalty") {
		t.Fatalf("rejection must name the field, got: %s", msg)
	}
}

func TestStrictRejectsMissingRequiredAndBadValues(t *testing.T) {
	srv := testServer(strictConfig())
	defer srv.Close()

	cases := []struct {
		name string
		body string
		want string
	}{
		{"missing max_tokens", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, "max_tokens"},
		{"missing model", `{"max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`, "model"},
		{"empty messages", `{"model":"m","max_tokens":10,"messages":[]}`, "messages"},
		{"temperature out of range", `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"temperature":1.5}`, "temperature"},
	}

	for _, tc := range cases {
		resp := postAnthropic(t, srv.URL, tc.body)
		if resp.StatusCode != 400 {
			t.Fatalf("%s: expected 400, got %d", tc.name, resp.StatusCode)
		}
		out := decodeBody(t, resp)
		msg := out["error"].(map[string]any)["message"].(string)
		if !strings.Contains(msg, tc.want) {
			t.Fatalf("%s: message should mention %q, got: %s", tc.name, tc.want, msg)
		}
	}
}

func TestStrictAcceptsProxyShapedToolRoundTrip(t *testing.T) {
	// Exactly what nanollm's transform emits for a tool round-trip — the
	// whole point of strict mode is validating that output.
	srv := testServer(strictConfig())
	defer srv.Close()

	body := `{
		"model": "claude-3-haiku-20240307",
		"max_tokens": 100,
		"system": "Be helpful.",
		"temperature": 0.7,
		"tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
		"tool_choice": {"type": "auto"},
		"messages": [
			{"role": "user", "content": "weather?"},
			{"role": "assistant", "content": [
				{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}}
			]},
			{"role": "user", "content": [
				{"type": "tool_result", "tool_use_id": "toolu_1", "content": "18C"}
			]}
		]
	}`

	resp := postAnthropic(t, srv.URL, body)
	if resp.StatusCode != 200 {
		out := decodeBody(t, resp)
		t.Fatalf("proxy-shaped request must pass strict validation, got %d: %v", resp.StatusCode, out)
	}
	resp.Body.Close()
}

func TestStrictRejectsNestedOpenAiShapes(t *testing.T) {
	// Strict mode validates BELOW the top level: OpenAI tool wrappers,
	// string tool_choice, OpenAI roles, and image_url blocks — the shapes a
	// proxy transform must reshape away — are rejected with the path named.
	srv := testServer(strictConfig())
	defer srv.Close()

	base := `"model":"m","max_tokens":10`
	userMsg := `{"role":"user","content":"hi"}`
	cases := []struct {
		name string
		body string
		want string
	}{
		{
			"OpenAI tool wrapper",
			`{` + base + `,"messages":[` + userMsg + `],"tools":[{"type":"function","function":{"name":"f"}}]}`,
			"function",
		},
		{
			"schema-less client tool",
			`{` + base + `,"messages":[` + userMsg + `],"tools":[{"name":"f"}]}`,
			"input_schema",
		},
		{
			"string tool_choice",
			`{` + base + `,"messages":[` + userMsg + `],"tool_choice":"auto"}`,
			"tool_choice",
		},
		{
			"OpenAI system role in messages",
			`{` + base + `,"messages":[{"role":"system","content":"x"},` + userMsg + `]}`,
			"role",
		},
		{
			"OpenAI tool role in messages",
			`{` + base + `,"messages":[{"role":"tool","content":"x"}]}`,
			"role",
		},
		{
			"image_url block",
			`{` + base + `,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"http://x"}}]}]}`,
			"image_url",
		},
		{
			"tool_use missing id",
			`{` + base + `,"messages":[{"role":"assistant","content":[{"type":"tool_use","name":"f","input":{}}]}]}`,
			"id",
		},
		{
			"thinking missing signature",
			`{` + base + `,"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"hm"}]}]}`,
			"signature",
		},
	}

	for _, tc := range cases {
		resp := postAnthropic(t, srv.URL, tc.body)
		if resp.StatusCode != 400 {
			t.Fatalf("%s: expected 400, got %d", tc.name, resp.StatusCode)
		}
		out := decodeBody(t, resp)
		msg := out["error"].(map[string]any)["message"].(string)
		if !strings.Contains(msg, tc.want) {
			t.Fatalf("%s: message should mention %q, got: %s", tc.name, tc.want, msg)
		}
	}
}

func TestStrictAcceptsTypedServerToolsAndUncommonBlocks(t *testing.T) {
	// Bounded depth: typed server tools and uncommon typed blocks pass on
	// shape — strict mode does not re-implement the full union.
	srv := testServer(strictConfig())
	defer srv.Close()

	body := `{
		"model": "m", "max_tokens": 10,
		"tools": [{"type": "web_search_20250305", "name": "web_search"}],
		"messages": [{"role": "user", "content": [
			{"type": "text", "text": "hi"},
			{"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}
		]}]
	}`
	resp := postAnthropic(t, srv.URL, body)
	if resp.StatusCode != 200 {
		out := decodeBody(t, resp)
		t.Fatalf("typed tools / typed blocks must pass, got %d: %v", resp.StatusCode, out)
	}
	resp.Body.Close()
}

func TestStreamingFixturesCarrySpecShapes(t *testing.T) {
	// message_delta carries the full MessageDelta + MessageDeltaUsage shapes,
	// and thinking blocks close with a signature_delta like the real API.
	cfg := defaultConfig()
	cfg.Anthropic.ReasoningTokens = 9
	srv := testServer(cfg)
	defer srv.Close()

	resp := postAnthropic(t, srv.URL, anthropicBody(true))
	defer resp.Body.Close()
	raw := new(strings.Builder)
	buf := make([]byte, 4096)
	for {
		n, err := resp.Body.Read(buf)
		raw.Write(buf[:n])
		if err != nil {
			break
		}
	}
	text := raw.String()

	// stop_details / inference_geo / output_tokens_details are REQUIRED
	// nullable fields per the pinned spec (Message.required, Usage.required,
	// MessageDelta.required, MessageDeltaUsage.required) and must be present
	// on the default wire.
	for _, want := range []string{
		`"signature_delta"`,
		`"container":null`,
		`"stop_details":null`,
		`"input_tokens":`,
		`"server_tool_use":null`,
		`"inference_geo":null`,
		`"output_tokens_details":null`,
		"event: ping",
		`{"type":"ping"}`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("stream missing spec shape %s:\n%s", want, text)
		}
	}

	// Genuinely-unknown probe fields appear only under the
	// emit_nonstandard_fields fault knob.
	for _, banned := range []string{"x_mock_unknown_field", "x_mock_unknown_usage_field"} {
		if strings.Contains(text, banned) {
			t.Fatalf("stream must not carry nonstandard field %s by default:\n%s", banned, text)
		}
	}
}

func TestLenientModeStillAcceptsUnknownFields(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	body := `{
		"model": "m", "max_tokens": 10,
		"messages": [{"role": "user", "content": "hi"}],
		"x_whatever": true
	}`
	resp := postAnthropic(t, srv.URL, body)
	if resp.StatusCode != 200 {
		t.Fatalf("default mode stays lenient, got %d", resp.StatusCode)
	}
	resp.Body.Close()
}

func TestResponseCarriesSpecRequiredFields(t *testing.T) {
	// Real responses always include citations (text blocks), caller
	// (tool_use blocks), stop_details/container, and the full usage set —
	// spec Message.required / Usage.required (Q3 spike finding).
	cfg := defaultConfig()
	cfg.Anthropic.ToolUseResponse = true
	cfg.Anthropic.CacheReadTokens = 7
	srv := testServer(cfg)
	defer srv.Close()

	out := decodeBody(t, postAnthropic(t, srv.URL, anthropicBody(false)))

	for _, key := range []string{"stop_details", "container"} {
		if _, ok := out[key]; !ok {
			t.Fatalf("message must carry %s (spec Message.required)", key)
		}
	}
	if _, ok := out["x_mock_unknown_field"]; ok {
		t.Fatal("unknown-field probe must be absent by default")
	}

	content := out["content"].([]any)
	text := content[0].(map[string]any)
	if _, ok := text["citations"]; !ok {
		t.Fatalf("text block must carry citations, got %v", text)
	}
	toolUse := content[1].(map[string]any)
	caller, ok := toolUse["caller"].(map[string]any)
	if !ok || caller["type"] != "direct" {
		t.Fatalf("tool_use block must carry caller {type: direct}, got %v", toolUse)
	}

	usage := out["usage"].(map[string]any)
	for _, key := range []string{
		"cache_read_input_tokens", "cache_creation_input_tokens",
		"cache_creation", "server_tool_use", "service_tier",
		"inference_geo", "output_tokens_details",
	} {
		if _, ok := usage[key]; !ok {
			t.Fatalf("usage must carry %s (spec Usage.required), got %v", key, usage)
		}
	}
	if _, ok := usage["x_mock_unknown_usage_field"]; ok {
		t.Fatalf("unknown-field probe must be absent from usage by default, got %v", usage)
	}
	if usage["cache_read_input_tokens"] != float64(7) {
		t.Fatalf("cache knob must drive cache_read_input_tokens, got %v", usage)
	}
}
