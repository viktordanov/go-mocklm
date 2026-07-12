package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// --- Spec-shape fidelity (G3/G8/G9/G10, G1/G2 gating) +
// harness knobs (X-MockLM-Fault, fail_first_n, disconnect_after_event) ---

// collectDataLines reads an OpenAI-style SSE stream and returns the payload
// of every "data: " line, including the [DONE] sentinel.
func collectDataLines(t *testing.T, srv, body string, headers map[string]string) []string {
	t.Helper()
	resp, err := postJSON(srv+"/v1/chat/completions", body, headers)
	if err != nil {
		t.Fatalf("POST /v1/chat/completions failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	var lines []string
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		if line := scanner.Text(); strings.HasPrefix(line, "data: ") {
			lines = append(lines, strings.TrimPrefix(line, "data: "))
		}
	}
	return lines
}

// readStreamUntilClosed drains an SSE response body, tolerating the
// connection-reset error that fault-injected streams end with.
func readStreamUntilClosed(t *testing.T, url, body string) string {
	t.Helper()
	resp := postAnthropic(t, url, body)
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
	return raw.String()
}

func openaiStreamBodyWithUsage() string {
	return `{"model":"gpt-4","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"Hello world"}]}`
}

func TestOpenAIIncludeUsageEmitsRealTrailerShape(t *testing.T) {
	cfg := defaultConfig()
	cfg.OpenAI.Tokens = 3
	srv := testServer(cfg)
	defer srv.Close()

	lines := collectDataLines(t, srv.URL, openaiStreamBodyWithUsage(), nil)

	// role + 3 content + finish + usage trailer + [DONE]
	if len(lines) != 7 {
		t.Fatalf("expected 7 data lines (incl. usage trailer), got %d: %v", len(lines), lines)
	}
	if lines[len(lines)-1] != "[DONE]" {
		t.Fatalf("expected [DONE] last, got %q", lines[len(lines)-1])
	}

	// Every chunk before the trailer carries "usage": null.
	for i := 0; i < len(lines)-2; i++ {
		var chunk map[string]any
		if err := json.Unmarshal([]byte(lines[i]), &chunk); err != nil {
			t.Fatalf("chunk %d unparseable: %v", i, err)
		}
		v, ok := chunk["usage"]
		if !ok || v != nil {
			t.Fatalf("chunk %d must carry usage:null, got %v", i, chunk)
		}
	}

	// The trailer has empty choices and the usage totals.
	var trailer map[string]any
	if err := json.Unmarshal([]byte(lines[len(lines)-2]), &trailer); err != nil {
		t.Fatalf("trailer unparseable: %v", err)
	}
	choices, ok := trailer["choices"].([]any)
	if !ok || len(choices) != 0 {
		t.Fatalf("trailer choices must be an empty array, got %v", trailer["choices"])
	}
	usage, ok := trailer["usage"].(map[string]any)
	if !ok {
		t.Fatalf("trailer must carry the usage object, got %v", trailer)
	}
	if usage["completion_tokens"] != float64(3) {
		t.Fatalf("trailer usage completion_tokens should be 3, got %v", usage)
	}
	// Detail sub-objects are unconditional (G9).
	for _, key := range []string{"prompt_tokens_details", "completion_tokens_details"} {
		if _, ok := usage[key].(map[string]any); !ok {
			t.Fatalf("trailer usage must carry %s unconditionally, got %v", key, usage)
		}
	}

	// The finish chunk itself carries usage:null, not the totals.
	var finish map[string]any
	json.Unmarshal([]byte(lines[len(lines)-3]), &finish)
	if fr := finish["choices"].([]any)[0].(map[string]any)["finish_reason"]; fr != "stop" {
		t.Fatalf("chunk before trailer must be the finish chunk, got %v", finish)
	}
}

func TestOpenAIStreamOmitsUsageWithoutIncludeUsage(t *testing.T) {
	cfg := defaultConfig()
	cfg.OpenAI.Tokens = 3
	srv := testServer(cfg)
	defer srv.Close()

	lines := collectDataLines(t, srv.URL, openaiChatBody(true), nil)

	// role + 3 content + finish + [DONE]: no trailer
	if len(lines) != 6 {
		t.Fatalf("expected 6 data lines (no usage trailer), got %d: %v", len(lines), lines)
	}
	for i := 0; i < len(lines)-1; i++ {
		var chunk map[string]any
		json.Unmarshal([]byte(lines[i]), &chunk)
		if _, ok := chunk["usage"]; ok {
			t.Fatalf("chunk %d must not carry a usage key without include_usage, got %v", i, chunk)
		}
	}
}

func TestOpenAILegacyStreamUsageCompatFlag(t *testing.T) {
	cfg := defaultConfig()
	cfg.OpenAI.Tokens = 3
	cfg.OpenAI.LegacyStreamUsage = true
	srv := testServer(cfg)
	defer srv.Close()

	// include_usage is ignored under the compat flag.
	lines := collectDataLines(t, srv.URL, openaiStreamBodyWithUsage(), nil)

	// role + 3 content + finish + [DONE]: usage rides the finish chunk
	if len(lines) != 6 {
		t.Fatalf("expected 6 data lines (legacy shape), got %d: %v", len(lines), lines)
	}
	var finish map[string]any
	json.Unmarshal([]byte(lines[4]), &finish)
	usage, ok := finish["usage"].(map[string]any)
	if !ok {
		t.Fatalf("legacy shape must put usage on the finish chunk, got %v", finish)
	}
	if usage["completion_tokens"] != float64(3) {
		t.Fatalf("legacy finish usage completion_tokens should be 3, got %v", usage)
	}
	var role map[string]any
	json.Unmarshal([]byte(lines[0]), &role)
	if _, ok := role["usage"]; ok {
		t.Fatalf("legacy shape has no usage key on non-finish chunks, got %v", role)
	}
}

func TestOpenAIChatAlwaysPresentFields(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
	if err != nil {
		t.Fatalf("POST failed: %v", err)
	}
	out := decodeBody(t, resp)

	if out["service_tier"] != "default" {
		t.Fatalf("response must carry service_tier, got %v", out["service_tier"])
	}

	message := out["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	refusal, ok := message["refusal"]
	if !ok || refusal != nil {
		t.Fatalf("message must carry refusal:null, got %v", message)
	}
	annotations, ok := message["annotations"].([]any)
	if !ok || len(annotations) != 0 {
		t.Fatalf("message must carry annotations:[], got %v", message)
	}

	// Usage details are unconditional even with zero reasoning tokens (G9).
	usage := out["usage"].(map[string]any)
	ptd, ok := usage["prompt_tokens_details"].(map[string]any)
	if !ok {
		t.Fatalf("usage must carry prompt_tokens_details unconditionally, got %v", usage)
	}
	if ptd["cached_tokens"] != float64(0) {
		t.Fatalf("expected cached_tokens 0, got %v", ptd)
	}
	ctd, ok := usage["completion_tokens_details"].(map[string]any)
	if !ok {
		t.Fatalf("usage must carry completion_tokens_details unconditionally, got %v", usage)
	}
	if ctd["reasoning_tokens"] != float64(0) {
		t.Fatalf("expected reasoning_tokens 0, got %v", ctd)
	}
}

func TestAnthropicNonstandardFieldsKnob(t *testing.T) {
	cfg := defaultConfig()
	cfg.Anthropic.EmitNonstandardFields = true
	srv := testServer(cfg)
	defer srv.Close()

	out := decodeBody(t, postAnthropic(t, srv.URL, anthropicBody(false)))

	// The knob injects genuinely-unknown probe fields...
	if _, ok := out["x_mock_unknown_field"]; !ok {
		t.Fatalf("emit_nonstandard_fields must inject x_mock_unknown_field, got %v", out)
	}
	usage := out["usage"].(map[string]any)
	if _, ok := usage["x_mock_unknown_usage_field"]; !ok {
		t.Fatalf("emit_nonstandard_fields must inject x_mock_unknown_usage_field, got %v", usage)
	}

	// ...while the spec-required nullable fields stay present regardless.
	if _, ok := out["stop_details"]; !ok {
		t.Fatalf("stop_details is spec-required and must always be present, got %v", out)
	}
	for _, key := range []string{"inference_geo", "output_tokens_details"} {
		if _, ok := usage[key]; !ok {
			t.Fatalf("usage.%s is spec-required and must always be present, got %v", key, usage)
		}
	}
}

func TestSuppressPingEventsKnob(t *testing.T) {
	cfg := defaultConfig()
	cfg.Anthropic.SuppressPingEvents = true
	srv := testServer(cfg)
	defer srv.Close()

	text := readStreamUntilClosed(t, srv.URL, anthropicBody(true))
	if strings.Contains(text, "event: ping") {
		t.Fatalf("suppress_ping_events must omit the typed ping:\n%s", text)
	}
	// The rest of the stream is intact.
	for _, want := range []string{"event: message_start", "event: message_delta", "event: message_stop"} {
		if !strings.Contains(text, want) {
			t.Fatalf("stream missing %s with pings suppressed:\n%s", want, text)
		}
	}
}

func TestAnthropicErrorEnvelopeMatchesSpec(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	// status → expected error union member (ErrorResponse discriminator).
	cases := map[int]string{
		529: "overloaded_error",
		500: "api_error",
		429: "rate_limit_error",
		400: "invalid_request_error",
	}
	for status, wantType := range cases {
		headers := map[string]string{
			"x-api-key":         "test-key",
			"anthropic-version": "2023-06-01",
			"X-MockLM-Fault":    fmt.Sprintf(`{"error_rate":1.0,"error_status":%d}`, status),
		}
		resp, err := postJSON(srv.URL+"/v1/messages", anthropicBody(false), headers)
		if err != nil {
			t.Fatalf("POST failed: %v", err)
		}
		if resp.StatusCode != status {
			t.Fatalf("expected %d, got %d", status, resp.StatusCode)
		}
		out := decodeBody(t, resp)

		// ErrorResponse.required = {type, error, request_id}
		if out["type"] != "error" {
			t.Fatalf("envelope type must be \"error\", got %v", out)
		}
		if _, ok := out["request_id"].(string); !ok {
			t.Fatalf("envelope must carry a request_id (ErrorResponse.required), got %v", out)
		}
		errObj := out["error"].(map[string]any)
		if errObj["type"] != wantType {
			t.Fatalf("status %d must map to %s (spec error union), got %v", status, wantType, errObj["type"])
		}
		if _, ok := errObj["message"].(string); !ok {
			t.Fatalf("error must carry a message, got %v", errObj)
		}
	}
}

func TestOpenAIErrorEnvelopeMatchesSpec(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	headers := map[string]string{"X-MockLM-Fault": `{"error_rate":1.0,"error_status":503}`}
	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), headers)
	if err != nil {
		t.Fatalf("POST failed: %v", err)
	}
	if resp.StatusCode != 503 {
		t.Fatalf("expected 503, got %d", resp.StatusCode)
	}
	out := decodeBody(t, resp)

	// Error.required = {type, message, param, code} — param nullable but
	// the key must be present.
	errObj := out["error"].(map[string]any)
	for _, key := range []string{"type", "message", "param", "code"} {
		if _, ok := errObj[key]; !ok {
			t.Fatalf("error must carry %s (spec Error.required), got %v", key, errObj)
		}
	}
	if errObj["param"] != nil {
		t.Fatalf("param should be null for injected errors, got %v", errObj["param"])
	}
	if errObj["type"] != "server_error" {
		t.Fatalf("5xx should map to server_error, got %v", errObj["type"])
	}
}

func TestFaultHeaderTargetsSingleRequest(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	// Faulted request: per-request error injection via header.
	headers := map[string]string{"X-MockLM-Fault": `{"error_rate":1.0,"error_status":503}`}
	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), headers)
	if err != nil {
		t.Fatalf("POST failed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 503 {
		t.Fatalf("header-targeted request must fail with 503, got %d", resp.StatusCode)
	}

	// A concurrent healthy request on the same server is unaffected.
	resp, err = postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
	if err != nil {
		t.Fatalf("POST failed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("headerless request must stay healthy, got %d", resp.StatusCode)
	}

	// Invalid header JSON is a loud 400, not a silent ignore.
	headers = map[string]string{"X-MockLM-Fault": `{not json`}
	resp, err = postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), headers)
	if err != nil {
		t.Fatalf("POST failed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 400 {
		t.Fatalf("invalid X-MockLM-Fault must 400, got %d", resp.StatusCode)
	}
}

func TestFaultHeaderWorksOnAnthropic(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	req := anthropicBody(false)
	resp := postAnthropic(t, srv.URL, req)
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("healthy request should pass, got %d", resp.StatusCode)
	}

	// postAnthropic has fixed headers, so drive the fault via a raw request.
	headers := map[string]string{
		"x-api-key":         "test-key",
		"anthropic-version": "2023-06-01",
		"X-MockLM-Fault":    `{"error_rate":1.0,"error_status":529}`,
	}
	resp2, err := postJSON(srv.URL+"/v1/messages", req, headers)
	if err != nil {
		t.Fatalf("POST failed: %v", err)
	}
	resp2.Body.Close()
	if resp2.StatusCode != 529 {
		t.Fatalf("header-targeted Anthropic request must fail with 529, got %d", resp2.StatusCode)
	}
}

func TestFailFirstNIsDeterministic(t *testing.T) {
	cfg := defaultConfig()
	cfg.OpenAI.FailFirstN = 2
	cfg.OpenAI.ErrorStatus = 500
	srv, state := testServerWithState(cfg)
	defer srv.Close()

	statuses := []int{}
	for i := 0; i < 3; i++ {
		resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
		if err != nil {
			t.Fatalf("POST %d failed: %v", i, err)
		}
		resp.Body.Close()
		statuses = append(statuses, resp.StatusCode)
	}
	if statuses[0] != 500 || statuses[1] != 500 || statuses[2] != 200 {
		t.Fatalf("fail_first_n=2 must yield [500 500 200], got %v", statuses)
	}

	// Config reset restarts the counter.
	state.Reset()
	cfgSnap, _ := state.Config()
	state.Update(cfgSnap.OpenAI, cfgSnap.Anthropic, cfgSnap.Bedrock, "")
	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
	if err != nil {
		t.Fatalf("POST after reset failed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 500 {
		t.Fatalf("counter must reset with the config, got %d", resp.StatusCode)
	}
}

func TestFailFirstNViaFaultHeader(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	headers := map[string]string{"X-MockLM-Fault": `{"fail_first_n":1,"error_status":503}`}
	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), headers)
	if err != nil {
		t.Fatalf("POST failed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 503 {
		t.Fatalf("first header-driven request must fail, got %d", resp.StatusCode)
	}

	resp, err = postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), headers)
	if err != nil {
		t.Fatalf("POST failed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("second header-driven request must succeed, got %d", resp.StatusCode)
	}
}

func TestDisconnectAfterEvent(t *testing.T) {
	cases := []struct {
		event      string
		wantLast   string
		mustNotSee []string
	}{
		// Cut after message_delta: client has a stop_reason but no message_stop.
		{"message_delta", "event: message_delta", []string{"event: message_stop"}},
		// Cut after content_block_start: block opened, no delta ever arrives.
		{"content_block_start", "event: content_block_start", []string{"text_delta", "event: message_stop"}},
	}

	for _, tc := range cases {
		cfg := defaultConfig()
		cfg.Anthropic.DisconnectAfterEvent = tc.event
		srv := testServer(cfg)

		text := readStreamUntilClosed(t, srv.URL, anthropicBody(true))
		srv.Close()

		if !strings.Contains(text, tc.wantLast) {
			t.Fatalf("%s: stream must include the cut event, got:\n%s", tc.event, text)
		}
		for _, banned := range tc.mustNotSee {
			if strings.Contains(text, banned) {
				t.Fatalf("%s: stream must be cut before %s, got:\n%s", tc.event, banned, text)
			}
		}
	}
}
