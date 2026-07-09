package main

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
)

func TestAdminGetConfig(t *testing.T) {
	srv, _ := testServerWithState(defaultConfig())
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/admin/config")
	if err != nil {
		t.Fatalf("GET /admin/config failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	var body map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode failed: %v", err)
	}

	if body["active_preset"] != "" {
		t.Errorf("expected empty active_preset, got %v", body["active_preset"])
	}

	openai, ok := body["openai"].(map[string]any)
	if !ok {
		t.Fatal("missing openai config")
	}

	if int(openai["tokens"].(float64)) != 5 {
		t.Errorf("expected openai tokens=5, got %v", openai["tokens"])
	}
}

func TestAdminPutConfig(t *testing.T) {
	srv, _ := testServerWithState(defaultConfig())
	defer srv.Close()

	payload := `{"openai":{"tokens":42,"error_rate":0.5,"error_status":503},"anthropic":{"tokens":30}}`

	req, err := http.NewRequest("PUT", srv.URL+"/admin/config", strings.NewReader(payload))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("PUT /admin/config failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	var body map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode failed: %v", err)
	}

	if body["status"] != "updated" {
		t.Errorf("expected status=updated, got %v", body["status"])
	}

	if body["active_preset"] != "custom" {
		t.Errorf("expected active_preset=custom, got %v", body["active_preset"])
	}

	// Verify the config was actually applied
	getResp, err := http.Get(srv.URL + "/admin/config")
	if err != nil {
		t.Fatalf("GET /admin/config failed: %v", err)
	}
	defer getResp.Body.Close()

	var cfg map[string]any
	json.NewDecoder(getResp.Body).Decode(&cfg)
	openai := cfg["openai"].(map[string]any)

	if int(openai["tokens"].(float64)) != 42 {
		t.Errorf("expected updated tokens=42, got %v", openai["tokens"])
	}

	if openai["error_rate"].(float64) != 0.5 {
		t.Errorf("expected error_rate=0.5, got %v", openai["error_rate"])
	}
}

func TestAdminPutPreset(t *testing.T) {
	srv, _ := testServerWithState(defaultConfig())
	defer srv.Close()

	req, err := http.NewRequest("PUT", srv.URL+"/admin/preset/openai-errors", nil)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("PUT /admin/preset failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	var body map[string]any
	json.NewDecoder(resp.Body).Decode(&body)

	if body["status"] != "activated" {
		t.Errorf("expected status=activated, got %v", body["status"])
	}

	if body["active_preset"] != "openai-errors" {
		t.Errorf("expected active_preset=openai-errors, got %v", body["active_preset"])
	}

	// Verify the preset was applied: OpenAI should now return 503
	chatResp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
	if err != nil {
		t.Fatalf("POST /v1/chat/completions failed: %v", err)
	}
	defer chatResp.Body.Close()

	if chatResp.StatusCode != 503 {
		t.Errorf("expected 503 from openai-errors preset, got %d", chatResp.StatusCode)
	}
}

func TestAdminPresetNotFound(t *testing.T) {
	srv, _ := testServerWithState(defaultConfig())
	defer srv.Close()

	req, err := http.NewRequest("PUT", srv.URL+"/admin/preset/nonexistent", nil)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("PUT /admin/preset failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 404 {
		t.Fatalf("expected 404, got %d", resp.StatusCode)
	}
}

func TestAdminReset(t *testing.T) {
	srv, state := testServerWithState(defaultConfig())
	defer srv.Close()

	// Change config
	state.Update(
		ProviderConfig{Tokens: 99, ErrorRate: 1.0, ErrorStatus: 503},
		ProviderConfig{Tokens: 99, ErrorStatus: 500},
		ProviderConfig{Tokens: 99, ErrorStatus: 500},
		"test",
	)

	// Reset
	req, err := http.NewRequest("POST", srv.URL+"/admin/reset", nil)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST /admin/reset failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	// Verify config was restored: tokens should be back to 5
	getResp, err := http.Get(srv.URL + "/admin/config")
	if err != nil {
		t.Fatalf("GET /admin/config failed: %v", err)
	}
	defer getResp.Body.Close()

	var cfg map[string]any
	json.NewDecoder(getResp.Body).Decode(&cfg)

	if cfg["active_preset"] != "" {
		t.Errorf("expected empty active_preset after reset, got %v", cfg["active_preset"])
	}

	openai := cfg["openai"].(map[string]any)
	if int(openai["tokens"].(float64)) != 5 {
		t.Errorf("expected tokens=5 after reset, got %v", openai["tokens"])
	}
}

func TestAdminGetPresets(t *testing.T) {
	srv, _ := testServerWithState(defaultConfig())
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/admin/presets")
	if err != nil {
		t.Fatalf("GET /admin/presets failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	var body map[string]any
	json.NewDecoder(resp.Body).Decode(&body)

	presets, ok := body["presets"].([]any)
	if !ok {
		t.Fatal("missing presets array")
	}

	if len(presets) != 18 {
		t.Errorf("expected 18 presets, got %d", len(presets))
	}

	// Verify each preset has name and description
	for _, p := range presets {
		preset := p.(map[string]any)
		if preset["name"] == nil || preset["name"] == "" {
			t.Error("preset missing name")
		}

		if preset["description"] == nil || preset["description"] == "" {
			t.Error("preset missing description")
		}
	}
}

func TestRuntimeConfigSwitch(t *testing.T) {
	srv, state := testServerWithState(defaultConfig())
	defer srv.Close()

	// Phase 1: healthy — request should succeed
	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
	if err != nil {
		t.Fatalf("healthy request failed: %v", err)
	}
	resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("expected 200 in healthy phase, got %d", resp.StatusCode)
	}

	// Phase 2: switch to openai-disconnect preset
	presets := builtinPresets()
	preset := presets["openai-disconnect"]
	openai := preset.OpenAI
	anthropic := preset.Anthropic
	bedrock := preset.Bedrock
	applyProviderDefaults(&openai)
	applyProviderDefaults(&anthropic)
	applyProviderDefaults(&bedrock)
	state.Update(openai, anthropic, bedrock, "openai-disconnect")

	// Streaming request should disconnect early (no [DONE])
	resp, err = postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(true), nil)
	if err != nil {
		t.Fatalf("disconnect request failed: %v", err)
	}

	data, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	body := string(data)

	if strings.Contains(body, "[DONE]") {
		t.Error("expected no [DONE] after disconnect, but found it")
	}

	// Phase 3: reset back to healthy
	state.Reset()

	resp, err = postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(true), nil)
	if err != nil {
		t.Fatalf("post-reset request failed: %v", err)
	}

	data, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	body = string(data)

	if !strings.Contains(body, "[DONE]") {
		t.Error("expected [DONE] after reset, but not found")
	}
}

func TestAdminGetRequests(t *testing.T) {
	srv, _ := testServerWithState(defaultConfig())
	defer srv.Close()

	// Initially empty
	resp, err := http.Get(srv.URL + "/admin/requests")
	if err != nil {
		t.Fatalf("GET /admin/requests failed: %v", err)
	}
	defer resp.Body.Close()

	var body map[string]any
	json.NewDecoder(resp.Body).Decode(&body)

	requests := body["requests"].([]any)
	if len(requests) != 0 {
		t.Errorf("expected 0 recorded requests, got %d", len(requests))
	}

	// Make an Anthropic request to generate a recording
	headers := map[string]string{
		"x-api-key":         "test-key",
		"anthropic-version": "2023-06-01",
	}
	chatResp, err := postJSON(srv.URL+"/v1/messages", anthropicBody(false), headers)
	if err != nil {
		t.Fatalf("POST /v1/messages failed: %v", err)
	}
	chatResp.Body.Close()

	// Should now have 1 recorded request
	resp2, err := http.Get(srv.URL + "/admin/requests")
	if err != nil {
		t.Fatalf("GET /admin/requests failed: %v", err)
	}
	defer resp2.Body.Close()

	var body2 map[string]any
	json.NewDecoder(resp2.Body).Decode(&body2)

	requests2 := body2["requests"].([]any)
	if len(requests2) != 1 {
		t.Fatalf("expected 1 recorded request, got %d", len(requests2))
	}

	// Verify recorded request structure
	rec := requests2[0].(map[string]any)
	if rec["provider"] != "anthropic" {
		t.Errorf("expected provider=anthropic, got %v", rec["provider"])
	}
	if rec["method"] != "POST" {
		t.Errorf("expected method=POST, got %v", rec["method"])
	}

	// Verify body is the raw JSON we sent
	recBody := rec["body"].(map[string]any)
	if recBody["model"] != "claude-3-haiku-20240307" {
		t.Errorf("expected model in recorded body, got %v", recBody["model"])
	}
}

func TestAdminClearRequests(t *testing.T) {
	srv, _ := testServerWithState(defaultConfig())
	defer srv.Close()

	// Make an Anthropic request
	headers := map[string]string{
		"x-api-key":         "test-key",
		"anthropic-version": "2023-06-01",
	}
	chatResp, err := postJSON(srv.URL+"/v1/messages", anthropicBody(false), headers)
	if err != nil {
		t.Fatalf("POST /v1/messages failed: %v", err)
	}
	chatResp.Body.Close()

	// Clear
	req, err := http.NewRequest("POST", srv.URL+"/admin/requests/clear", nil)
	if err != nil {
		t.Fatal(err)
	}
	clearResp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST /admin/requests/clear failed: %v", err)
	}
	defer clearResp.Body.Close()

	if clearResp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d", clearResp.StatusCode)
	}

	// Verify empty
	resp, err := http.Get(srv.URL + "/admin/requests")
	if err != nil {
		t.Fatalf("GET /admin/requests failed: %v", err)
	}
	defer resp.Body.Close()

	var body map[string]any
	json.NewDecoder(resp.Body).Decode(&body)
	requests := body["requests"].([]any)
	if len(requests) != 0 {
		t.Errorf("expected 0 after clear, got %d", len(requests))
	}
}

func TestDeterministicAnthropicPreset(t *testing.T) {
	cfg := defaultConfig()
	cfg.Anthropic.Deterministic = true
	cfg.Anthropic.Tokens = 5
	srv := testServer(cfg)
	defer srv.Close()

	headers := map[string]string{
		"x-api-key":         "test-key",
		"anthropic-version": "2023-06-01",
	}

	// Two requests should produce identical responses
	resp1, _ := postJSON(srv.URL+"/v1/messages", anthropicBody(false), headers)
	var body1 map[string]any
	json.NewDecoder(resp1.Body).Decode(&body1)
	resp1.Body.Close()

	resp2, _ := postJSON(srv.URL+"/v1/messages", anthropicBody(false), headers)
	var body2 map[string]any
	json.NewDecoder(resp2.Body).Decode(&body2)
	resp2.Body.Close()

	// IDs should be identical
	if body1["id"] != "msg_mock_deterministic" {
		t.Errorf("expected deterministic ID, got %v", body1["id"])
	}
	if body1["id"] != body2["id"] {
		t.Errorf("expected same IDs, got %v and %v", body1["id"], body2["id"])
	}

	// Content should be identical
	content1 := body1["content"].([]any)[0].(map[string]any)["text"].(string)
	content2 := body2["content"].([]any)[0].(map[string]any)["text"].(string)
	if content1 != content2 {
		t.Errorf("expected identical content, got:\n%s\nvs\n%s", content1, content2)
	}

	// Content should match expected deterministic output (first 5 words from wordList)
	expected := "Quantum neural reasoning context inference."
	if content1 != expected {
		t.Errorf("expected content %q, got %q", expected, content1)
	}
}

func TestDeterministicAnthropicStreamPreset(t *testing.T) {
	cfg := defaultConfig()
	cfg.Anthropic.Deterministic = true
	cfg.Anthropic.Tokens = 3
	srv := testServer(cfg)
	defer srv.Close()

	headers := map[string]string{
		"x-api-key":         "test-key",
		"anthropic-version": "2023-06-01",
	}

	resp, _ := postJSON(srv.URL+"/v1/messages", anthropicBody(true), headers)
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	// Parse SSE events
	type sseEvent struct {
		Event string
		Data  string
	}
	var events []sseEvent
	scanner := bufio.NewScanner(resp.Body)
	var currentEvent string
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "event: ") {
			currentEvent = strings.TrimPrefix(line, "event: ")
		} else if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			events = append(events, sseEvent{Event: currentEvent, Data: data})
			currentEvent = ""
		}
	}

	// Verify full event sequence
	expectedEvents := []string{
		"message_start",
		"ping",
		"content_block_start",
		"content_block_delta", // 3 words = 3 deltas
		"content_block_delta",
		"content_block_delta",
		"content_block_stop",
		"message_delta",
		"message_stop",
	}

	if len(events) != len(expectedEvents) {
		t.Fatalf("expected %d events, got %d", len(expectedEvents), len(events))
	}

	for i, ev := range events {
		if ev.Event != expectedEvents[i] {
			t.Errorf("event %d: expected %q, got %q", i, expectedEvents[i], ev.Event)
		}
	}

	// Verify deterministic ID in message_start
	var msgStart map[string]any
	json.Unmarshal([]byte(events[0].Data), &msgStart)
	msg := msgStart["message"].(map[string]any)
	if msg["id"] != "msg_mock_deterministic" {
		t.Errorf("expected deterministic ID in stream, got %v", msg["id"])
	}

	// Verify content deltas produce deterministic text
	var assembled string
	for i := 3; i <= 5; i++ {
		var delta map[string]any
		json.Unmarshal([]byte(events[i].Data), &delta)
		d := delta["delta"].(map[string]any)
		assembled += d["text"].(string)
	}

	expected := "Quantum neural reasoning."
	if assembled != expected {
		t.Errorf("expected assembled %q, got %q", expected, assembled)
	}
}

func TestRequestRecordingPreservesUnknownFields(t *testing.T) {
	srv, _ := testServerWithState(defaultConfig())
	defer srv.Close()

	// Send a request with extra unknown fields
	body := `{
		"model": "claude-3-haiku-20240307",
		"max_tokens": 100,
		"messages": [{"role": "user", "content": "Hello"}],
		"x_test_sentinel": "should_be_recorded",
		"nested_unknown": {"key": "value"}
	}`
	headers := map[string]string{
		"x-api-key":         "test-key",
		"anthropic-version": "2023-06-01",
	}
	chatResp, err := postJSON(srv.URL+"/v1/messages", body, headers)
	if err != nil {
		t.Fatalf("POST /v1/messages failed: %v", err)
	}
	chatResp.Body.Close()

	// Get recorded request
	resp, err := http.Get(srv.URL + "/admin/requests")
	if err != nil {
		t.Fatalf("GET /admin/requests failed: %v", err)
	}
	defer resp.Body.Close()

	var result map[string]any
	json.NewDecoder(resp.Body).Decode(&result)

	requests := result["requests"].([]any)
	if len(requests) != 1 {
		t.Fatalf("expected 1 request, got %d", len(requests))
	}

	recBody := requests[0].(map[string]any)["body"].(map[string]any)

	// Verify unknown fields are preserved in recording
	if recBody["x_test_sentinel"] != "should_be_recorded" {
		t.Errorf("expected x_test_sentinel=should_be_recorded, got %v", recBody["x_test_sentinel"])
	}

	nested := recBody["nested_unknown"].(map[string]any)
	if nested["key"] != "value" {
		t.Errorf("expected nested_unknown.key=value, got %v", nested["key"])
	}
}
