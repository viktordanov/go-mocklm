package main

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

// testServer creates an httptest.Server wired up like main.go.
func testServer(cfg *Config) *httptest.Server {
	state := NewServerState(cfg)

	return httptest.NewServer(buildMux(state))
}

// testServerWithState returns both the test server and its state for runtime config switching.
func testServerWithState(cfg *Config) (*httptest.Server, *ServerState) {
	state := NewServerState(cfg)

	return httptest.NewServer(buildMux(state)), state
}

// defaultConfig returns a Config with fast defaults for testing.
func defaultConfig() *Config {
	return &Config{
		OpenAI: ProviderConfig{
			Tokens:      5,
			ErrorStatus: 500,
		},
		Anthropic: ProviderConfig{
			Tokens:      5,
			ErrorStatus: 500,
		},
	}
}

// --- Helper to build an OpenAI chat request body ---
func openaiChatBody(stream bool) string {
	return `{"model":"gpt-4","stream":` + boolStr(stream) + `,"messages":[{"role":"user","content":"Hello world"}]}`
}

func boolStr(b bool) string {
	if b {
		return "true"
	}
	return "false"
}

// --- Helper to build an Anthropic messages request body ---
func anthropicBody(stream bool) string {
	return `{"model":"claude-3-haiku-20240307","stream":` + boolStr(stream) + `,"max_tokens":100,"messages":[{"role":"user","content":"Hello world"}]}`
}

// --- Helper: make a POST request with JSON body ---
func postJSON(url, body string, headers map[string]string) (*http.Response, error) {
	req, err := http.NewRequest("POST", url, strings.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	return http.DefaultClient.Do(req)
}

// ============================================================
// Tests
// ============================================================

func TestHealthEndpoint(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/health")
	if err != nil {
		t.Fatalf("GET /health failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("expected status 200, got %d", resp.StatusCode)
	}

	var body map[string]string
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if body["status"] != "ok" {
		t.Fatalf("expected status=ok, got %q", body["status"])
	}
}

func TestOpenAIModels(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/v1/models")
	if err != nil {
		t.Fatalf("GET /v1/models failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("expected status 200, got %d", resp.StatusCode)
	}

	var body map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if body["object"] != "list" {
		t.Fatalf("expected object=list, got %v", body["object"])
	}

	data, ok := body["data"].([]any)
	if !ok {
		t.Fatalf("data is not an array")
	}
	if len(data) != 3 {
		t.Fatalf("expected 3 models, got %d", len(data))
	}

	// Verify model IDs
	expectedIDs := map[string]bool{"gpt-4": true, "gpt-4o": true, "gpt-3.5-turbo": true}
	for _, m := range data {
		model := m.(map[string]any)
		id := model["id"].(string)
		if !expectedIDs[id] {
			t.Errorf("unexpected model id: %s", id)
		}
	}
}

func TestOpenAIChatNonStreaming(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
	if err != nil {
		t.Fatalf("POST /v1/chat/completions failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("expected status 200, got %d", resp.StatusCode)
	}

	var body map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	// Verify required fields
	if _, ok := body["id"]; !ok {
		t.Error("missing 'id' field")
	}
	if body["object"] != "chat.completion" {
		t.Errorf("expected object=chat.completion, got %v", body["object"])
	}
	if body["model"] != "gpt-4" {
		t.Errorf("expected model=gpt-4, got %v", body["model"])
	}

	// Verify choices
	choices, ok := body["choices"].([]any)
	if !ok || len(choices) == 0 {
		t.Fatal("missing or empty choices array")
	}
	choice := choices[0].(map[string]any)
	msg := choice["message"].(map[string]any)
	if msg["role"] != "assistant" {
		t.Errorf("expected role=assistant, got %v", msg["role"])
	}
	content, ok := msg["content"].(string)
	if !ok || content == "" {
		t.Error("missing or empty content in message")
	}
	if choice["finish_reason"] != "stop" {
		t.Errorf("expected finish_reason=stop, got %v", choice["finish_reason"])
	}

	// Verify usage
	usage, ok := body["usage"].(map[string]any)
	if !ok {
		t.Fatal("missing usage field")
	}
	if usage["prompt_tokens"] == nil || usage["completion_tokens"] == nil || usage["total_tokens"] == nil {
		t.Error("usage is missing token fields")
	}
	promptTokens := usage["prompt_tokens"].(float64)
	completionTokens := usage["completion_tokens"].(float64)
	totalTokens := usage["total_tokens"].(float64)
	if totalTokens != promptTokens+completionTokens {
		t.Errorf("total_tokens (%v) != prompt_tokens (%v) + completion_tokens (%v)", totalTokens, promptTokens, completionTokens)
	}
}

func TestOpenAIChatStreaming(t *testing.T) {
	cfg := defaultConfig()
	cfg.OpenAI.Tokens = 3
	srv := testServer(cfg)
	defer srv.Close()

	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(true), nil)
	if err != nil {
		t.Fatalf("POST /v1/chat/completions (stream) failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("expected status 200, got %d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Errorf("expected Content-Type text/event-stream, got %s", ct)
	}

	scanner := bufio.NewScanner(resp.Body)
	var dataLines []string
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			dataLines = append(dataLines, strings.TrimPrefix(line, "data: "))
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("error reading stream: %v", err)
	}

	// Expect: role chunk + 3 content chunks + finish chunk + [DONE] = 6 data lines
	if len(dataLines) < 6 {
		t.Fatalf("expected at least 6 data lines, got %d: %v", len(dataLines), dataLines)
	}

	// First data line: role chunk
	var roleChunk map[string]any
	if err := json.Unmarshal([]byte(dataLines[0]), &roleChunk); err != nil {
		t.Fatalf("failed to parse role chunk: %v", err)
	}
	choices := roleChunk["choices"].([]any)
	delta := choices[0].(map[string]any)["delta"].(map[string]any)
	if delta["role"] != "assistant" {
		t.Errorf("expected role=assistant in first chunk, got %v", delta["role"])
	}

	// Content chunks (indices 1..3)
	var assembled string
	for i := 1; i <= 3; i++ {
		var chunk map[string]any
		if err := json.Unmarshal([]byte(dataLines[i]), &chunk); err != nil {
			t.Fatalf("failed to parse content chunk %d: %v", i, err)
		}
		cs := chunk["choices"].([]any)
		d := cs[0].(map[string]any)["delta"].(map[string]any)
		token, ok := d["content"].(string)
		if !ok {
			t.Errorf("chunk %d missing content in delta", i)
		}
		assembled += token
	}
	if assembled == "" {
		t.Error("assembled content is empty")
	}

	// Finish chunk
	var finishChunk map[string]any
	if err := json.Unmarshal([]byte(dataLines[4]), &finishChunk); err != nil {
		t.Fatalf("failed to parse finish chunk: %v", err)
	}
	fc := finishChunk["choices"].([]any)[0].(map[string]any)
	if fc["finish_reason"] != "stop" {
		t.Errorf("expected finish_reason=stop, got %v", fc["finish_reason"])
	}
	// Verify usage in finish chunk
	if finishChunk["usage"] == nil {
		t.Error("missing usage in finish chunk")
	}

	// [DONE] sentinel
	lastLine := dataLines[len(dataLines)-1]
	if lastLine != "[DONE]" {
		t.Errorf("expected last data line to be [DONE], got %q", lastLine)
	}
}

func TestAnthropicNonStreaming(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	headers := map[string]string{
		"x-api-key":         "test-key",
		"anthropic-version": "2023-06-01",
	}
	resp, err := postJSON(srv.URL+"/v1/messages", anthropicBody(false), headers)
	if err != nil {
		t.Fatalf("POST /v1/messages failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("expected status 200, got %d", resp.StatusCode)
	}

	var body map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	// Verify required fields
	if _, ok := body["id"]; !ok {
		t.Error("missing 'id' field")
	}
	if body["type"] != "message" {
		t.Errorf("expected type=message, got %v", body["type"])
	}
	if body["role"] != "assistant" {
		t.Errorf("expected role=assistant, got %v", body["role"])
	}
	if body["model"] != "claude-3-haiku-20240307" {
		t.Errorf("expected model=claude-3-haiku-20240307, got %v", body["model"])
	}
	if body["stop_reason"] != "end_turn" {
		t.Errorf("expected stop_reason=end_turn, got %v", body["stop_reason"])
	}

	// Verify content array
	contentArr, ok := body["content"].([]any)
	if !ok || len(contentArr) == 0 {
		t.Fatal("missing or empty content array")
	}
	block := contentArr[0].(map[string]any)
	if block["type"] != "text" {
		t.Errorf("expected content block type=text, got %v", block["type"])
	}
	text, ok := block["text"].(string)
	if !ok || text == "" {
		t.Error("content block has missing or empty text")
	}

	// Verify usage
	usage, ok := body["usage"].(map[string]any)
	if !ok {
		t.Fatal("missing usage field")
	}
	if usage["input_tokens"] == nil || usage["output_tokens"] == nil {
		t.Error("usage is missing token fields")
	}
}

func TestAnthropicStreaming(t *testing.T) {
	cfg := defaultConfig()
	cfg.Anthropic.Tokens = 3
	srv := testServer(cfg)
	defer srv.Close()

	headers := map[string]string{
		"x-api-key":         "test-key",
		"anthropic-version": "2023-06-01",
	}
	resp, err := postJSON(srv.URL+"/v1/messages", anthropicBody(true), headers)
	if err != nil {
		t.Fatalf("POST /v1/messages (stream) failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("expected status 200, got %d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Errorf("expected Content-Type text/event-stream, got %s", ct)
	}

	// Parse SSE event+data pairs
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
	if err := scanner.Err(); err != nil {
		t.Fatalf("error reading stream: %v", err)
	}

	// Expected event sequence:
	// message_start, content_block_start, 3x content_block_delta, content_block_stop, message_delta, message_stop
	expectedEvents := []string{
		"message_start",
		"content_block_start",
		"content_block_delta",
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

	// Verify message_start data
	var msgStart map[string]any
	if err := json.Unmarshal([]byte(events[0].Data), &msgStart); err != nil {
		t.Fatalf("failed to parse message_start: %v", err)
	}
	if msgStart["type"] != "message_start" {
		t.Errorf("message_start type mismatch: %v", msgStart["type"])
	}
	msg := msgStart["message"].(map[string]any)
	if msg["role"] != "assistant" {
		t.Errorf("expected role=assistant, got %v", msg["role"])
	}

	// Verify content deltas have text
	var assembled string
	for i := 2; i <= 4; i++ {
		var delta map[string]any
		if err := json.Unmarshal([]byte(events[i].Data), &delta); err != nil {
			t.Fatalf("failed to parse content_block_delta %d: %v", i, err)
		}
		d := delta["delta"].(map[string]any)
		text := d["text"].(string)
		assembled += text
	}
	if assembled == "" {
		t.Error("assembled text from deltas is empty")
	}

	// Verify message_delta has stop_reason
	var msgDelta map[string]any
	if err := json.Unmarshal([]byte(events[6].Data), &msgDelta); err != nil {
		t.Fatalf("failed to parse message_delta: %v", err)
	}
	d := msgDelta["delta"].(map[string]any)
	if d["stop_reason"] != "end_turn" {
		t.Errorf("expected stop_reason=end_turn, got %v", d["stop_reason"])
	}
}

func TestAnthropicMissingAPIKey(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	// No x-api-key header
	headers := map[string]string{
		"anthropic-version": "2023-06-01",
	}
	resp, err := postJSON(srv.URL+"/v1/messages", anthropicBody(false), headers)
	if err != nil {
		t.Fatalf("POST /v1/messages failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 401 {
		t.Fatalf("expected status 401, got %d", resp.StatusCode)
	}

	var body map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("failed to decode error response: %v", err)
	}
	if body["type"] != "error" {
		t.Errorf("expected type=error, got %v", body["type"])
	}
	errObj := body["error"].(map[string]any)
	if errObj["type"] != "authentication_error" {
		t.Errorf("expected error type=authentication_error, got %v", errObj["type"])
	}
}

func TestAnthropicMissingVersion(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	// Has x-api-key but no anthropic-version
	headers := map[string]string{
		"x-api-key": "test-key",
	}
	resp, err := postJSON(srv.URL+"/v1/messages", anthropicBody(false), headers)
	if err != nil {
		t.Fatalf("POST /v1/messages failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 401 {
		t.Fatalf("expected status 401, got %d", resp.StatusCode)
	}

	var body map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("failed to decode error response: %v", err)
	}
	errObj := body["error"].(map[string]any)
	if errObj["type"] != "authentication_error" {
		t.Errorf("expected error type=authentication_error, got %v", errObj["type"])
	}
}

func TestErrorRate(t *testing.T) {
	cfg := defaultConfig()
	cfg.OpenAI.ErrorRate = 1.0 // 100% error rate
	cfg.OpenAI.ErrorStatus = 503
	srv := testServer(cfg)
	defer srv.Close()

	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
	if err != nil {
		t.Fatalf("POST /v1/chat/completions failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 503 {
		t.Fatalf("expected status 503, got %d", resp.StatusCode)
	}

	var body map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("failed to decode error response: %v", err)
	}
	errObj := body["error"].(map[string]any)
	if errObj["type"] != "server_error" {
		t.Errorf("expected error type=server_error, got %v", errObj["type"])
	}
}

func TestRateLimiting(t *testing.T) {
	cfg := defaultConfig()
	cfg.OpenAI.RateLimitRPM = 1
	srv := testServer(cfg)
	defer srv.Close()

	// First request should succeed
	resp1, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
	if err != nil {
		t.Fatalf("first request failed: %v", err)
	}
	resp1.Body.Close()
	if resp1.StatusCode != 200 {
		t.Fatalf("first request: expected status 200, got %d", resp1.StatusCode)
	}

	// Second request should be rate limited
	resp2, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
	if err != nil {
		t.Fatalf("second request failed: %v", err)
	}
	defer resp2.Body.Close()
	if resp2.StatusCode != 429 {
		t.Fatalf("second request: expected status 429, got %d", resp2.StatusCode)
	}

	// Verify Retry-After header is present
	retryAfter := resp2.Header.Get("Retry-After")
	if retryAfter == "" {
		t.Error("missing Retry-After header on 429 response")
	}

	var body map[string]any
	if err := json.NewDecoder(resp2.Body).Decode(&body); err != nil {
		t.Fatalf("failed to decode rate limit response: %v", err)
	}
	errObj := body["error"].(map[string]any)
	if errObj["type"] != "rate_limit_error" {
		t.Errorf("expected error type=rate_limit_error, got %v", errObj["type"])
	}
}

func TestDisconnectAfterChunks(t *testing.T) {
	cfg := defaultConfig()
	cfg.OpenAI.Tokens = 10
	cfg.OpenAI.DisconnectAfterChunks = 2
	srv := testServer(cfg)
	defer srv.Close()

	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(true), nil)
	if err != nil {
		t.Fatalf("POST /v1/chat/completions (stream) failed: %v", err)
	}
	defer resp.Body.Close()

	// Read all data from the stream
	data, err := io.ReadAll(resp.Body)
	// The connection may be abruptly closed, so we might get an error
	// That's expected behavior - the server hijacks and closes the connection
	body := string(data)

	// We should have the role chunk + 2 content chunks before disconnect
	// The role chunk is always sent first, then content chunks with fault check
	lines := strings.Split(body, "\n")
	var dataLines []string
	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") {
			dataLines = append(dataLines, strings.TrimPrefix(line, "data: "))
		}
	}

	// Role chunk + 2 content chunks (indices 0 and 1) = 3 data lines
	// The disconnect happens at chunk index 2, so chunks 0 and 1 get through
	if len(dataLines) < 3 {
		t.Logf("got %d data lines (body read error: %v)", len(dataLines), err)
		// At minimum we should have the role chunk
		if len(dataLines) < 1 {
			t.Fatal("expected at least 1 data line (role chunk)")
		}
	}

	// Should NOT have [DONE] since connection was disconnected
	if strings.Contains(body, "[DONE]") {
		t.Error("expected no [DONE] sentinel after disconnect")
	}
}

func TestMalformedChunk(t *testing.T) {
	cfg := defaultConfig()
	cfg.OpenAI.Tokens = 4
	cfg.OpenAI.MalformedChunk = true
	srv := testServer(cfg)
	defer srv.Close()

	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(true), nil)
	if err != nil {
		t.Fatalf("POST /v1/chat/completions (stream) failed: %v", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("error reading response: %v", err)
	}
	body := string(data)

	// Verify that the corrupt JSON appears in the stream
	if !strings.Contains(body, "{INVALID JSON CORRUPT") {
		t.Error("expected malformed chunk with '{INVALID JSON CORRUPT' in stream")
	}

	// The stream should still complete (malformed chunk doesn't terminate the stream)
	if !strings.Contains(body, "[DONE]") {
		t.Error("expected [DONE] sentinel even with malformed chunk")
	}
}

func TestContentGeneration(t *testing.T) {
	t.Run("generateWords returns correct count", func(t *testing.T) {
		words := generateWords(10)
		if len(words) != 10 {
			t.Fatalf("expected 10 words, got %d", len(words))
		}
		// All words should be non-empty
		for i, w := range words {
			if w == "" {
				t.Errorf("word %d is empty", i)
			}
		}
	})

	t.Run("generateWords wraps around word list", func(t *testing.T) {
		// Request more words than the list contains
		words := generateWords(len(wordList) + 5)
		if len(words) != len(wordList)+5 {
			t.Fatalf("expected %d words, got %d", len(wordList)+5, len(words))
		}
	})

	t.Run("joinContent empty", func(t *testing.T) {
		result := joinContent(nil)
		if result != "" {
			t.Errorf("expected empty string, got %q", result)
		}
	})

	t.Run("joinContent single word", func(t *testing.T) {
		result := joinContent([]string{"hello"})
		if result != "Hello." {
			t.Errorf("expected 'Hello.', got %q", result)
		}
	})

	t.Run("joinContent multiple words", func(t *testing.T) {
		result := joinContent([]string{"hello", "world"})
		if result != "Hello world." {
			t.Errorf("expected 'Hello world.', got %q", result)
		}
	})
}

func TestConfigDefaults(t *testing.T) {
	// Write a minimal TOML config to a temp file
	tmpDir := t.TempDir()
	cfgPath := tmpDir + "/test.toml"
	// Write minimal config that just defines empty sections
	cfgContent := `[server]
[openai]
[anthropic]
`
	if err := writeFile(cfgPath, cfgContent); err != nil {
		t.Fatalf("failed to write temp config: %v", err)
	}

	// Set CONFIG_PATH to our test file
	t.Setenv("CONFIG_PATH", cfgPath)

	cfg, err := loadConfig()
	if err != nil {
		t.Fatalf("loadConfig failed: %v", err)
	}

	// Verify defaults
	if cfg.Server.Port != 9999 {
		t.Errorf("expected default port 9999, got %d", cfg.Server.Port)
	}
	if cfg.Server.Host != "0.0.0.0" {
		t.Errorf("expected default host 0.0.0.0, got %s", cfg.Server.Host)
	}
	if cfg.OpenAI.Tokens != 20 {
		t.Errorf("expected default openai tokens 20, got %d", cfg.OpenAI.Tokens)
	}
	if cfg.Anthropic.Tokens != 20 {
		t.Errorf("expected default anthropic tokens 20, got %d", cfg.Anthropic.Tokens)
	}
	if cfg.OpenAI.ErrorStatus != 500 {
		t.Errorf("expected default openai error_status 500, got %d", cfg.OpenAI.ErrorStatus)
	}
	if cfg.Anthropic.ErrorStatus != 500 {
		t.Errorf("expected default anthropic error_status 500, got %d", cfg.Anthropic.ErrorStatus)
	}
}

// writeFile is a minimal helper to write a string to a file.
func writeFile(path, content string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.WriteString(content)
	return err
}
