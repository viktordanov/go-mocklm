package main

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"
)

// --- Phase 1: request fidelity, tool echo, stop_reason/cache config ---

func postAnthropic(t *testing.T, url, body string) *http.Response {
	t.Helper()
	req, _ := http.NewRequest("POST", url+"/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", "test-key")
	req.Header.Set("anthropic-version", "2023-06-01")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	return resp
}

func decodeBody(t *testing.T, resp *http.Response) map[string]any {
	t.Helper()
	defer resp.Body.Close()
	var out map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode failed: %v", err)
	}
	return out
}

func TestAnthropicArrayContentAccepted(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	// Exactly what an OpenAI->Anthropic proxy emits for a tool round-trip:
	// assistant tool_use history + user tool_result blocks.
	body := `{
		"model": "claude-3-haiku-20240307",
		"max_tokens": 50,
		"messages": [
			{"role": "user", "content": "What is the weather?"},
			{"role": "assistant", "content": [
				{"type": "text", "text": "Let me check."},
				{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}}
			]},
			{"role": "user", "content": [
				{"type": "tool_result", "tool_use_id": "toolu_1", "content": "18C and sunny"}
			]}
		]
	}`

	resp := postAnthropic(t, srv.URL, body)
	if resp.StatusCode != 200 {
		t.Fatalf("array content must be accepted, got %d", resp.StatusCode)
	}
	out := decodeBody(t, resp)
	if out["type"] != "message" {
		t.Fatalf("expected message response, got %v", out)
	}
}

func TestOpenAIArrayContentAccepted(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	body := `{
		"model": "gpt-4",
		"messages": [
			{"role": "user", "content": [
				{"type": "text", "text": "Describe this"},
				{"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}
			]}
		]
	}`

	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("multi-part content must be accepted, got %d", resp.StatusCode)
	}
	resp.Body.Close()
}

func TestAnthropicToolEcho(t *testing.T) {
	cfg := defaultConfig()
	cfg.Anthropic.ToolUseResponse = true
	srv := testServer(cfg)
	defer srv.Close()

	body := `{
		"model": "claude-3-haiku-20240307",
		"max_tokens": 50,
		"messages": [{"role": "user", "content": "search something"}],
		"tools": [
			{"name": "web_search", "input_schema": {"type": "object"}},
			{"name": "other_tool", "input_schema": {"type": "object"}}
		]
	}`

	out := decodeBody(t, postAnthropic(t, srv.URL, body))

	if out["stop_reason"] != "tool_use" {
		t.Fatalf("expected stop_reason tool_use, got %v", out["stop_reason"])
	}
	content := out["content"].([]any)
	toolBlock := content[len(content)-1].(map[string]any)
	if toolBlock["type"] != "tool_use" || toolBlock["name"] != "web_search" {
		t.Fatalf("expected tool_use echoing first offered tool, got %v", toolBlock)
	}
}

func TestAnthropicToolEchoOpenAIShapedTools(t *testing.T) {
	// Tools in OpenAI wrapper shape (as sent by a proxy that didn't reshape)
	// still resolve a name.
	cfg := defaultConfig()
	cfg.Anthropic.ToolUseResponse = true
	srv := testServer(cfg)
	defer srv.Close()

	body := `{
		"model": "claude-3-haiku-20240307",
		"max_tokens": 50,
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [{"type": "function", "function": {"name": "lookup_db"}}]
	}`

	out := decodeBody(t, postAnthropic(t, srv.URL, body))
	content := out["content"].([]any)
	toolBlock := content[len(content)-1].(map[string]any)
	if toolBlock["name"] != "lookup_db" {
		t.Fatalf("expected lookup_db, got %v", toolBlock["name"])
	}
}

func TestAnthropicStreamedToolUse(t *testing.T) {
	cfg := defaultConfig()
	cfg.Anthropic.ToolUseResponse = true
	srv := testServer(cfg)
	defer srv.Close()

	body := `{
		"model": "claude-3-haiku-20240307",
		"max_tokens": 50,
		"stream": true,
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [{"name": "web_search", "input_schema": {"type": "object"}}]
	}`

	resp := postAnthropic(t, srv.URL, body)
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

	if !strings.Contains(text, `"type":"tool_use"`) {
		t.Fatalf("stream must contain a tool_use content_block_start:\n%s", text)
	}
	if !strings.Contains(text, "input_json_delta") {
		t.Fatalf("stream must contain input_json_delta fragments:\n%s", text)
	}
	if !strings.Contains(text, `"stop_reason":"tool_use"`) {
		t.Fatalf("stream must finish with stop_reason tool_use:\n%s", text)
	}

	// Reassembled partial_json fragments must form valid JSON.
	var fragments strings.Builder
	for _, line := range strings.Split(text, "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var event struct {
			Type  string `json:"type"`
			Delta struct {
				Type        string `json:"type"`
				PartialJSON string `json:"partial_json"`
			} `json:"delta"`
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event); err != nil {
			continue
		}
		if event.Delta.Type == "input_json_delta" {
			fragments.WriteString(event.Delta.PartialJSON)
		}
	}
	var input map[string]any
	if err := json.Unmarshal([]byte(fragments.String()), &input); err != nil {
		t.Fatalf("reassembled partial_json is not valid JSON: %q (%v)", fragments.String(), err)
	}
}

func TestAnthropicStopReasonOverride(t *testing.T) {
	cfg := defaultConfig()
	cfg.Anthropic.StopReason = "pause_turn"
	srv := testServer(cfg)
	defer srv.Close()

	out := decodeBody(t, postAnthropic(t, srv.URL, anthropicBody(false)))
	if out["stop_reason"] != "pause_turn" {
		t.Fatalf("expected stop_reason override pause_turn, got %v", out["stop_reason"])
	}
}

func TestAnthropicCacheUsageFields(t *testing.T) {
	cfg := defaultConfig()
	cfg.Anthropic.CacheReadTokens = 128
	cfg.Anthropic.CacheCreationTokens = 64
	srv := testServer(cfg)
	defer srv.Close()

	out := decodeBody(t, postAnthropic(t, srv.URL, anthropicBody(false)))
	usage := out["usage"].(map[string]any)
	if usage["cache_read_input_tokens"] != float64(128) {
		t.Fatalf("expected cache_read_input_tokens=128, got %v", usage)
	}
	if usage["cache_creation_input_tokens"] != float64(64) {
		t.Fatalf("expected cache_creation_input_tokens=64, got %v", usage)
	}
}

func TestAnthropicThinkingRequestTriggersThinkingBlock(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	body := `{
		"model": "claude-3-haiku-20240307",
		"max_tokens": 50,
		"thinking": {"type": "enabled", "budget_tokens": 1024},
		"messages": [{"role": "user", "content": "think hard"}]
	}`

	out := decodeBody(t, postAnthropic(t, srv.URL, body))
	content := out["content"].([]any)
	first := content[0].(map[string]any)
	if first["type"] != "thinking" {
		t.Fatalf("thinking-enabled request should produce a thinking block, got %v", content)
	}
}

func TestOpenAIToolEcho(t *testing.T) {
	cfg := defaultConfig()
	cfg.OpenAI.ToolUseResponse = true
	srv := testServer(cfg)
	defer srv.Close()

	body := `{
		"model": "gpt-4",
		"messages": [{"role": "user", "content": "search"}],
		"tools": [{"type": "function", "function": {"name": "web_search"}}]
	}`

	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	out := decodeBody(t, resp)

	choice := out["choices"].([]any)[0].(map[string]any)
	if choice["finish_reason"] != "tool_calls" {
		t.Fatalf("expected finish_reason tool_calls, got %v", choice["finish_reason"])
	}
	message := choice["message"].(map[string]any)
	calls := message["tool_calls"].([]any)
	fn := calls[0].(map[string]any)["function"].(map[string]any)
	if fn["name"] != "web_search" {
		t.Fatalf("expected echoed tool name web_search, got %v", fn["name"])
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(fn["arguments"].(string)), &args); err != nil {
		t.Fatalf("arguments must be a JSON string: %v", err)
	}
}

func TestRecordingCap(t *testing.T) {
	state := NewServerState(defaultConfig())
	for i := 0; i < maxRecordedRequests+50; i++ {
		state.RecordRequest(RecordedRequest{Timestamp: time.Now(), Provider: "openai"})
	}
	if got := len(state.Requests()); got != maxRecordedRequests {
		t.Fatalf("recording buffer must cap at %d, got %d", maxRecordedRequests, got)
	}
}
