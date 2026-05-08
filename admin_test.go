package main

import (
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

	if len(presets) != 11 {
		t.Errorf("expected 11 presets, got %d", len(presets))
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
	applyProviderDefaults(&openai)
	applyProviderDefaults(&anthropic)
	state.Update(openai, anthropic, "openai-disconnect")

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
