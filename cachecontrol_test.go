package main

import (
	"encoding/json"
	"strings"
	"testing"
)

// --- reject_cache_control: the inbound-leak oracle ---

func TestFindCacheControlLeakWalker(t *testing.T) {
	parse := func(s string) any {
		var v any
		if err := json.Unmarshal([]byte(s), &v); err != nil {
			t.Fatalf("bad fixture: %v", err)
		}
		return v
	}
	for _, tc := range []struct {
		name     string
		body     string
		wantPath string
		wantHit  bool
	}{
		{"nested content block",
			`{"messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"text","text":"there","cache_control":{"type":"ephemeral"}}]}]}`,
			"messages[0].content[1].cache_control", true},
		{"system array",
			`{"system":[{"type":"text","text":"sys","cache_control":{"type":"ephemeral"}}]}`,
			"system[0].cache_control", true},
		{"tools",
			`{"tools":[{"name":"t","input_schema":{},"cache_control":{"type":"ephemeral","ttl":"5m"}}]}`,
			"tools[0].cache_control", true},
		{"top level",
			`{"cache_control":{"type":"ephemeral"},"model":"m"}`,
			"cache_control", true},
		{"arrays of arrays",
			`{"input":[[{"cache_control":{"type":"ephemeral"}}]]}`,
			"input[0][0].cache_control", true},
		{"no type member: accepted",
			`{"messages":[{"cache_control":{"ttl":"5m"}}]}`, "", false},
		{"cache_control null: accepted",
			`{"messages":[{"cache_control":null}]}`, "", false},
		{"cache_control string: accepted",
			`{"messages":[{"cache_control":"ephemeral"}]}`, "", false},
		{"benign look-alike keys: accepted",
			`{"my_cache_control":{"type":"x"},"cache_control_hint":{"type":"y"}}`, "", false},
		{"multiple occurrences: deterministic first path (lexicographic keys)",
			`{"zz":[{"cache_control":{"type":"ephemeral"}}],"aa":{"cache_control":{"type":"ephemeral"}}}`,
			"aa.cache_control", true},
		{"object's own match beats children",
			`{"cache_control":{"type":"e"},"child":{"cache_control":{"type":"e"}}}`,
			"cache_control", true},
		{"awkward key escaping on the path",
			`{"we ird":[{"cache_control":{"type":"e"}}]}`,
			`["we ird"][0].cache_control`, true},
		{"newline in key is escaped, not literal",
			`{"a\nb":[{"cache_control":{"type":"e"}}]}`,
			"[\"a\\nb\"][0].cache_control", true},
		{"tab in key is escaped",
			`{"a\tb":{"cache_control":{"type":"e"}}}`,
			"[\"a\\tb\"].cache_control", true},
		{"carriage return in key is escaped",
			`{"a\rb":{"cache_control":{"type":"e"}}}`,
			"[\"a\\rb\"].cache_control", true},
		{"embedded quote in key is escaped",
			`{"he\"y":{"cache_control":{"type":"e"}}}`,
			"[\"he\\\"y\"].cache_control", true},
		{"backslash in key is escaped",
			`{"a\\b":{"cache_control":{"type":"e"}}}`,
			"[\"a\\\\b\"].cache_control", true},
	} {
		path, hit := findCacheControlLeak(parse(tc.body), "")
		if hit != tc.wantHit || path != tc.wantPath {
			t.Errorf("%s: got (%q, %v), want (%q, %v)", tc.name, path, hit, tc.wantPath, tc.wantHit)
		}
	}
}

func TestRejectCacheControlAcrossSurfaces(t *testing.T) {
	// The oracle covers every body-bearing surface via the provider knob.
	leak := `,"messages":[{"role":"user","content":[{"type":"text","text":"x","cache_control":{"type":"ephemeral"}}]}]`
	surfaces := []struct {
		path, body, provider string
	}{
		{"/v1/chat/completions", `{"model":"gpt-4"` + leak + `}`, "openai"},
		{"/v1/responses", `{"model":"gpt-4","input":[]` + leak + `}`, "openai"},
		{"/v1/completions", `{"model":"gpt-3.5-turbo-instruct","prompt":"x"` + leak + `}`, "openai"},
		{"/v1/embeddings", `{"model":"text-embedding-3-small","input":"x"` + leak + `}`, "openai"},
		{"/v1/messages", `{"model":"claude-3-haiku-20240307","max_tokens":10` + leak + `}`, "anthropic"},
		{"/model/claude-x/converse", `{` + strings.TrimPrefix(leak, ",") + `}`, "bedrock"},
	}

	cfg := defaultConfig()
	cfg.OpenAI.RejectCacheControl = true
	cfg.Anthropic.RejectCacheControl = true
	cfg.Bedrock.RejectCacheControl = true
	srv := testServer(cfg)
	defer srv.Close()

	anthropicHeaders := map[string]string{
		"x-api-key":         "test-key",
		"anthropic-version": "2023-06-01",
	}
	for _, s := range surfaces {
		var headers map[string]string
		if s.provider == "anthropic" {
			headers = anthropicHeaders
		}
		resp, err := postJSON(srv.URL+s.path, s.body, headers)
		if err != nil {
			t.Fatalf("%s: request failed: %v", s.path, err)
		}
		raw := make([]byte, 2048)
		n, _ := resp.Body.Read(raw)
		resp.Body.Close()
		if resp.StatusCode != 400 {
			t.Fatalf("%s: leak must draw 400, got %d (%s)", s.path, resp.StatusCode, raw[:n])
		}
		if !strings.Contains(string(raw[:n]), "messages[0].content[0].cache_control") {
			t.Errorf("%s: diagnostic must name the leak path, got %s", s.path, raw[:n])
		}
		if s.provider == "bedrock" {
			if et := resp.Header.Get("X-Amzn-ErrorType"); et != "ValidationException" {
				t.Errorf("bedrock leak must be a modeled ValidationException, got %q", et)
			}
		}
	}
}

func TestRejectCacheControlDefaultOffAndNegatives(t *testing.T) {
	// Default off: the historical accept/echo behavior is untouched.
	srv := testServer(defaultConfig())
	defer srv.Close()
	body := `{"model":"gpt-4","messages":[{"role":"user","content":[{"type":"text","text":"x","cache_control":{"type":"ephemeral"}}]}]}`
	resp, err := postJSON(srv.URL+"/v1/chat/completions", body, nil)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("default-off must accept cache_control, got %d", resp.StatusCode)
	}

	// Enabled but no leak (cache_control without "type", benign keys): 200.
	cfg := defaultConfig()
	cfg.OpenAI.RejectCacheControl = true
	srv2 := testServer(cfg)
	defer srv2.Close()
	clean := `{"model":"gpt-4","cache_control_hint":"x","messages":[{"role":"user","content":"hi","cache_control":{"ttl":"5m"}}]}`
	resp, err = postJSON(srv2.URL+"/v1/chat/completions", clean, nil)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("no-type cache_control must be accepted (consumer predicate), got %d", resp.StatusCode)
	}

	// Malformed JSON keeps the normal invalid-JSON error, not the oracle.
	resp, err = postJSON(srv2.URL+"/v1/chat/completions", `{"model":`, nil)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	raw := make([]byte, 512)
	n, _ := resp.Body.Read(raw)
	resp.Body.Close()
	if resp.StatusCode != 400 || !strings.Contains(string(raw[:n]), "Invalid JSON") {
		t.Fatalf("malformed body must keep the invalid-JSON error, got %d %s", resp.StatusCode, raw[:n])
	}
}

func TestRejectCacheControlBeatsFaults(t *testing.T) {
	// The oracle is an inbound assertion: it must fire BEFORE configured
	// faults could mask it (a 503 here would hide the leak).
	cfg := defaultConfig()
	cfg.OpenAI.RejectCacheControl = true
	cfg.OpenAI.FailFirstN = 5
	cfg.OpenAI.ErrorStatus = 503
	srv := testServer(cfg)
	defer srv.Close()

	body := `{"model":"gpt-4","messages":[{"role":"user","content":[{"type":"text","text":"x","cache_control":{"type":"ephemeral"}}]}]}`
	resp, err := postJSON(srv.URL+"/v1/chat/completions", body, nil)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 400 {
		t.Fatalf("the oracle must beat fail_first_n: want 400, got %d", resp.StatusCode)
	}
	// And a clean request still draws the fault, proving the fault config
	// itself is live.
	resp, err = postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 503 {
		t.Fatalf("clean request should draw the fail_first_n fault, got %d", resp.StatusCode)
	}
}

func TestRejectCacheControlPerScenario(t *testing.T) {
	// Scenarios reuse ProviderConfig verbatim, so the oracle is
	// per-scenario for free: the matched scenario rejects, non-scenario
	// traffic on the same provider keeps accepting.
	srv := testServer(defaultConfig())
	defer srv.Close()

	reg, _ := json.Marshal(map[string]any{
		"id": "cc-oracle", "provider": "openai", "model": "gpt-cc",
		"output": map[string]any{"text": "ok"},
		"config": map[string]any{"reject_cache_control": true},
	})
	mustRegisterScenario(t, srv.URL, string(reg))

	leaky := func(model string) string {
		return `{"model":"` + model + `","messages":[{"role":"user","content":[{"type":"text","text":"x","cache_control":{"type":"ephemeral"}}]}]}`
	}
	resp, err := postJSON(srv.URL+"/v1/chat/completions", leaky("gpt-cc"), nil)
	if err != nil {
		t.Fatalf("scenario request failed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 400 {
		t.Fatalf("scenario-scoped oracle must reject, got %d", resp.StatusCode)
	}
	resp, err = postJSON(srv.URL+"/v1/chat/completions", leaky("gpt-4"), nil)
	if err != nil {
		t.Fatalf("non-scenario request failed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("non-scenario traffic must keep accepting (provider knob off), got %d", resp.StatusCode)
	}
}
