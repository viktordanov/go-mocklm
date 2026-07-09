package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"unicode/utf8"
)

// --- Phase 3a: scenario registry — chunkExact losslessness, exact-content
// emit paths, per-scenario raw capture + counters, provider/surface
// binding, register-time rejection ---

// chunkExact property: strings.Join(chunkExact(s, c), "") == s for EVERY
// mode and EVERY input, including empty and whitespace-only — the gate the
// exact-content primitive stands on.
func TestChunkExactLossless(t *testing.T) {
	inputs := []string{
		"",
		" ",
		"   \n\t  ",
		"hello",
		"hello world",
		"  leading and trailing  ",
		"runs  of   spaces\n\nand\nnewlines\t tabs ",
		"héllo wörld — ünïcode 你好世界 🎉🎊 done",
		"\n\nonly\nnewlines\n\n",
		"a",
		"one-word",
		"x y",
	}
	modes := []string{"", "whole", "runes", "words"}
	sizes := []int{0, 1, 2, 3, 7, 100}

	for _, s := range inputs {
		for _, mode := range modes {
			for _, size := range sizes {
				c := Chunking{Mode: mode, Size: size}
				chunks := chunkExact(s, c)
				if got := strings.Join(chunks, ""); got != s {
					t.Fatalf("mode=%q size=%d input=%q: join(chunks) = %q, want the input back", mode, size, s, got)
				}
				for i, ch := range chunks {
					if ch == "" {
						t.Fatalf("mode=%q size=%d input=%q: chunk %d is empty", mode, size, s, i)
					}
					if !utf8.ValidString(ch) {
						t.Fatalf("mode=%q size=%d input=%q: chunk %d is not valid UTF-8 (%q)", mode, size, s, i, ch)
					}
					if (mode == "runes" || mode == "") && size > 0 {
						if n := len([]rune(ch)); n > size {
							t.Fatalf("runes size=%d input=%q: chunk %d has %d runes", size, s, i, n)
						}
					}
				}
			}
		}
	}
}

func TestChunkExactShapes(t *testing.T) {
	// Words mode: leading whitespace rides the first chunk, each word
	// carries its trailing whitespace, boundaries fall only before a word
	// start.
	got := chunkExact("  hello   world x", Chunking{Mode: "words", Size: 1})
	want := []string{"  hello   ", "world ", "x"}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("words size=1: got %q, want %q", got, want)
	}
	// Whitespace-only input is one chunk equal to itself.
	got = chunkExact(" \n ", Chunking{Mode: "words", Size: 2})
	if len(got) != 1 || got[0] != " \n " {
		t.Fatalf("whitespace-only: got %q", got)
	}
	// Runes mode never splits a multibyte rune.
	got = chunkExact("héllo", Chunking{Mode: "runes", Size: 2})
	want = []string{"hé", "ll", "o"}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("runes size=2: got %q, want %q", got, want)
	}
	// Empty input: zero chunks (join == "" still holds).
	if got := chunkExact("", Chunking{Mode: "whole"}); len(got) != 0 {
		t.Fatalf("empty input: got %q, want no chunks", got)
	}
}

// --- registration + admin surface ---

func registerScenario(t *testing.T, srvURL, body, query string) *http.Response {
	t.Helper()
	resp, err := postJSON(srvURL+"/admin/scenarios"+query, body, nil)
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	return resp
}

func mustRegisterScenario(t *testing.T, srvURL, body string) {
	t.Helper()
	resp := registerScenario(t, srvURL, body, "")
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("register returned %d: %s", resp.StatusCode, raw)
	}
}

func TestScenarioRegisterRejections(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	cases := []struct {
		name string
		body string
		want string // substring of the rejection message
	}{
		{"missing id", `{"provider":"openai"}`, "id is required"},
		{"slash in id", `{"id":"a/b","provider":"openai"}`, "path segment"},
		{"unknown provider", `{"id":"s1","provider":"cohere"}`, "unknown scenario provider"},
		{"R5 rate_limit_rpm", `{"id":"s1","provider":"openai","config":{"rate_limit_rpm":5}}`, "provider-global"},
		{"R5 max_concurrent", `{"id":"s1","provider":"openai","config":{"max_concurrent":2}}`, "provider-global"},
		{"surface mismatch", `{"id":"s1","provider":"openai","surface":"messages"}`, "does not match provider"},
		{"bad chunking mode", `{"id":"s1","provider":"openai","output":{"text":"x","chunking":{"mode":"chars","size":1}}}`, "unknown chunking mode"},
		{"exact output + tool_use_response", `{"id":"s1","provider":"openai","output":{"text":"x"},"config":{"tool_use_response":true}}`, "cannot be combined with tool_use_response"},
		{"bad fault mode", `{"id":"s1","provider":"openai","config":{"faults":[{"mode":"explode"}]}}`, "unknown fault mode"},
		{"unknown fault preset", `{"id":"s1","provider":"openai","fault_preset":"nope"}`, "Unknown fault preset"},
	}
	for _, tc := range cases {
		resp := registerScenario(t, srv.URL, tc.body, "")
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode == 200 {
			t.Errorf("%s: registration succeeded, want rejection", tc.name)
			continue
		}
		if !strings.Contains(string(raw), tc.want) {
			t.Errorf("%s: rejection %q does not mention %q", tc.name, raw, tc.want)
		}
	}
}

func TestScenarioModelCollision(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	mustRegisterScenario(t, srv.URL, `{"id":"a","provider":"openai","model":"gpt-cell-1","output":{"text":"A"}}`)

	// A different id claiming the same (provider, model) is a 409...
	resp := registerScenario(t, srv.URL, `{"id":"b","provider":"openai","model":"gpt-cell-1","output":{"text":"B"}}`, "")
	if resp.StatusCode != 409 {
		t.Fatalf("colliding registration: got %d, want 409", resp.StatusCode)
	}
	resp.Body.Close()

	// ...unless ?replace=1 takes it over.
	resp = registerScenario(t, srv.URL, `{"id":"b","provider":"openai","model":"gpt-cell-1","output":{"text":"B"}}`, "?replace=1")
	if resp.StatusCode != 200 {
		t.Fatalf("replace registration: got %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()

	// Re-POSTing the same id always replaces.
	mustRegisterScenario(t, srv.URL, `{"id":"b","provider":"openai","model":"gpt-cell-1","output":{"text":"B2"}}`)
}

// --- exact content through the emit paths ---

// exactProbe carries every whitespace shape ContentText destroys: newlines,
// runs of spaces, leading/trailing whitespace, tabs, multibyte runes.
const exactProbe = "  Line one\n\nline  two\twith\ttabs — ünïcode 你好 🎉 trailing  "

func openaiChatBodyModel(model string, stream bool) string {
	return `{"model":"` + model + `","stream":` + boolStr(stream) + `,"messages":[{"role":"user","content":"Hello"}]}`
}

func anthropicBodyModel(model string, stream bool) string {
	return `{"model":"` + model + `","stream":` + boolStr(stream) + `,"max_tokens":100,"messages":[{"role":"user","content":"Hello"}]}`
}

func TestScenarioExactContentOpenAINonStream(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	body, _ := json.Marshal(map[string]any{
		"id": "oai-exact", "provider": "openai", "model": "gpt-exact",
		"output": map[string]any{"text": exactProbe, "output_tokens": 42},
	})
	mustRegisterScenario(t, srv.URL, string(body))

	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBodyModel("gpt-exact", false), nil)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decoding response: %v", err)
	}
	if out.Choices[0].Message.Content != exactProbe {
		t.Fatalf("content = %q, want the exact probe %q", out.Choices[0].Message.Content, exactProbe)
	}
	if out.Usage.CompletionTokens != 42 {
		t.Fatalf("completion_tokens = %d, want the pinned 42 (R9: explicit output_tokens honored)", out.Usage.CompletionTokens)
	}
}

// collectOpenAIDeltas reads an OpenAI chat SSE stream and returns the
// concatenated delta.content plus the delta count.
func collectOpenAIDeltas(t *testing.T, srvURL, body string, headers map[string]string) (string, int) {
	t.Helper()
	resp, err := postJSON(srvURL+"/v1/chat/completions", body, headers)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 200, got %d: %s", resp.StatusCode, raw)
	}
	var sb strings.Builder
	count := 0
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content *string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk); err != nil {
			t.Fatalf("chunk is invalid JSON: %v", err)
		}
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != nil && *chunk.Choices[0].Delta.Content != "" {
			sb.WriteString(*chunk.Choices[0].Delta.Content)
			count++
		}
	}
	return sb.String(), count
}

func TestScenarioExactContentOpenAIStreamChunking(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	body, _ := json.Marshal(map[string]any{
		"id": "oai-stream-exact", "provider": "openai", "model": "gpt-stream-exact",
		"output": map[string]any{
			"text":     exactProbe,
			"chunking": map[string]any{"mode": "runes", "size": 3},
		},
	})
	mustRegisterScenario(t, srv.URL, string(body))

	got, count := collectOpenAIDeltas(t, srv.URL, openaiChatBodyModel("gpt-stream-exact", true), nil)
	if got != exactProbe {
		t.Fatalf("reassembled stream = %q, want the exact probe %q", got, exactProbe)
	}
	wantChunks := len(chunkExact(exactProbe, Chunking{Mode: "runes", Size: 3}))
	if count != wantChunks {
		t.Fatalf("stream carried %d content deltas, want %d (runes/3 slicing)", count, wantChunks)
	}
}

// collectAnthropicEventsHeaders is collectAnthropicEvents plus extra request
// headers (scenario targeting).
func collectAnthropicEventsHeaders(t *testing.T, srvURL, body string, headers map[string]string) []streamEvent {
	t.Helper()
	req, _ := http.NewRequest("POST", srvURL+"/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", "test-key")
	req.Header.Set("anthropic-version", "2023-06-01")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 200, got %d: %s", resp.StatusCode, raw)
	}
	var events []streamEvent
	current := ""
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		switch {
		case strings.HasPrefix(line, "event: "):
			current = strings.TrimPrefix(line, "event: ")
		case strings.HasPrefix(line, "data: "):
			var data map[string]any
			if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &data); err != nil {
				t.Fatalf("event %q carries invalid JSON: %v", current, err)
			}
			events = append(events, streamEvent{name: current, data: data})
		}
	}
	return events
}

const exactThinkingProbe = "Deliberate  step one.\nStep two with  spacing."

func TestScenarioExactContentAnthropicWithThinking(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	body, _ := json.Marshal(map[string]any{
		"id": "anth-exact", "provider": "anthropic", "model": "claude-exact",
		"output": map[string]any{
			"text":     exactProbe,
			"thinking": exactThinkingProbe,
			"chunking": map[string]any{"mode": "words", "size": 2},
		},
	})
	mustRegisterScenario(t, srv.URL, string(body))

	// Non-stream: text block and thinking block byte-equal.
	resp := postAnthropic(t, srv.URL, anthropicBodyModel("claude-exact", false))
	defer resp.Body.Close()
	var out struct {
		Content []struct {
			Type     string `json:"type"`
			Text     string `json:"text"`
			Thinking string `json:"thinking"`
		} `json:"content"`
		Usage struct {
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decoding response: %v", err)
	}
	if len(out.Content) != 2 || out.Content[0].Type != "thinking" || out.Content[1].Type != "text" {
		t.Fatalf("content blocks = %+v, want [thinking, text]", out.Content)
	}
	if out.Content[0].Thinking != exactThinkingProbe {
		t.Fatalf("thinking = %q, want %q", out.Content[0].Thinking, exactThinkingProbe)
	}
	if out.Content[1].Text != exactProbe {
		t.Fatalf("text = %q, want %q", out.Content[1].Text, exactProbe)
	}
	// R9 derived rule: max(1, (runes(text)+runes(thinking))/4).
	wantTokens := (len([]rune(exactProbe)) + len([]rune(exactThinkingProbe))) / 4
	if out.Usage.OutputTokens != wantTokens {
		t.Fatalf("output_tokens = %d, want derived %d", out.Usage.OutputTokens, wantTokens)
	}

	// Stream: thinking_delta and text_delta reassemble byte-exact.
	events := collectAnthropicEventsHeaders(t, srv.URL, anthropicBodyModel("claude-exact", true), nil)
	var thinking, text strings.Builder
	for _, ev := range events {
		if ev.name != "content_block_delta" {
			continue
		}
		delta := ev.data["delta"].(map[string]any)
		switch delta["type"] {
		case "thinking_delta":
			thinking.WriteString(delta["thinking"].(string))
		case "text_delta":
			text.WriteString(delta["text"].(string))
		}
	}
	if thinking.String() != exactThinkingProbe {
		t.Fatalf("streamed thinking = %q, want %q", thinking.String(), exactThinkingProbe)
	}
	if text.String() != exactProbe {
		t.Fatalf("streamed text = %q, want %q", text.String(), exactProbe)
	}
}

// --- matching: header targeting, provider binding, unwired surfaces ---

func TestScenarioHeaderTargetingAndProviderBinding(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	body, _ := json.Marshal(map[string]any{
		"id": "anth-only", "provider": "anthropic",
		"output": map[string]any{"text": "anthropic exact"},
	})
	mustRegisterScenario(t, srv.URL, string(body))

	// Header targeting works on the matching provider route (no model key).
	resp := postAnthropic(t, srv.URL, anthropicBodyModel("claude-any", false))
	resp.Body.Close()
	req, _ := http.NewRequest("POST", srv.URL+"/v1/messages", strings.NewReader(anthropicBodyModel("claude-any", false)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", "k")
	req.Header.Set("anthropic-version", "2023-06-01")
	req.Header.Set("X-MockLM-Scenario", "anth-only")
	r2, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	var out struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
	}
	json.NewDecoder(r2.Body).Decode(&out)
	r2.Body.Close()
	if len(out.Content) == 0 || out.Content[0].Text != "anthropic exact" {
		t.Fatalf("header-targeted response = %+v, want the exact text", out)
	}

	// K2: the same scenario on the OpenAI route is a 409 — a scenario
	// cannot change the wire shape of a route it did not route to.
	resp, err = postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), map[string]string{"X-MockLM-Scenario": "anth-only"})
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 409 {
		t.Fatalf("provider mismatch: got %d, want 409", resp.StatusCode)
	}

	// Unknown scenario id is a loud 404.
	resp, err = postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), map[string]string{"X-MockLM-Scenario": "no-such"})
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 404 {
		t.Fatalf("unknown scenario: got %d, want 404", resp.StatusCode)
	}

	// R4: a targeting header on an unwired surface is a loud 400, not a
	// silent no-op.
	for _, path := range []string{"/v1/responses", "/v1/completions", "/v1/embeddings"} {
		resp, err = postJSON(srv.URL+path, `{"model":"gpt-4","input":"x"}`, map[string]string{"X-MockLM-Scenario": "anth-only"})
		if err != nil {
			t.Fatalf("request failed: %v", err)
		}
		resp.Body.Close()
		if resp.StatusCode != 400 {
			t.Fatalf("scenario header on %s: got %d, want 400 (R4)", path, resp.StatusCode)
		}
	}
}

// --- per-scenario capture + counters ---

func TestScenarioCaptureCountersAndLifecycle(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	body, _ := json.Marshal(map[string]any{
		"id": "cap", "provider": "openai", "model": "gpt-cap",
		"output": map[string]any{"text": "captured"},
	})
	mustRegisterScenario(t, srv.URL, string(body))

	// Two matched requests — the second with deliberately odd whitespace
	// the capture must preserve byte-for-byte — plus one non-matching.
	first := openaiChatBodyModel("gpt-cap", false)
	second := "{\"model\":\"gpt-cap\",  \"stream\":false,\n\t\"messages\":[{\"role\":\"user\",\"content\":\"exact  bytes\"}]}"
	for _, b := range []string{first, second} {
		resp, err := postJSON(srv.URL+"/v1/chat/completions", b, nil)
		if err != nil {
			t.Fatalf("request failed: %v", err)
		}
		resp.Body.Close()
	}
	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil) // gpt-4: no match
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	resp.Body.Close()

	// last-request returns the second body byte-exact.
	lr, err := http.Get(srv.URL + "/admin/scenarios/cap/last-request")
	if err != nil {
		t.Fatalf("last-request failed: %v", err)
	}
	raw, _ := io.ReadAll(lr.Body)
	lr.Body.Close()
	if string(raw) != second {
		t.Fatalf("captured body = %q, want byte-exact %q", raw, second)
	}
	if m := lr.Header.Get("X-MockLM-Captured-Method"); m != "POST" {
		t.Fatalf("captured method = %q", m)
	}

	// request-count = matched-raw-requests (2), independent of the
	// provider-global attempt counter (3 openai requests).
	var rc struct {
		Count int64 `json:"count"`
	}
	getJSON(t, srv.URL+"/admin/scenarios/cap/request-count", &rc)
	if rc.Count != 2 {
		t.Fatalf("scenario request-count = %d, want 2", rc.Count)
	}
	var attempts struct {
		OpenAI  int64 `json:"openai"`
		Bedrock int64 `json:"bedrock"`
	}
	getJSON(t, srv.URL+"/admin/request-count", &attempts)
	if attempts.OpenAI != 3 {
		t.Fatalf("provider attempt counter = %d, want 3 (K6: capture must not touch it)", attempts.OpenAI)
	}

	// Per-scenario reset zeroes only this scenario's counter.
	resp, err = postJSON(srv.URL+"/admin/scenarios/cap/request-count/reset", "", nil)
	if err != nil {
		t.Fatalf("reset failed: %v", err)
	}
	resp.Body.Close()
	getJSON(t, srv.URL+"/admin/scenarios/cap/request-count", &rc)
	if rc.Count != 0 {
		t.Fatalf("after reset, request-count = %d, want 0", rc.Count)
	}

	// R6: POST /admin/reset does NOT wipe scenarios...
	resp, err = postJSON(srv.URL+"/admin/reset", "", nil)
	if err != nil {
		t.Fatalf("admin reset failed: %v", err)
	}
	resp.Body.Close()
	gr, err := http.Get(srv.URL + "/admin/scenarios/cap")
	if err != nil {
		t.Fatalf("get scenario failed: %v", err)
	}
	gr.Body.Close()
	if gr.StatusCode != 200 {
		t.Fatalf("scenario gone after /admin/reset: %d (R6: scenarios have independent lifetimes)", gr.StatusCode)
	}

	// ...DELETE /admin/scenarios is the lifecycle boundary.
	req, _ := http.NewRequest("DELETE", srv.URL+"/admin/scenarios", nil)
	dr, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("clear failed: %v", err)
	}
	dr.Body.Close()
	gr, err = http.Get(srv.URL + "/admin/scenarios/cap")
	if err != nil {
		t.Fatalf("get scenario failed: %v", err)
	}
	gr.Body.Close()
	if gr.StatusCode != 404 {
		t.Fatalf("scenario still present after DELETE /admin/scenarios: %d", gr.StatusCode)
	}
}

func getJSON(t *testing.T, url string, out any) {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		t.Fatalf("GET %s failed: %v", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("GET %s returned %d: %s", url, resp.StatusCode, raw)
	}
	if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
		t.Fatalf("GET %s: invalid JSON: %v", url, err)
	}
}

// --- scenario faults + presets ---

func TestScenarioFaultsAndHeaderOverlay(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	// attempt_faults on the scenario config index the PROVIDER-GLOBAL
	// counter (K1/R6): attempt 0 on openai fails, attempt 1 succeeds.
	body, _ := json.Marshal(map[string]any{
		"id": "flt", "provider": "openai", "model": "gpt-flt",
		"output": map[string]any{"text": "ok"},
		"config": map[string]any{
			"attempt_faults": [][]map[string]any{
				{{"mode": "error", "error_status": 503}},
				{},
			},
		},
	})
	mustRegisterScenario(t, srv.URL, string(body))

	for i, want := range []int{503, 200} {
		resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBodyModel("gpt-flt", false), nil)
		if err != nil {
			t.Fatalf("request %d failed: %v", i, err)
		}
		resp.Body.Close()
		if resp.StatusCode != want {
			t.Fatalf("attempt %d: got %d, want %d", i, resp.StatusCode, want)
		}
	}

	// X-MockLM-Fault still overlays on top of a matched scenario's config.
	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBodyModel("gpt-flt", false),
		map[string]string{"X-MockLM-Fault": `{"error_rate":1.0,"error_status":418}`})
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 418 {
		t.Fatalf("fault-header overlay on scenario: got %d, want 418", resp.StatusCode)
	}
}

func TestScenarioFaultPresetBase(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	// fault_preset seeds the config; the scenario's own config overlays.
	body := `{"id":"preset-based","provider":"openai","model":"gpt-preset",` +
		`"output":{"text":"x"},"fault_preset":"mid-stream-cut","config":{"tokens":9}}`
	mustRegisterScenario(t, srv.URL, body)

	var def struct {
		Config struct {
			Tokens int         `json:"tokens"`
			Faults []FaultSpec `json:"faults"`
		} `json:"config"`
	}
	getJSON(t, srv.URL+"/admin/scenarios/preset-based", &def)
	if len(def.Config.Faults) != 1 || def.Config.Faults[0].Mode != "disconnect" || def.Config.Faults[0].AfterN != 3 {
		t.Fatalf("preset faults not carried into the scenario config: %+v", def.Config.Faults)
	}
	if def.Config.Tokens != 9 {
		t.Fatalf("config overlay lost tokens=9: %+v", def.Config)
	}
}

// K7: replacement swaps a brand-new *Scenario under the write lock and
// requests hold snapshots — hammering re-registration while matched
// requests stream must be race-free (this test is only meaningful under
// -race, which CI runs).
func TestScenarioStoreConcurrentReplaceWhileServing(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	reg := func(text string) {
		body, _ := json.Marshal(map[string]any{
			"id": "race", "provider": "openai", "model": "gpt-race",
			"output": map[string]any{"text": text, "chunking": map[string]any{"mode": "runes", "size": 2}},
		})
		resp := registerScenario(t, srv.URL, string(body), "")
		resp.Body.Close()
	}
	reg("initial payload")

	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := range 25 {
			reg(fmt.Sprintf("replacement %d with  spacing\n", i))
		}
	}()
	for range 25 {
		got, _ := collectOpenAIDeltas(t, srv.URL, openaiChatBodyModel("gpt-race", true), nil)
		if got == "" {
			t.Fatalf("a matched stream reassembled to empty text")
		}
	}
	<-done
}

// --- fault catalog + fault presets admin surface (Phase 3d/4.4) ---

func TestAdminFaultCatalogAndPresets(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	var catalog struct {
		Faults []FaultCatalogEntry `json:"faults"`
	}
	getJSON(t, srv.URL+"/admin/faults", &catalog)
	byMode := map[string]FaultCatalogEntry{}
	for _, e := range catalog.Faults {
		byMode[e.Mode] = e
	}
	for _, mode := range []string{"error", "non_json_body", "disconnect", "stall", "malformed_chunk", "unknown_event", "unknown_block", "stream_error"} {
		if _, ok := byMode[mode]; !ok {
			t.Fatalf("catalog is missing mode %q", mode)
		}
	}
	if !byMode["unknown_event"].RequiresValidateOff || !byMode["stream_error"].RequiresValidateOff {
		t.Fatalf("decoder-fault modes must be flagged requires_validate_responses_false")
	}
	if byMode["error"].Phase != "pre-body" || byMode["stall"].Phase != "stream" {
		t.Fatalf("catalog phases wrong: error=%q stall=%q", byMode["error"].Phase, byMode["stall"].Phase)
	}

	var presets struct {
		FaultPresets []FaultPreset `json:"fault_presets"`
	}
	getJSON(t, srv.URL+"/admin/fault-presets", &presets)
	names := map[string]bool{}
	for _, p := range presets.FaultPresets {
		names[p.Name] = true
	}
	for _, want := range []string{"retry-storm", "decoder-probe", "usage-omit", "transport-crlf-coalesce", "mid-stream-cut"} {
		if !names[want] {
			t.Fatalf("fault presets missing %q (got %v)", want, names)
		}
	}
}

// --- /v1/responses stream in the fault loop (Phase 3d/4.2a) ---

func TestResponsesStreamFaultInjection(t *testing.T) {
	// A stream-phase disconnect spec must now fire on the Responses stream
	// (it rode the generic injector into handleResponsesStream).
	cfg := defaultConfig()
	cfg.OpenAI.Faults = []FaultSpec{{Mode: "disconnect", AfterN: 2}}
	srv := testServer(cfg)
	defer srv.Close()

	resp, err := postJSON(srv.URL+"/v1/responses", `{"model":"gpt-4","stream":true,"input":[{"role":"user","content":"hi"}]}`, nil)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()
	raw, readErr := io.ReadAll(resp.Body)
	if readErr == nil && strings.Contains(string(raw), "response.completed") {
		t.Fatalf("stream ran to completion; the after_n=2 disconnect never fired")
	}

	// Transport faults reach the Responses stream too: CRLF framing shows
	// up as \r\n line endings on the wire.
	cfg2 := defaultConfig()
	cfg2.OpenAI.CrlfFrames = true
	srv2 := testServer(cfg2)
	defer srv2.Close()
	resp2, err := postJSON(srv2.URL+"/v1/responses", `{"model":"gpt-4","stream":true,"input":[{"role":"user","content":"hi"}]}`, nil)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp2.Body.Close()
	raw2, _ := io.ReadAll(resp2.Body)
	if !strings.Contains(string(raw2), "\r\n") {
		t.Fatalf("crlf_frames did not reach the Responses stream")
	}
}
