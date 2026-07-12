package main

import (
	"encoding/json"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"
)

// --- "delay" fault mode: non-terminal pre-body wait ---
//
// delay holds delay_ms cancelably, then CONTINUES normal handling — the
// only non-terminal fault mode. With attempt_faults and the per-scenario
// attempt counter it expresses deterministic per-attempt timing sequences
// ("first request waits 5s, second is instant") that parallel background
// traffic cannot shift.

func TestDelayFaultContinuesNormally(t *testing.T) {
	cfg := defaultConfig()
	cfg.OpenAI.Faults = []FaultSpec{{Mode: "delay", DelayMs: 120}}
	srv := testServer(cfg)
	defer srv.Close()

	start := time.Now()
	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("delay is non-terminal: expected 200, got %d", resp.StatusCode)
	}
	if elapsed := time.Since(start); elapsed < 120*time.Millisecond {
		t.Fatalf("response arrived in %v, before the 120ms delay elapsed", elapsed)
	}
	var body map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("delayed response is not a normal completion body: %v", err)
	}
	if body["object"] != "chat.completion" {
		t.Fatalf("delayed response object = %v, want chat.completion", body["object"])
	}
}

func TestDelayFaultAppliesToStreams(t *testing.T) {
	// The delay is pre-body, so a streaming request waits it out and then
	// streams normally.
	cfg := defaultConfig()
	cfg.OpenAI.Faults = []FaultSpec{{Mode: "delay", DelayMs: 100}}
	srv := testServer(cfg)
	defer srv.Close()

	start := time.Now()
	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(true), nil)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()
	raw := make([]byte, 4096)
	n, _ := resp.Body.Read(raw)
	if elapsed := time.Since(start); elapsed < 100*time.Millisecond {
		t.Fatalf("first stream bytes arrived in %v, before the 100ms delay", elapsed)
	}
	if !strings.Contains(string(raw[:n]), "data: ") {
		t.Fatalf("expected SSE data after the delay, got %q", raw[:n])
	}
}

func TestDelayComposesWithTerminalError(t *testing.T) {
	// Declaration order: the delay waits, then the terminal error fires.
	cfg := defaultConfig()
	cfg.OpenAI.Faults = []FaultSpec{
		{Mode: "delay", DelayMs: 80},
		{Mode: "error", ErrorStatus: 503},
	}
	srv := testServer(cfg)
	defer srv.Close()

	start := time.Now()
	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 503 {
		t.Fatalf("terminal error after delay: expected 503, got %d", resp.StatusCode)
	}
	if elapsed := time.Since(start); elapsed < 80*time.Millisecond {
		t.Fatalf("503 arrived in %v, before the 80ms delay", elapsed)
	}
}

func TestDelayFaultValidation(t *testing.T) {
	for _, tc := range []struct {
		name string
		spec FaultSpec
	}{
		{"missing delay_ms", FaultSpec{Mode: "delay"}},
		{"non-positive delay_ms", FaultSpec{Mode: "delay", DelayMs: -5}},
		{"stream WHEN after_n", FaultSpec{Mode: "delay", DelayMs: 10, AfterN: 1}},
		{"stream WHEN after_event", FaultSpec{Mode: "delay", DelayMs: 10, AfterEvent: "message_delta"}},
		{"stream WHEN after_bytes", FaultSpec{Mode: "delay", DelayMs: 10, AfterBytes: 1}},
		{"after_ms confusion", FaultSpec{Mode: "delay", DelayMs: 10, AfterMs: 10}},
	} {
		if err := validateFaultSpec(&tc.spec); err == nil {
			t.Errorf("%s: expected a validation error, got nil", tc.name)
		}
	}
	ok := FaultSpec{Mode: "delay", DelayMs: 10}
	if err := validateFaultSpec(&ok); err != nil {
		t.Errorf("plain delay spec should validate, got %v", err)
	}

	// A bad spec is a loud 400 at request time via X-MockLM-Fault too.
	srv := testServer(defaultConfig())
	defer srv.Close()
	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false),
		map[string]string{"X-MockLM-Fault": `{"faults":[{"mode":"delay"}]}`})
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 400 {
		t.Fatalf("invalid delay spec via header: expected 400, got %d", resp.StatusCode)
	}
}

func TestDelayCancellationFreesHandler(t *testing.T) {
	// A client that times out during the delay unblocks the handler:
	// srv.Close() below would hang on a leaked one.
	cfg := defaultConfig()
	cfg.OpenAI.Faults = []FaultSpec{{Mode: "delay", DelayMs: 30_000}}
	srv := testServer(cfg)
	defer srv.Close()

	client := &http.Client{Timeout: 200 * time.Millisecond}
	req, _ := http.NewRequest("POST", srv.URL+"/v1/chat/completions", strings.NewReader(openaiChatBody(false)))
	req.Header.Set("Content-Type", "application/json")
	start := time.Now()
	_, err := client.Do(req)
	if err == nil {
		t.Fatalf("expected the client timeout to fire during the 30s delay")
	}
	if time.Since(start) > 5*time.Second {
		t.Fatalf("client timeout took %v — the delay did not cancel", time.Since(start))
	}
}

// TestScenarioDelaySequenceUnderParallelLoad is the consumer's exact use
// case: a per-scenario timing sequence — first attempt slow, second
// instant — deterministic while background traffic hammers the provider.
func TestScenarioDelaySequenceUnderParallelLoad(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	body, _ := json.Marshal(map[string]any{
		"id": "slow-first", "provider": "openai", "model": "gpt-slow",
		"output": map[string]any{"text": "ok"},
		"config": map[string]any{
			"attempt_faults": [][]map[string]any{
				{{"mode": "delay", "delay_ms": 400}},
				{},
			},
		},
	})
	mustRegisterScenario(t, srv.URL, string(body))

	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() { // background traffic racing the scenario's sequence
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
			if err != nil {
				return
			}
			resp.Body.Close()
		}
	}()

	timeAttempt := func() time.Duration {
		t.Helper()
		start := time.Now()
		resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBodyModel("gpt-slow", false), nil)
		if err != nil {
			t.Fatalf("scenario request failed: %v", err)
		}
		resp.Body.Close()
		if resp.StatusCode != 200 {
			t.Fatalf("scenario request status %d", resp.StatusCode)
		}
		return time.Since(start)
	}

	if first := timeAttempt(); first < 400*time.Millisecond {
		t.Fatalf("attempt 1 returned in %v, before its 400ms delay", first)
	}
	if second := timeAttempt(); second >= 400*time.Millisecond {
		t.Fatalf("attempt 2 took %v — the delay leaked past the first attempt (or background traffic shifted the sequence)", second)
	}
	close(stop)
	wg.Wait()

	// Background traffic must never absorb the scenario's delay: with the
	// scenario's sequence exhausted, a fresh background request is fast.
	start := time.Now()
	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
	if err != nil {
		t.Fatalf("background request failed: %v", err)
	}
	resp.Body.Close()
	if elapsed := time.Since(start); elapsed >= 400*time.Millisecond {
		t.Fatalf("background request took %v — a scenario delay leaked onto provider traffic", elapsed)
	}
}
