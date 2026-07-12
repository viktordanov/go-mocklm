package main

import (
	"encoding/json"
	"net/http"
	"sync"
	"testing"
)

// --- Per-scenario fault-attempt counters ---
//
// A matched scenario's fail_first_n / attempt_faults index the scenario's
// OWN counter (exposed at /admin/scenarios/{id}/attempt-count), not the
// provider counter — so per-scenario retry tests are deterministic under
// parallel load. The provider counter (/admin/request-count) still counts
// every request, scenario traffic included. These tests are only meaningful
// under -race (CI enforces it).

func scenarioWithFailFirst(id, model string) string {
	body, _ := json.Marshal(map[string]any{
		"id": id, "provider": "openai", "model": model,
		"output": map[string]any{"text": "ok"},
		"config": map[string]any{"fail_first_n": 1, "error_status": 503},
	})
	return string(body)
}

func scenarioAttemptCount(t *testing.T, srvURL, id string) int64 {
	t.Helper()
	var rc struct {
		Count int64 `json:"count"`
	}
	getJSON(t, srvURL+"/admin/scenarios/"+id+"/attempt-count", &rc)
	return rc.Count
}

func providerRequestCount(t *testing.T, srvURL string) int64 {
	t.Helper()
	var counts struct {
		OpenAI int64 `json:"openai"`
	}
	getJSON(t, srvURL+"/admin/request-count", &counts)
	return counts.OpenAI
}

// TestScenarioAttemptIsolationUnderParallelLoad is the headline guarantee:
// two scenarios on ONE provider, each fail_first_n=1, hammered concurrently
// alongside non-scenario background traffic. Each scenario fails exactly
// its own first attempt; the provider counter sees all traffic. Scenario A
// is model-matched, scenario B header-matched, covering both match paths.
func TestScenarioAttemptIsolationUnderParallelLoad(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	mustRegisterScenario(t, srv.URL, scenarioWithFailFirst("iso-a", "gpt-iso-a"))
	mustRegisterScenario(t, srv.URL, scenarioWithFailFirst("iso-b", "gpt-iso-b"))

	const perLane = 6 // requests per scenario and per background lane
	statuses := make(map[string][]int, 3)
	var mu sync.Mutex
	var wg sync.WaitGroup
	record := func(lane string, code int) {
		mu.Lock()
		defer mu.Unlock()
		statuses[lane] = append(statuses[lane], code)
	}
	for i := 0; i < perLane; i++ {
		wg.Add(3)
		go func() { // scenario A by model match
			defer wg.Done()
			resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBodyModel("gpt-iso-a", false), nil)
			if err != nil {
				t.Errorf("iso-a request failed: %v", err)
				return
			}
			resp.Body.Close()
			record("a", resp.StatusCode)
		}()
		go func() { // scenario B by header match
			defer wg.Done()
			resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false),
				map[string]string{"X-MockLM-Scenario": "iso-b"})
			if err != nil {
				t.Errorf("iso-b request failed: %v", err)
				return
			}
			resp.Body.Close()
			record("b", resp.StatusCode)
		}()
		go func() { // non-scenario background traffic on the same provider
			defer wg.Done()
			resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
			if err != nil {
				t.Errorf("background request failed: %v", err)
				return
			}
			resp.Body.Close()
			record("bg", resp.StatusCode)
		}()
	}
	wg.Wait()

	for _, lane := range []string{"a", "b"} {
		fails := 0
		for _, code := range statuses[lane] {
			if code == 503 {
				fails++
			} else if code != 200 {
				t.Fatalf("lane %s: unexpected status %d (want only 503/200): %v", lane, code, statuses[lane])
			}
		}
		if fails != 1 {
			t.Fatalf("lane %s: fail_first_n=1 must fail EXACTLY one attempt, got %d failures: %v", lane, fails, statuses[lane])
		}
	}
	for _, code := range statuses["bg"] {
		if code != 200 {
			t.Fatalf("background traffic must never absorb a scenario's fault, got %d: %v", code, statuses["bg"])
		}
	}

	if got := scenarioAttemptCount(t, srv.URL, "iso-a"); got != perLane {
		t.Fatalf("iso-a attempt-count = %d, want %d", got, perLane)
	}
	if got := scenarioAttemptCount(t, srv.URL, "iso-b"); got != perLane {
		t.Fatalf("iso-b attempt-count = %d, want %d", got, perLane)
	}
	if got := providerRequestCount(t, srv.URL); got != 3*perLane {
		t.Fatalf("provider request-count = %d, want %d (all traffic, scenario-matched included)", got, 3*perLane)
	}
}

// TestScenarioAttemptFaultSequenceIndependentOfBackground drives one
// scenario's attempt_faults sequence in order while parallel background
// traffic hammers the same provider: the sequence must come out exactly
// 503, 429, 200 — background requests cannot shift it.
func TestScenarioAttemptFaultSequenceIndependentOfBackground(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	body, _ := json.Marshal(map[string]any{
		"id": "seq", "provider": "openai", "model": "gpt-seq",
		"output": map[string]any{"text": "ok"},
		"config": map[string]any{
			"attempt_faults": [][]map[string]any{
				{{"mode": "error", "error_status": 503}},
				{{"mode": "error", "error_status": 429}},
				{},
			},
		},
	})
	mustRegisterScenario(t, srv.URL, string(body))

	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() { // background load, racing the scenario's sequential attempts
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

	for i, want := range []int{503, 429, 200, 200} {
		resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBodyModel("gpt-seq", false), nil)
		if err != nil {
			t.Fatalf("attempt %d failed: %v", i, err)
		}
		resp.Body.Close()
		if resp.StatusCode != want {
			t.Fatalf("attempt %d: got %d, want %d (background traffic must not shift the scenario's sequence)", i, resp.StatusCode, want)
		}
	}
	close(stop)
	wg.Wait()
}

// TestScenarioAttemptResetIsolation proves resetting one scenario's attempt
// counter re-arms only that scenario: the sibling's counter and the
// provider counter are untouched.
func TestScenarioAttemptResetIsolation(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	mustRegisterScenario(t, srv.URL, scenarioWithFailFirst("rst-a", "gpt-rst-a"))
	mustRegisterScenario(t, srv.URL, scenarioWithFailFirst("rst-b", "gpt-rst-b"))

	hit := func(model string, want int) {
		t.Helper()
		resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBodyModel(model, false), nil)
		if err != nil {
			t.Fatalf("request failed: %v", err)
		}
		resp.Body.Close()
		if resp.StatusCode != want {
			t.Fatalf("%s: got %d, want %d", model, resp.StatusCode, want)
		}
	}

	hit("gpt-rst-a", 503) // a attempt 1
	hit("gpt-rst-a", 200) // a attempt 2
	hit("gpt-rst-b", 503) // b attempt 1
	providerBefore := providerRequestCount(t, srv.URL)

	resp, err := postJSON(srv.URL+"/admin/scenarios/rst-a/attempt-count/reset", "", nil)
	if err != nil {
		t.Fatalf("attempt-count reset failed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("attempt-count reset returned %d", resp.StatusCode)
	}

	if got := scenarioAttemptCount(t, srv.URL, "rst-a"); got != 0 {
		t.Fatalf("rst-a attempt-count after reset = %d, want 0", got)
	}
	if got := scenarioAttemptCount(t, srv.URL, "rst-b"); got != 1 {
		t.Fatalf("rst-b attempt-count = %d, want 1 (sibling reset must not leak)", got)
	}
	if got := providerRequestCount(t, srv.URL); got != providerBefore {
		t.Fatalf("provider request-count changed by a scenario attempt reset: %d -> %d", providerBefore, got)
	}

	hit("gpt-rst-a", 503) // re-armed: attempt 1 again
	hit("gpt-rst-b", 200) // b unaffected: attempt 2 succeeds
}

// TestScenarioAttemptsSurviveAdminReset: POST /admin/reset resets provider
// config and provider counters but must NOT re-arm a scenario's attempt
// faults (R6). Replacement of the scenario DOES re-arm it.
func TestScenarioAttemptsSurviveAdminReset(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	mustRegisterScenario(t, srv.URL, scenarioWithFailFirst("arm", "gpt-arm"))

	hit := func(want int) {
		t.Helper()
		resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBodyModel("gpt-arm", false), nil)
		if err != nil {
			t.Fatalf("request failed: %v", err)
		}
		resp.Body.Close()
		if resp.StatusCode != want {
			t.Fatalf("got %d, want %d", resp.StatusCode, want)
		}
	}

	hit(503) // attempt 1 fires the fault
	hit(200) // attempt 2 succeeds

	resp, err := postJSON(srv.URL+"/admin/reset", "", nil)
	if err != nil {
		t.Fatalf("admin reset failed: %v", err)
	}
	resp.Body.Close()

	hit(200) // NOT re-armed: /admin/reset leaves scenario counters alone
	if got := scenarioAttemptCount(t, srv.URL, "arm"); got != 3 {
		t.Fatalf("attempt-count after /admin/reset = %d, want 3 (must survive)", got)
	}

	// Replacement swaps in a fresh *Scenario — fresh counter, re-armed.
	mustRegisterScenario(t, srv.URL, scenarioWithFailFirst("arm", "gpt-arm"))
	if got := scenarioAttemptCount(t, srv.URL, "arm"); got != 0 {
		t.Fatalf("attempt-count after replacement = %d, want 0 (fresh counter)", got)
	}
	hit(503) // attempt 1 of the replacement fires again
	hit(200)
}

// TestScenarioAttemptCountEndpointErrors: unknown ids are loud 404s on both
// the getter and the reset.
func TestScenarioAttemptCountEndpointErrors(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/admin/scenarios/nope/attempt-count")
	if err != nil {
		t.Fatalf("GET failed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 404 {
		t.Fatalf("GET unknown attempt-count: got %d, want 404", resp.StatusCode)
	}
	resp, err = postJSON(srv.URL+"/admin/scenarios/nope/attempt-count/reset", "", nil)
	if err != nil {
		t.Fatalf("POST failed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 404 {
		t.Fatalf("POST unknown attempt-count/reset: got %d, want 404", resp.StatusCode)
	}
}

// TestHeaderLevelAttemptFaultsStayProviderIndexed: without a scenario, the
// fault-attempt sequence still rides the provider counter — the historical
// contract for header- and provider-level attempt_faults is unchanged.
func TestHeaderLevelAttemptFaultsStayProviderIndexed(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	h := map[string]string{
		"X-MockLM-Fault": `{"attempt_faults":[[],[{"mode":"error","error_status":503}]]}`,
	}
	// A plain request advances the provider counter to 1...
	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	resp.Body.Close()
	// ...so the header-fault request lands on attempt 2 and draws
	// attempt_faults[1]: provider-indexed, shiftable by unrelated traffic —
	// exactly why scenarios get their own counter.
	resp, err = postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), h)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 503 {
		t.Fatalf("header-level attempt_faults[1] should fire on provider attempt 2, got %d", resp.StatusCode)
	}
}
