package main

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"
)

// --- Phase 2a: two-knob fault engine (WHEN × HOW), per-attempt fault
// arrays, request-count introspection, and the decoder-fault group
// (B1 unknown_event / B2 unknown_block / B5 stream_error) ---

// anthropicStreamEventsWithFaults drives a streaming Anthropic request
// against a server configured with the given fault specs (validation off —
// decoder faults are off-vocabulary by design) and returns the parsed
// event list.
func anthropicStreamEventsWithFaults(t *testing.T, faults []FaultSpec, mutate func(*Config)) []streamEvent {
	t.Helper()
	cfg := defaultConfig()
	cfg.Anthropic.Faults = faults
	cfg.Anthropic.ValidateResponses = boolPtr(false)
	if mutate != nil {
		mutate(cfg)
	}
	srv := testServer(cfg)
	defer srv.Close()
	return collectAnthropicEvents(t, srv.URL, anthropicBody(true))
}

func eventNames(events []streamEvent) []string {
	names := make([]string, len(events))
	for i, ev := range events {
		names[i] = ev.name
	}
	return names
}

// --- Fault engine: per-attempt arrays + introspection ---

func TestAttemptFaultsFirstAttemptFailsSecondSucceeds(t *testing.T) {
	// The canonical retry/fallback scenario without header targeting:
	// attempt 0 fails with a 503, attempt 1 (and everything after the
	// array) succeeds.
	cfg := defaultConfig()
	cfg.OpenAI.AttemptFaults = [][]FaultSpec{
		{{Mode: "error", ErrorStatus: 503, ErrorMessage: "attempt-0 fails"}},
		{},
	}
	srv := testServer(cfg)
	defer srv.Close()

	for i, wantStatus := range []int{503, 200, 200} {
		resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
		if err != nil {
			t.Fatalf("request %d failed: %v", i, err)
		}
		if resp.StatusCode != wantStatus {
			t.Fatalf("request %d: expected %d, got %d", i, wantStatus, resp.StatusCode)
		}
		if wantStatus == 503 {
			var body map[string]any
			json.NewDecoder(resp.Body).Decode(&body)
			errObj, _ := body["error"].(map[string]any)
			if errObj["message"] != "attempt-0 fails" || errObj["type"] != "server_error" {
				t.Fatalf("attempt-0 error envelope wrong: %v", body)
			}
		}
		resp.Body.Close()
	}
}

func TestAttemptFaultsViaFaultHeader(t *testing.T) {
	// attempt_faults can ride the X-MockLM-Fault header; the attempt
	// counter is server-side, so the same header on the retry hits the
	// empty second entry and succeeds.
	srv := testServer(defaultConfig())
	defer srv.Close()

	h := map[string]string{
		"X-MockLM-Fault": `{"attempt_faults":[[{"mode":"error","error_status":429}]]}`,
	}
	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), h)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 429 {
		t.Fatalf("attempt 0 should 429, got %d", resp.StatusCode)
	}
	var body map[string]any
	json.NewDecoder(resp.Body).Decode(&body)
	if errObj, _ := body["error"].(map[string]any); errObj["type"] != "rate_limit_error" {
		t.Fatalf("429 should derive rate_limit_error, got %v", body)
	}

	resp2, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), h)
	if err != nil {
		t.Fatalf("retry failed: %v", err)
	}
	defer resp2.Body.Close()
	if resp2.StatusCode != 200 {
		t.Fatalf("attempt 1 should succeed, got %d", resp2.StatusCode)
	}
}

func TestRequestCountIntrospection(t *testing.T) {
	srv, state := testServerWithState(defaultConfig())
	defer srv.Close()

	getCounts := func() (float64, float64) {
		t.Helper()
		resp, err := http.Get(srv.URL + "/admin/request-count")
		if err != nil {
			t.Fatalf("GET /admin/request-count failed: %v", err)
		}
		defer resp.Body.Close()
		var counts map[string]float64
		if err := json.NewDecoder(resp.Body).Decode(&counts); err != nil {
			t.Fatalf("decode failed: %v", err)
		}
		return counts["openai"], counts["anthropic"]
	}

	for i := 0; i < 2; i++ {
		resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
		if err != nil {
			t.Fatalf("request failed: %v", err)
		}
		resp.Body.Close()
	}
	postAnthropic(t, srv.URL, anthropicBody(false)).Body.Close()

	if o, a := getCounts(); o != 2 || a != 1 {
		t.Fatalf("expected openai=2 anthropic=1, got openai=%v anthropic=%v", o, a)
	}

	// Explicit reset zeroes both counters.
	resp, err := http.Post(srv.URL+"/admin/request-count/reset", "application/json", nil)
	if err != nil {
		t.Fatalf("reset failed: %v", err)
	}
	resp.Body.Close()
	if o, a := getCounts(); o != 0 || a != 0 {
		t.Fatalf("counters should be zero after reset, got openai=%v anthropic=%v", o, a)
	}

	// A config update resets too, so each scenario starts counting fresh.
	postAnthropic(t, srv.URL, anthropicBody(false)).Body.Close()
	snap, _ := state.Config()
	state.Update(snap.OpenAI, snap.Anthropic, snap.Bedrock, "custom")
	if o, a := getCounts(); o != 0 || a != 0 {
		t.Fatalf("counters should reset on config update, got openai=%v anthropic=%v", o, a)
	}
}

func TestFaultSpecErrorModeCannotFireMidStream(t *testing.T) {
	// "error" with a stream-phase WHEN is a contradiction (the status is
	// locked once the body starts) and must be rejected loudly, not
	// silently test nothing.
	cfg := defaultConfig()
	cfg.OpenAI.Faults = []FaultSpec{{Mode: "error", AfterEvent: "message_delta"}}
	srv := testServer(cfg)
	defer srv.Close()

	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 400 {
		t.Fatalf("expected 400 for invalid fault spec, got %d", resp.StatusCode)
	}
}

func TestFaultSpecUnknownModeRejected(t *testing.T) {
	cfg := defaultConfig()
	cfg.Anthropic.Faults = []FaultSpec{{Mode: "explode"}}
	srv := testServer(cfg)
	defer srv.Close()

	resp := postAnthropic(t, srv.URL, anthropicBody(false))
	defer resp.Body.Close()
	if resp.StatusCode != 400 {
		t.Fatalf("expected 400 for unknown fault mode, got %d", resp.StatusCode)
	}
}

func TestFaultSpecDisconnectAfterEvent(t *testing.T) {
	// The generalized engine expresses the Phase-0 disconnect_after_event
	// knob as {mode: disconnect, after_event: ...}: the stream carries
	// message_delta and is severed before message_stop.
	cfg := defaultConfig()
	cfg.Anthropic.Faults = []FaultSpec{{Mode: "disconnect", AfterEvent: "message_delta"}}
	srv := testServer(cfg)
	defer srv.Close()

	raw := readStreamUntilClosed(t, srv.URL, anthropicBody(true))
	if !strings.Contains(raw, "event: message_delta") {
		t.Fatalf("stream should reach message_delta, got: %s", raw)
	}
	if strings.Contains(raw, "event: message_stop") {
		t.Fatalf("stream should be cut before message_stop, got: %s", raw)
	}
}

func TestFaultSpecDisconnectAfterNChunksOpenAI(t *testing.T) {
	// after_n on an OpenAI stream counts data frames; the stream dies
	// without [DONE].
	cfg := defaultConfig()
	cfg.OpenAI.Tokens = 10
	cfg.OpenAI.Faults = []FaultSpec{{Mode: "disconnect", AfterN: 3}}
	srv := testServer(cfg)
	defer srv.Close()

	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(true), nil)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
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
	body := raw.String()
	// The RST may discard flushed-but-unread frames on the client side, so
	// assert the deterministic upper bound and that the stream was severed.
	if got := strings.Count(body, "data: "); got > 3 {
		t.Fatalf("expected at most 3 data frames before the RST, got %d: %s", got, body)
	}
	if strings.Contains(body, "[DONE]") {
		t.Fatalf("severed stream must not carry [DONE]: %s", body)
	}
}

func TestWaitCancelableReturnsOnCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	start := time.Now()
	if waitCancelable(ctx, 5000) {
		t.Fatal("canceled context should return false")
	}
	if time.Since(start) > time.Second {
		t.Fatal("canceled wait should return immediately")
	}
	if !waitCancelable(context.Background(), 0) {
		t.Fatal("zero wait should return true")
	}
}

// --- B1: unknown top-level event ---

func TestUnknownEventInjection(t *testing.T) {
	events := anthropicStreamEventsWithFaults(t, []FaultSpec{
		{Mode: "unknown_event", EventType: "message_future", AfterEvent: "content_block_start", Repeat: 2},
	}, nil)

	future := 0
	for _, ev := range events {
		if ev.name == "message_future" {
			future++
			if ev.data["type"] != "message_future" {
				t.Fatalf("payload type must match the SSE event name, got %v", ev.data)
			}
		}
	}
	if future != 2 {
		t.Fatalf("expected 2 message_future events (repeat: 2), got %d in %v", future, eventNames(events))
	}

	// The surrounding stream is intact: it still ends in
	// message_delta + message_stop with text deltas in between.
	names := eventNames(events)
	if names[len(names)-1] != "message_stop" {
		t.Fatalf("stream should complete normally, got %v", names)
	}
}

// --- B2: unknown content_block types (real spec type + invented) ---

func TestUnknownBlockInjectionRealAndInventedTypes(t *testing.T) {
	events := anthropicStreamEventsWithFaults(t, []FaultSpec{
		{Mode: "unknown_block", BlockType: "redacted_thinking", AfterEvent: "message_start"},
		{Mode: "unknown_block", BlockType: "server_tool_use", AfterEvent: "message_start"},
		{Mode: "unknown_block", BlockType: "x_mock_novel_block", AfterEvent: "message_start"},
	}, nil)

	// Every injected block is complete (start + stop with the same index)
	// and well-formed.
	sawStart := map[string]float64{}
	openStops := map[float64]bool{}
	for _, ev := range events {
		switch ev.name {
		case "content_block_start":
			block, _ := ev.data["content_block"].(map[string]any)
			bt, _ := block["type"].(string)
			idx, _ := ev.data["index"].(float64)
			switch bt {
			case "redacted_thinking":
				if _, ok := block["data"].(string); !ok {
					t.Fatalf("redacted_thinking must carry the spec-required data field: %v", block)
				}
				sawStart[bt] = idx
				openStops[idx] = true
			case "server_tool_use":
				if block["name"] != "web_search" || block["id"] == nil {
					t.Fatalf("server_tool_use must carry spec-plausible id/name: %v", block)
				}
				sawStart[bt] = idx
				openStops[idx] = true
			case "x_mock_novel_block":
				sawStart[bt] = idx
				openStops[idx] = true
			}
		case "content_block_stop":
			idx, _ := ev.data["index"].(float64)
			delete(openStops, idx)
		}
	}
	for _, bt := range []string{"redacted_thinking", "server_tool_use", "x_mock_novel_block"} {
		if _, ok := sawStart[bt]; !ok {
			t.Fatalf("missing injected %s block in %v", bt, eventNames(events))
		}
	}
	if len(openStops) != 0 {
		t.Fatalf("every injected block must be closed by a matching stop, still open: %v", openStops)
	}

	// The real text block still streams and the stream completes.
	sawText := false
	for _, ev := range events {
		if ev.name == "content_block_delta" {
			if delta, _ := ev.data["delta"].(map[string]any); delta["type"] == "text_delta" {
				sawText = true
			}
		}
	}
	if !sawText || eventNames(events)[len(events)-1] != "message_stop" {
		t.Fatalf("injected blocks must not derail the real stream: %v", eventNames(events))
	}
}

// --- B5: mid-stream event: error ---

func TestStreamErrorInjectionOverloaded(t *testing.T) {
	events := anthropicStreamEventsWithFaults(t, []FaultSpec{
		{Mode: "stream_error", ErrorType: "overloaded_error", ErrorMessage: "Overloaded", AfterN: 3},
	}, nil)

	found := false
	for _, ev := range events {
		if ev.name != "error" {
			continue
		}
		found = true
		if ev.data["type"] != "error" {
			t.Fatalf("error event payload type must be error, got %v", ev.data)
		}
		errObj, _ := ev.data["error"].(map[string]any)
		if errObj["type"] != "overloaded_error" || errObj["message"] != "Overloaded" {
			t.Fatalf("error payload wrong: %v", ev.data)
		}
	}
	if !found {
		t.Fatalf("expected an injected error event, got %v", eventNames(events))
	}
	// By default the stream continues after the error event.
	if names := eventNames(events); names[len(names)-1] != "message_stop" {
		t.Fatalf("stream should continue past the injected error, got %v", names)
	}
}

func TestStreamErrorComposesWithDisconnect(t *testing.T) {
	// Two-knob composition: a disconnect keyed on the injected error event
	// reproduces the real API's behavior (error, then the stream dies).
	cfg := defaultConfig()
	cfg.Anthropic.ValidateResponses = boolPtr(false)
	cfg.Anthropic.Faults = []FaultSpec{
		{Mode: "stream_error", ErrorType: "overloaded_error", AfterEvent: "content_block_start"},
		{Mode: "disconnect", AfterEvent: "error"},
	}
	srv := testServer(cfg)
	defer srv.Close()

	raw := readStreamUntilClosed(t, srv.URL, anthropicBody(true))
	if !strings.Contains(raw, "event: error") {
		t.Fatalf("stream should carry the injected error event: %s", raw)
	}
	if strings.Contains(raw, "event: message_delta") || strings.Contains(raw, "event: message_stop") {
		t.Fatalf("disconnect after error should cut the stream: %s", raw)
	}
}

func TestDecoderFaultsAreCaughtByValidationWhenOn(t *testing.T) {
	// The decoder faults emit well-formed but off-union payloads; with
	// validate_responses ON the self-validator severs the stream — the
	// contract that forces deliberate-fault scenarios to opt out with
	// validate_responses:false.
	before := validationFailures.Load()

	cfg := defaultConfig()
	cfg.Anthropic.ValidateResponses = boolPtr(true)
	cfg.Anthropic.Faults = []FaultSpec{
		{Mode: "unknown_event", EventType: "message_future", AfterEvent: "message_start"},
	}
	srv := testServer(cfg)
	defer srv.Close()

	raw := readStreamUntilClosed(t, srv.URL, anthropicBody(true))
	if strings.Contains(raw, "event: message_future") {
		t.Fatalf("validator should sever before writing the off-union frame: %s", raw)
	}
	if strings.Contains(raw, "event: message_stop") {
		t.Fatalf("stream should be severed by the validator: %s", raw)
	}
	if got := validationFailures.Load() - before; got != 1 {
		t.Fatalf("expected exactly 1 recorded validation failure, got %d", got)
	}
}
