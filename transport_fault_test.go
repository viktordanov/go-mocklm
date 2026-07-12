package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

// --- Usage faults (D1-D3), deterministic transport faults
// (A2 fragmentation / A3 CRLF+coalesce / A7 stall), and the HTTP fault
// trio (C2 Retry-After date / C5 via error mode / C9 non-JSON 200) ---

// --- D. Usage faults ---

func TestUsageFaultOmitNonStreamAndStream(t *testing.T) {
	cfg := defaultConfig()
	cfg.OpenAI.UsageFault = "omit"
	// Omission is spec-valid (usage is optional in the pinned response
	// root), so validation stays ON to pin exactly that.
	cfg.OpenAI.ValidateResponses = boolPtr(true)
	srv := testServer(cfg)
	defer srv.Close()

	// Non-stream: no usage key at all.
	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	var body map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if _, ok := body["usage"]; ok {
		t.Fatalf("usage_fault=omit must drop the usage key, got %v", body["usage"])
	}

	// Stream: no usage anywhere, even though include_usage was requested.
	lines := collectDataLines(t, srv.URL, openaiStreamBodyWithUsage(), nil)
	for _, l := range lines {
		if strings.Contains(l, `"usage"`) {
			t.Fatalf("usage_fault=omit must suppress stream usage, got line: %s", l)
		}
	}
	if lines[len(lines)-1] != "[DONE]" {
		t.Fatalf("stream must still terminate with [DONE], got %q", lines[len(lines)-1])
	}
}

func TestUsageFaultPartialEmitsPromptTokensOnly(t *testing.T) {
	cfg := defaultConfig()
	cfg.OpenAI.UsageFault = "partial"
	// Partial usage violates CompletionUsage.required by design.
	cfg.OpenAI.ValidateResponses = boolPtr(false)
	srv := testServer(cfg)
	defer srv.Close()

	// Non-stream: usage == {"prompt_tokens": N} and nothing else.
	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()
	var body map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	usage, ok := body["usage"].(map[string]any)
	if !ok {
		t.Fatalf("expected a usage object, got %v", body["usage"])
	}
	if len(usage) != 1 {
		t.Fatalf("partial usage must carry prompt_tokens only, got %v", usage)
	}
	if _, ok := usage["prompt_tokens"]; !ok {
		t.Fatalf("partial usage must carry prompt_tokens, got %v", usage)
	}

	// Stream: the include_usage trailer carries the same partial object.
	lines := collectDataLines(t, srv.URL, openaiStreamBodyWithUsage(), nil)
	var trailerUsage map[string]any
	for _, l := range lines {
		if l == "[DONE]" {
			continue
		}
		var chunk map[string]any
		if err := json.Unmarshal([]byte(l), &chunk); err != nil {
			t.Fatalf("invalid chunk JSON: %v", err)
		}
		if choices, ok := chunk["choices"].([]any); ok && len(choices) == 0 {
			trailerUsage, _ = chunk["usage"].(map[string]any)
		}
	}
	if trailerUsage == nil {
		t.Fatalf("expected a trailing usage chunk, got lines: %v", lines)
	}
	if len(trailerUsage) != 1 {
		t.Fatalf("trailer usage must be partial (prompt_tokens only), got %v", trailerUsage)
	}
}

func TestUsageFaultPartialIsCaughtByValidation(t *testing.T) {
	// The tri-state stays load-bearing: partial usage with validation ON
	// fails loudly instead of shipping an off-spec body.
	cfg := defaultConfig()
	cfg.OpenAI.UsageFault = "partial"
	cfg.OpenAI.ValidateResponses = boolPtr(true)
	srv := testServer(cfg)
	defer srv.Close()

	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 500 {
		t.Fatalf("partial usage must trip the self-validator (500), got %d", resp.StatusCode)
	}
}

func TestUsageFaultTrailerForcedWithoutStreamOptions(t *testing.T) {
	// D3 as a fault variant: the real include_usage wire shape even though
	// the request never set stream_options. Spec-valid → validation ON.
	cfg := defaultConfig()
	cfg.OpenAI.UsageFault = "trailer"
	cfg.OpenAI.ValidateResponses = boolPtr(true)
	srv := testServer(cfg)
	defer srv.Close()

	lines := collectDataLines(t, srv.URL, openaiChatBody(true), nil)
	if lines[len(lines)-1] != "[DONE]" {
		t.Fatalf("expected [DONE] last, got %q", lines[len(lines)-1])
	}

	sawNullUsageContentChunk := false
	var trailerUsage map[string]any
	for _, l := range lines[:len(lines)-1] {
		var chunk map[string]any
		if err := json.Unmarshal([]byte(l), &chunk); err != nil {
			t.Fatalf("invalid chunk JSON: %v", err)
		}
		usage, hasUsage := chunk["usage"]
		if !hasUsage {
			t.Fatalf("trailer mode: every chunk must carry a usage key, missing on: %s", l)
		}
		choices := chunk["choices"].([]any)
		if len(choices) == 0 {
			trailerUsage, _ = usage.(map[string]any)
		} else if usage == nil {
			sawNullUsageContentChunk = true
		} else {
			t.Fatalf("non-trailer chunks must carry usage: null, got: %s", l)
		}
	}
	if !sawNullUsageContentChunk {
		t.Fatal("expected usage:null on the content chunks")
	}
	if trailerUsage == nil {
		t.Fatal("expected a trailing choices:[] usage chunk")
	}
	if _, ok := trailerUsage["total_tokens"]; !ok {
		t.Fatalf("trailer usage must be the full spec shape, got %v", trailerUsage)
	}
}

func TestContentTextOverridesGeneratedContent(t *testing.T) {
	const text = "café über naïve piñata"
	cfg := defaultConfig()
	cfg.OpenAI.ContentText = text
	cfg.Anthropic.ContentText = text
	srv := testServer(cfg)
	defer srv.Close()

	// OpenAI non-stream: verbatim, no capitalize/period decoration.
	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()
	var body map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	content := body["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)["content"].(string)
	if content != text {
		t.Fatalf("content_text must be emitted verbatim, got %q", content)
	}

	// Anthropic stream: the text_delta tokens reassemble to the same text.
	events := collectAnthropicEvents(t, srv.URL, anthropicBody(true))
	var b strings.Builder
	for _, ev := range events {
		if ev.name != "content_block_delta" {
			continue
		}
		delta := ev.data["delta"].(map[string]any)
		if delta["type"] == "text_delta" {
			b.WriteString(delta["text"].(string))
		}
	}
	if b.String() != text {
		t.Fatalf("streamed content_text mismatch: got %q, want %q", b.String(), text)
	}
}

// --- A. Deterministic SSE transport faults (sseWriter-level) ---

// flushRecorder records the byte span of every Write between Flushes, so
// tests can assert exactly where the frame was cut.
type flushRecorder struct {
	header http.Header
	// segments[i] is everything written before the i-th Flush.
	segments []string
	current  strings.Builder
}

func newFlushRecorder() *flushRecorder {
	return &flushRecorder{header: make(http.Header)}
}

func (f *flushRecorder) Header() http.Header { return f.header }
func (f *flushRecorder) WriteHeader(int)     {}
func (f *flushRecorder) Write(b []byte) (int, error) {
	f.current.Write(b)
	return len(b), nil
}
func (f *flushRecorder) Flush() {
	f.segments = append(f.segments, f.current.String())
	f.current.Reset()
}

func faultedSSEWriter(cfg ProviderConfig) (*sseWriter, *flushRecorder) {
	rec := newFlushRecorder()
	sse := newSSEWriter(rec)
	sse.applyTransportFaults(context.Background(), &cfg)
	return sse, rec
}

func TestFragmentOffsetSplitsEveryFrameInTwoFlushes(t *testing.T) {
	sse, rec := faultedSSEWriter(ProviderConfig{FragmentOffset: 10, FragmentDelayMs: 1})
	payload := `{"content":"` + strings.Repeat("x", 30) + `"}`
	sse.writeData(payload)

	want := "data: " + payload + "\n\n"
	if len(rec.segments) != 2 {
		t.Fatalf("expected 2 flushed segments, got %d: %q", len(rec.segments), rec.segments)
	}
	if len(rec.segments[0]) != 10 {
		t.Fatalf("first segment must be exactly fragment_offset bytes, got %d", len(rec.segments[0]))
	}
	if rec.segments[0]+rec.segments[1] != want {
		t.Fatalf("fragments must reassemble to the intact frame:\n got %q\nwant %q", rec.segments[0]+rec.segments[1], want)
	}
}

func TestFragmentRuneSplitsInsideFirstMultibyteRune(t *testing.T) {
	sse, rec := faultedSSEWriter(ProviderConfig{FragmentSplit: "rune", FragmentDelayMs: 1})
	sse.writeData(`{"text":"café latte"}`)

	if len(rec.segments) != 2 {
		t.Fatalf("expected 2 flushed segments, got %d", len(rec.segments))
	}
	first := rec.segments[0]
	last := first[len(first)-1]
	if last&0xC0 != 0xC0 {
		t.Fatalf("first segment must end with a dangling UTF-8 lead byte, ends with 0x%02X", last)
	}
	if json.Valid([]byte(strings.TrimPrefix(strings.TrimSpace(first), "data: "))) {
		t.Fatal("first segment should be an incomplete frame, not a valid JSON payload")
	}
}

func TestFragmentRuneFallsBackToOffsetOnASCII(t *testing.T) {
	sse, rec := faultedSSEWriter(ProviderConfig{FragmentSplit: "rune", FragmentOffset: 7, FragmentDelayMs: 1})
	sse.writeData(`{"text":"plain ascii"}`)

	if len(rec.segments) != 2 || len(rec.segments[0]) != 7 {
		t.Fatalf("ASCII frame must fall back to fragment_offset, got segments %q", rec.segments)
	}
}

func TestFragmentEventSplitsBetweenEventAndDataLines(t *testing.T) {
	sse, rec := faultedSSEWriter(ProviderConfig{FragmentSplit: "event", FragmentDelayMs: 1})
	sse.writeEvent("message_start", `{"type":"message_start"}`)

	if len(rec.segments) != 2 {
		t.Fatalf("expected 2 flushed segments, got %d", len(rec.segments))
	}
	if rec.segments[0] != "event: message_start\n" {
		t.Fatalf("first segment must be exactly the event line, got %q", rec.segments[0])
	}
	if !strings.HasPrefix(rec.segments[1], "data: ") {
		t.Fatalf("second segment must start with the data line, got %q", rec.segments[1])
	}
}

func TestCrlfFramesUseCRLFEverywhere(t *testing.T) {
	sse, rec := faultedSSEWriter(ProviderConfig{CrlfFrames: true})
	sse.writeEvent("ping", `{"type":"ping"}`)
	sse.writeDone()

	if rec.segments[0] != "event: ping\r\ndata: {\"type\":\"ping\"}\r\n\r\n" {
		t.Fatalf("CRLF event frame malformed: %q", rec.segments[0])
	}
	if rec.segments[1] != "data: [DONE]\r\n\r\n" {
		t.Fatalf("CRLF done frame malformed: %q", rec.segments[1])
	}
}

func TestCoalesceFramesBuffersIntoSingleWrite(t *testing.T) {
	sse, rec := faultedSSEWriter(ProviderConfig{CoalesceFrames: 3})
	sse.writeData("one")
	sse.writeData("two")
	if len(rec.segments) != 0 {
		t.Fatalf("frames below the coalesce threshold must not flush, got %q", rec.segments)
	}
	sse.writeData("three")
	if len(rec.segments) != 1 {
		t.Fatalf("expected one coalesced flush, got %d", len(rec.segments))
	}
	if rec.segments[0] != "data: one\n\ndata: two\n\ndata: three\n\n" {
		t.Fatalf("coalesced chunk malformed: %q", rec.segments[0])
	}

	// A trailing partial batch is flushed with the [DONE] sentinel.
	sse.writeData("four")
	sse.writeDone()
	if len(rec.segments) != 2 || rec.segments[1] != "data: four\n\ndata: [DONE]\n\n" {
		t.Fatalf("stream end must flush the coalesced tail, got %q", rec.segments)
	}
}

// --- A7. Stall ---

func TestStallHoldsConnectionUntilClientGivesUp(t *testing.T) {
	cfg := defaultConfig()
	cfg.OpenAI.Faults = []FaultSpec{{Mode: "stall", AfterN: 2}}
	srv := testServer(cfg)
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 400*time.Millisecond)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "POST", srv.URL+"/v1/chat/completions", strings.NewReader(openaiChatBody(true)))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	start := time.Now()
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("stream should start before the stall, got %v", err)
	}
	defer resp.Body.Close()

	raw, readErr := io.ReadAll(resp.Body)
	elapsed := time.Since(start)
	if readErr == nil {
		t.Fatalf("a stalled stream must never complete; read %d bytes to EOF", len(raw))
	}
	if elapsed < 350*time.Millisecond {
		t.Fatalf("the read should have blocked until the client deadline, returned after %v", elapsed)
	}
	if elapsed > 5*time.Second {
		t.Fatalf("the handler must free promptly on client cancel, took %v", elapsed)
	}
	// The frames written before the stall did arrive.
	if !strings.Contains(string(raw), "data: ") {
		t.Fatalf("expected pre-stall frames, got %q", string(raw))
	}
	if strings.Contains(string(raw), "[DONE]") {
		t.Fatal("a stalled stream must not reach [DONE]")
	}
}

// --- C2 / C9 ---

func TestErrorFaultSetsRetryAfterHeaderVerbatim(t *testing.T) {
	const date = "Wed, 01 Jan 2031 00:00:00 GMT"
	cfg := defaultConfig()
	cfg.OpenAI.Faults = []FaultSpec{{Mode: "error", ErrorStatus: 429, RetryAfter: date}}
	srv := testServer(cfg)
	defer srv.Close()

	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 429 {
		t.Fatalf("expected 429, got %d", resp.StatusCode)
	}
	if got := resp.Header.Get("Retry-After"); got != date {
		t.Fatalf("Retry-After must be set verbatim: got %q, want %q", got, date)
	}
}

func TestNonJSONBodyFaultReturns200HTML(t *testing.T) {
	cfg := defaultConfig()
	cfg.OpenAI.Faults = []FaultSpec{{Mode: "non_json_body"}}
	srv := testServer(cfg)
	defer srv.Close()

	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("C9 is a 200 with a non-JSON body, got status %d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/html") {
		t.Fatalf("expected text/html, got %q", ct)
	}
	raw, _ := io.ReadAll(resp.Body)
	if json.Valid(raw) {
		t.Fatalf("body must not be JSON, got %s", raw)
	}
	if !strings.Contains(string(raw), "<html>") {
		t.Fatalf("expected an HTML error page, got %s", raw)
	}
}

func TestNonJSONBodyFaultRejectsStreamWhens(t *testing.T) {
	cfg := defaultConfig()
	cfg.OpenAI.Faults = []FaultSpec{{Mode: "non_json_body", AfterN: 1}}
	srv := testServer(cfg)
	defer srv.Close()

	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 400 {
		t.Fatalf("non_json_body with a stream WHEN must be rejected with 400, got %d", resp.StatusCode)
	}
}
