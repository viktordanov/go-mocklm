package main

import (
	"bufio"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

// --- /v1/completions streaming in the shared fault pipeline ---
//
// The legacy completions stream runs through the same three fault layers as
// the other stream surfaces: SSE transport faults, the resolved streamFaults
// injector, and the legacy checkStreamingFault knobs. The stream is exactly
// two real data frames (content, finish) + [DONE]; after_n fires AFTER the
// Nth real frame.

func completionsBody(stream bool) string {
	return `{"model":"gpt-3.5-turbo-instruct","stream":` + boolStr(stream) + `,"prompt":"Hello"}`
}

// collectCompletionsData posts a streaming completions request against a
// fresh server built from cfg and returns the SSE data-line payloads plus
// any read error (mid-stream disconnects surface as read errors or
// truncation, so callers assert on both).
func collectCompletionsData(t *testing.T, cfg *Config) ([]string, error) {
	t.Helper()
	srv := testServer(cfg)
	defer srv.Close()
	resp, err := postJSON(srv.URL+"/v1/completions", completionsBody(true), nil)
	if err != nil {
		t.Fatalf("POST /v1/completions (stream) failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("expected status 200, got %d", resp.StatusCode)
	}

	scanner := bufio.NewScanner(resp.Body)
	var data []string
	for scanner.Scan() {
		if line := scanner.Text(); strings.HasPrefix(line, "data: ") {
			data = append(data, strings.TrimPrefix(line, "data: "))
		}
	}
	return data, scanner.Err()
}

func TestCompletionsStreamDefaultShapeUnchanged(t *testing.T) {
	// No faults armed: content frame, finish frame, [DONE] — byte-order
	// identical to the pre-pipeline behavior.
	data, err := collectCompletionsData(t, defaultConfig())
	if err != nil {
		t.Fatalf("stream read failed: %v", err)
	}
	if len(data) != 3 {
		t.Fatalf("expected 3 data lines (content, finish, [DONE]), got %d: %v", len(data), data)
	}
	if !strings.Contains(data[0], `"finish_reason":null`) {
		t.Errorf("content frame should carry finish_reason:null, got %s", data[0])
	}
	if !strings.Contains(data[1], `"finish_reason":"stop"`) || !strings.Contains(data[1], `"usage"`) {
		t.Errorf("finish frame should carry finish_reason stop + usage, got %s", data[1])
	}
	if data[2] != "[DONE]" {
		t.Errorf("expected [DONE] sentinel, got %q", data[2])
	}
}

func TestCompletionsStreamDisconnectAfterN(t *testing.T) {
	// after_n counts REAL frames and fires AFTER the Nth: after_n=1 lets
	// the content frame out and cuts before the finish frame; after_n=2
	// lets the finish frame out and cuts before [DONE].
	for _, tc := range []struct {
		afterN    int
		wantData  int
		lastFrame string
	}{
		{1, 1, `"finish_reason":null`},
		{2, 2, `"finish_reason":"stop"`},
	} {
		cfg := defaultConfig()
		cfg.OpenAI.Faults = []FaultSpec{{Mode: "disconnect", AfterN: tc.afterN}}
		data, err := collectCompletionsData(t, cfg)
		if len(data) != tc.wantData {
			t.Fatalf("after_n=%d: expected %d data lines, got %d: %v (err=%v)", tc.afterN, tc.wantData, len(data), data, err)
		}
		if !strings.Contains(data[len(data)-1], tc.lastFrame) {
			t.Errorf("after_n=%d: last frame should contain %s, got %s", tc.afterN, tc.lastFrame, data[len(data)-1])
		}
		for _, d := range data {
			if d == "[DONE]" {
				t.Errorf("after_n=%d: [DONE] must not survive the disconnect", tc.afterN)
			}
		}
	}
}

func TestCompletionsStreamMalformedChunkInjected(t *testing.T) {
	// Injector malformed_chunk: a corrupt raw frame after the first real
	// frame; the stream continues through [DONE].
	cfg := defaultConfig()
	cfg.OpenAI.Faults = []FaultSpec{{Mode: "malformed_chunk", AfterN: 1}}
	data, err := collectCompletionsData(t, cfg)
	if err != nil {
		t.Fatalf("stream read failed: %v", err)
	}
	if len(data) != 4 {
		t.Fatalf("expected 4 data lines (content, corrupt, finish, [DONE]), got %d: %v", len(data), data)
	}
	if data[1] != "{INVALID JSON CORRUPT" {
		t.Errorf("expected corrupt frame after the first real frame, got %q", data[1])
	}
	if data[3] != "[DONE]" {
		t.Errorf("stream should continue to [DONE] after the corrupt frame, got %q", data[3])
	}
}

func TestCompletionsStreamLegacyKnobs(t *testing.T) {
	// The legacy checkStreamingFault knobs work on completions exactly as
	// on chat: disconnect_after_chunks cuts before the (index >= N)th
	// frame; malformed_chunk corrupts at the frame-count midpoint.
	cfg := defaultConfig()
	cfg.OpenAI.DisconnectAfterChunks = 1
	data, _ := collectCompletionsData(t, cfg)
	if len(data) != 1 {
		t.Fatalf("disconnect_after_chunks=1: expected 1 data line, got %d: %v", len(data), data)
	}

	cfg = defaultConfig()
	cfg.OpenAI.MalformedChunk = true
	data, err := collectCompletionsData(t, cfg)
	if err != nil {
		t.Fatalf("stream read failed: %v", err)
	}
	if len(data) != 4 || data[1] != "{INVALID JSON CORRUPT" {
		t.Fatalf("malformed_chunk: expected corrupt frame at midpoint (4 lines), got %v", data)
	}
	if data[3] != "[DONE]" {
		t.Errorf("stream should continue after the legacy malformed chunk, got %q", data[3])
	}
}

func TestCompletionsStreamTransportFaultsApply(t *testing.T) {
	// crlf_frames now reaches the completions stream: every SSE line ending
	// goes out as \r\n.
	cfg := defaultConfig()
	cfg.OpenAI.CrlfFrames = true
	srv := testServer(cfg)
	defer srv.Close()

	resp, err := postJSON(srv.URL+"/v1/completions", completionsBody(true), nil)
	if err != nil {
		t.Fatalf("POST /v1/completions (stream) failed: %v", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read failed: %v", err)
	}
	if !strings.Contains(string(raw), "data: [DONE]\r\n\r\n") {
		t.Fatalf("expected CRLF-framed [DONE], got %q", string(raw))
	}
	if strings.Contains(strings.ReplaceAll(string(raw), "\r\n", ""), "\n") {
		t.Errorf("found bare \\n line endings despite crlf_frames")
	}
}

func TestCompletionsStreamStallFreesHandlerOnCancel(t *testing.T) {
	// stall goes silent after the first frame; a client that gives up
	// unblocks the handler (context cancellation) instead of leaking it.
	cfg := defaultConfig()
	cfg.OpenAI.Faults = []FaultSpec{{Mode: "stall"}}
	srv := testServer(cfg)
	defer srv.Close()

	client := &http.Client{Timeout: 400 * time.Millisecond}
	req, _ := http.NewRequest("POST", srv.URL+"/v1/completions", strings.NewReader(completionsBody(true)))
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("request failed before streaming: %v", err)
	}
	defer resp.Body.Close()

	scanner := bufio.NewScanner(resp.Body)
	var data []string
	for scanner.Scan() {
		if line := scanner.Text(); strings.HasPrefix(line, "data: ") {
			data = append(data, strings.TrimPrefix(line, "data: "))
		}
	}
	if scanner.Err() == nil {
		t.Fatalf("expected a client-timeout read error from the stalled stream, got clean EOF with %v", data)
	}
	if len(data) != 1 {
		t.Fatalf("stall should fire after the first real frame, got %d data lines: %v", len(data), data)
	}
	// srv.Close blocks until the handler returns — a leaked handler would
	// hang the test here.
}
