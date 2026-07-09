package main

import (
	"context"
	"io"
	"net/http"
	"strings"
)

type sseWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
	// bytes counts everything written to the stream body so far; it backs
	// the after_bytes fault-trigger knob (fault.go).
	bytes int

	// Transport-fault knobs (A2/A3), armed by applyTransportFaults. All
	// zero-valued by default, in which case every frame is one write + one
	// flush like before.
	ctx             context.Context
	eol             string // "\n" normally, "\r\n" under crlf_frames (A3)
	fragmentOffset  int
	fragmentSplit   string
	fragmentDelayMs int
	coalesce        int
	pending         []byte
	pendingN        int
}

func newSSEWriter(w http.ResponseWriter) *sseWriter {
	flusher, _ := w.(http.Flusher)
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	return &sseWriter{w: w, flusher: flusher, eol: "\n"}
}

// applyTransportFaults arms the deterministic SSE transport faults:
// crlf_frames (A3) switches every line ending to \r\n; fragment_offset /
// fragment_split flush each frame in two writes separated by
// fragment_delay_ms (A2 — the mid-frame TCP boundary that reproduces
// re-framing corruption in naive proxies); coalesce_frames buffers N frames
// into a single write+flush (multi-frame TCP chunks). Coalescing and
// fragmentation are mutually exclusive; coalescing wins.
func (s *sseWriter) applyTransportFaults(ctx context.Context, cfg *ProviderConfig) {
	s.ctx = ctx
	if cfg.CrlfFrames {
		s.eol = "\r\n"
	}
	s.fragmentOffset = cfg.FragmentOffset
	s.fragmentSplit = cfg.FragmentSplit
	s.fragmentDelayMs = cfg.FragmentDelayMs
	if (s.fragmentOffset > 0 || s.fragmentSplit != "") && s.fragmentDelayMs == 0 {
		// A fragmented frame must actually land in two reads on the peer:
		// back-to-back flushes coalesce in the kernel buffer, so default a
		// small inter-write pause that keeps the boundary observable.
		s.fragmentDelayMs = 5
	}
	s.coalesce = cfg.CoalesceFrames
}

// bodyBytes implements streamSink: total body bytes written so far, backing
// the after_bytes fault-trigger knob.
func (s *sseWriter) bodyBytes() int {
	return s.bytes
}

// writeCorrupt implements streamSink: a corrupt non-JSON SSE data frame
// (the malformed_chunk fault), written raw — never routed through the
// validating frameWriter, being invalid is the point.
func (s *sseWriter) writeCorrupt() {
	s.writeData("{INVALID JSON CORRUPT")
}

// writeData writes an SSE data-only line (OpenAI format).
func (s *sseWriter) writeData(data string) {
	s.writeFrame("data: " + data + s.eol + s.eol)
}

// writeEvent writes an SSE event+data pair (Anthropic format).
func (s *sseWriter) writeEvent(event, data string) {
	s.writeFrame("event: " + event + s.eol + "data: " + data + s.eol + s.eol)
}

// writeDone writes the OpenAI [DONE] sentinel and flushes any coalesced tail.
func (s *sseWriter) writeDone() {
	s.writeFrame("data: [DONE]" + s.eol + s.eol)
	s.flushPending()
}

// writePing writes an SSE comment line for keep-alive.
func (s *sseWriter) writePing() {
	s.writeFrame(": ping" + s.eol + s.eol)
}

// writeFrame routes one complete SSE frame through the armed transport
// faults: coalescing buffers it, fragmentation splits it into two flushed
// writes, otherwise it goes out whole (the historical behavior).
func (s *sseWriter) writeFrame(frame string) {
	s.bytes += len(frame)

	if s.coalesce > 1 {
		s.pending = append(s.pending, frame...)
		s.pendingN++
		if s.pendingN >= s.coalesce {
			s.flushPending()
		}
		return
	}

	if p := s.splitPoint(frame); p > 0 {
		io.WriteString(s.w, frame[:p])
		s.flush()
		if s.ctx != nil {
			if !waitCancelable(s.ctx, s.fragmentDelayMs) {
				return
			}
		}
		io.WriteString(s.w, frame[p:])
		s.flush()
		return
	}

	io.WriteString(s.w, frame)
	s.flush()
}

// splitPoint resolves where a frame is cut in two (0 = no cut):
//   - fragment_split "rune":  one byte into the first multibyte UTF-8
//     sequence — the peer's first read ends with a dangling lead byte
//   - fragment_split "event": right after the first line — between the
//     `event:` and `data:` lines of an Anthropic frame
//   - otherwise fragment_offset bytes in, when it lands inside the frame
func (s *sseWriter) splitPoint(frame string) int {
	switch s.fragmentSplit {
	case "rune":
		for i := 0; i < len(frame); i++ {
			if frame[i]&0xC0 == 0x80 { // first UTF-8 continuation byte
				return i
			}
		}
		// No multibyte rune in this frame; fall back to the offset knob.
	case "event":
		if i := strings.Index(frame, s.eol); i >= 0 && i+len(s.eol) < len(frame) {
			return i + len(s.eol)
		}
		return 0
	}
	if s.fragmentOffset > 0 && s.fragmentOffset < len(frame) {
		return s.fragmentOffset
	}
	return 0
}

// flushPending writes out coalesced frames as one chunk.
func (s *sseWriter) flushPending() {
	if len(s.pending) == 0 {
		return
	}
	s.w.Write(s.pending)
	s.pending = s.pending[:0]
	s.pendingN = 0
	s.flush()
}

func (s *sseWriter) flush() {
	if s.flusher != nil {
		s.flusher.Flush()
	}
}
