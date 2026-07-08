package main

import (
	"fmt"
	"net/http"
)

type sseWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
	// bytes counts everything written to the stream body so far; it backs
	// the after_bytes fault-trigger knob (fault.go).
	bytes int
}

func newSSEWriter(w http.ResponseWriter) *sseWriter {
	flusher, _ := w.(http.Flusher)
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	return &sseWriter{w: w, flusher: flusher}
}

// writeData writes an SSE data-only line (OpenAI format).
func (s *sseWriter) writeData(data string) {
	n, _ := fmt.Fprintf(s.w, "data: %s\n\n", data)
	s.bytes += n
	if s.flusher != nil {
		s.flusher.Flush()
	}
}

// writeEvent writes an SSE event+data pair (Anthropic format).
func (s *sseWriter) writeEvent(event, data string) {
	n, _ := fmt.Fprintf(s.w, "event: %s\ndata: %s\n\n", event, data)
	s.bytes += n
	if s.flusher != nil {
		s.flusher.Flush()
	}
}

// writeDone writes the OpenAI [DONE] sentinel.
func (s *sseWriter) writeDone() {
	n, _ := fmt.Fprint(s.w, "data: [DONE]\n\n")
	s.bytes += n
	if s.flusher != nil {
		s.flusher.Flush()
	}
}

// writePing writes an SSE comment line for keep-alive.
func (s *sseWriter) writePing() {
	n, _ := fmt.Fprint(s.w, ": ping\n\n")
	s.bytes += n
	if s.flusher != nil {
		s.flusher.Flush()
	}
}
