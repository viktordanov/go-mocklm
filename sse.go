package main

import (
	"fmt"
	"net/http"
)

type sseWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
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
	fmt.Fprintf(s.w, "data: %s\n\n", data)
	if s.flusher != nil {
		s.flusher.Flush()
	}
}

// writeEvent writes an SSE event+data pair (Anthropic format).
func (s *sseWriter) writeEvent(event, data string) {
	fmt.Fprintf(s.w, "event: %s\ndata: %s\n\n", event, data)
	if s.flusher != nil {
		s.flusher.Flush()
	}
}

// writeDone writes the OpenAI [DONE] sentinel.
func (s *sseWriter) writeDone() {
	fmt.Fprint(s.w, "data: [DONE]\n\n")
	if s.flusher != nil {
		s.flusher.Flush()
	}
}

// writePing writes an SSE comment line for keep-alive.
func (s *sseWriter) writePing() {
	fmt.Fprint(s.w, ": ping\n\n")
	if s.flusher != nil {
		s.flusher.Flush()
	}
}
