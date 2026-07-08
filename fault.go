package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"strconv"
	"sync"
	"time"
)

// FaultSpec is the generalized two-knob fault model (baml-rest inspired):
// WHEN the fault fires × HOW it fails. All WHEN knobs are optional and AND
// together; a spec with no WHEN fires at the first opportunity (pre-body
// for pre-body modes, after the first written frame for stream modes).
// Each spec fires at most once per request.
type FaultSpec struct {
	// WHEN --------------------------------------------------------------
	// AfterMs arms the fault only once the response is at least this many
	// milliseconds old (pre-body modes wait it out cancel-aware; stream
	// modes check it after each written frame).
	AfterMs int `toml:"after_ms" json:"after_ms,omitempty"`
	// AfterBytes arms a stream fault once at least this many body bytes
	// have been written.
	AfterBytes int `toml:"after_bytes" json:"after_bytes,omitempty"`
	// AfterEvent fires a stream fault immediately after the named SSE
	// event type is written (Anthropic streams; OpenAI data-only frames
	// have no event name and never match).
	AfterEvent string `toml:"after_event" json:"after_event,omitempty"`
	// AfterN arms a stream fault once at least N SSE frames (Anthropic
	// events / OpenAI data chunks) have been written.
	AfterN int `toml:"after_n" json:"after_n,omitempty"`

	// HOW ----------------------------------------------------------------
	// Mode selects the failure:
	//   "error"           pre-body HTTP error (error_status/error_type/
	//                     error_message); cannot fire mid-stream — once the
	//                     SSE 200 is sent the status is locked, use
	//                     stream_error instead
	//   "disconnect"      TCP RST; pre-body when no stream WHEN is set,
	//                     otherwise mid-stream
	//   "malformed_chunk" a corrupt non-JSON SSE frame, stream continues
	//   "unknown_event"   B1: a well-formed but off-vocabulary Anthropic
	//                     top-level event (event_type, default
	//                     "message_future"), repeated `repeat` times
	//   "unknown_block"   B2: a complete content block (start+stop) whose
	//                     content_block.type is block_type — spec-accurate
	//                     shapes for real suppressed types
	//                     (redacted_thinking, server_tool_use), a generic
	//                     probe shape otherwise
	//   "stream_error"    B5: a mid-stream `event: error` carrying
	//                     error_type/error_message; stream continues
	//                     (compose {"mode":"disconnect","after_event":
	//                     "error"} to cut after it)
	//   "stall"           A7: stop writing mid-stream and hold the
	//                     connection open — no bytes, no close — until the
	//                     client disconnects. Stream-phase only (fires
	//                     after the first frame when no WHEN is set).
	//   "non_json_body"   C9: a 200 with a text/html body instead of JSON —
	//                     the classic proxy error page. Pre-body only.
	//
	// The decoder-fault modes (unknown_event / unknown_block /
	// stream_error) emit WELL-FORMED but off-union payloads: the pinned
	// MessageStreamEvent union has no arm for them, so scenarios driving
	// them must set validate_responses:false or the self-validator severs
	// the stream.
	Mode string `toml:"mode" json:"mode"`

	// Mode parameters.
	ErrorStatus  int    `toml:"error_status" json:"error_status,omitempty"`
	ErrorType    string `toml:"error_type" json:"error_type,omitempty"`
	ErrorMessage string `toml:"error_message" json:"error_message,omitempty"`
	// RetryAfter sets the Retry-After header verbatim on an "error" fault —
	// numeric seconds or an HTTP-date (C2: the date form must be ignored by
	// spec-narrow clients and fall back to their own backoff).
	RetryAfter string `toml:"retry_after" json:"retry_after,omitempty"`
	EventType    string `toml:"event_type" json:"event_type,omitempty"`
	BlockType    string `toml:"block_type" json:"block_type,omitempty"`
	// Repeat emits the injected frame/block this many times (default 1) —
	// the same type twice in one stream is the decoder's warn-once probe.
	Repeat int `toml:"repeat" json:"repeat,omitempty"`
}

// streamPhase reports whether the spec fires during the SSE stream (vs
// pre-body in checkFaults).
func (f *FaultSpec) streamPhase() bool {
	switch f.Mode {
	case "malformed_chunk", "unknown_event", "unknown_block", "stream_error", "stall":
		return true
	case "disconnect":
		return f.AfterEvent != "" || f.AfterN > 0 || f.AfterBytes > 0
	}
	return false
}

// validateFaultSpec rejects specs that cannot fire as written.
func validateFaultSpec(f *FaultSpec) error {
	switch f.Mode {
	case "error", "non_json_body":
		if f.AfterEvent != "" || f.AfterN > 0 || f.AfterBytes > 0 {
			return fmt.Errorf("fault mode %q cannot fire mid-stream (the status is locked once the body starts); use \"stream_error\" or \"disconnect\"", f.Mode)
		}
	case "disconnect", "malformed_chunk", "unknown_event", "unknown_block", "stream_error", "stall":
	default:
		return fmt.Errorf("unknown fault mode %q", f.Mode)
	}
	return nil
}

// waitCancelable sleeps for ms, returning false early when the request is
// canceled — fault-suite hygiene: a client timeout frees the handler
// instead of leaking it into the sleep.
func waitCancelable(ctx context.Context, ms int) bool {
	if ms <= 0 {
		return true
	}
	t := time.NewTimer(time.Duration(ms) * time.Millisecond)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-ctx.Done():
		return false
	}
}

// RateLimiter implements a sliding-window rate limiter.
type RateLimiter struct {
	mu       sync.Mutex
	requests []time.Time
	rpm      int
}

func newRateLimiter(rpm int) *RateLimiter {
	return &RateLimiter{rpm: rpm}
}

// Allow checks if a request is allowed. Returns (allowed, retryAfterSeconds).
func (rl *RateLimiter) Allow() (bool, int) {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	window := now.Add(-60 * time.Second)

	// Remove expired entries
	valid := rl.requests[:0]
	for _, t := range rl.requests {
		if t.After(window) {
			valid = append(valid, t)
		}
	}
	rl.requests = valid

	if len(rl.requests) >= rl.rpm {
		// Calculate retry-after from oldest request in window
		oldest := rl.requests[0]
		retryAfter := int(oldest.Add(60*time.Second).Sub(now).Seconds()) + 1
		if retryAfter < 1 {
			retryAfter = 1
		}
		return false, retryAfter
	}

	rl.requests = append(rl.requests, now)

	return true, 0
}

// applyFaultHeader overlays per-request fault knobs from the X-MockLM-Fault
// header — a JSON ProviderConfig fragment, e.g.
// {"error_rate":1.0,"error_status":503} — onto this request's config
// snapshot, mirroring the X-MockLM-Tokens header (tokens.go). Keys absent
// from the JSON keep their configured values. Note: rate_limit_rpm and
// max_concurrent are inherently cross-request and cannot be targeted here.
func applyFaultHeader(r *http.Request, cfg *ProviderConfig) error {
	h := r.Header.Get("X-MockLM-Fault")
	if h == "" {
		return nil
	}
	if err := json.Unmarshal([]byte(h), cfg); err != nil {
		return fmt.Errorf("invalid X-MockLM-Fault header value: %v", err)
	}
	return nil
}

// faultErrorStatus resolves the status and provider-valid error type for an
// injected error, defaulting to 500 when error_status is unset. Anthropic
// types follow the pinned spec's ErrorResponse discriminator mapping (there
// is no "server_error" arm); OpenAI's Error.type is a free-form string where
// the real API uses server_error for 5xx.
func faultErrorStatus(cfg *ProviderConfig, provider string) (int, string) {
	status := cfg.ErrorStatus
	if status == 0 {
		status = 500
	}
	return status, errorTypeForStatus(status, provider)
}

// errorTypeForStatus derives the provider-valid error type for a status.
func errorTypeForStatus(status int, provider string) string {
	if provider == "anthropic" {
		return anthropicErrorType(status)
	}
	switch {
	case status == 429:
		return "rate_limit_error"
	case status >= 500:
		return "server_error"
	default:
		return "invalid_request_error"
	}
}

// anthropicErrorType maps an HTTP status to the error union member the real
// Anthropic API uses for it (spec ErrorResponse discriminator mapping).
func anthropicErrorType(status int) string {
	switch status {
	case 400:
		return "invalid_request_error"
	case 401:
		return "authentication_error"
	case 403:
		return "permission_error"
	case 404:
		return "not_found_error"
	case 429:
		return "rate_limit_error"
	case 504:
		return "timeout_error"
	case 529:
		return "overloaded_error"
	}
	if status >= 500 {
		return "api_error"
	}
	return "invalid_request_error"
}

// checkFaults runs fault injection checks. Returns true if the request was handled (caller should return).
func checkFaults(w http.ResponseWriter, r *http.Request, cfg *ProviderConfig, limiter *RateLimiter, state *ServerState, provider string) bool {
	ctx := r.Context()

	// 0. Per-request fault targeting: the header overlays cfg, so knobs it
	// sets also steer everything downstream of checkFaults (stream faults,
	// stop reasons, strict mode, ...).
	if err := applyFaultHeader(r, cfg); err != nil {
		writeErrorResponse(w, cfg, 400, provider, "invalid_request_error", err.Error())
		return true
	}

	// 0b. Attempt accounting: every request that reaches fault processing
	// counts, so retries the proxy makes are observable at
	// /admin/request-count even when they end up rate-limited or failed.
	// fail_first_n and attempt_faults index off the same counter.
	var attempt int64 = 1
	if state != nil {
		attempt = state.NextAttempt(provider)
	}

	// 0c. Resolve this request's fault specs: global faults plus this
	// attempt's entry in attempt_faults (0-based; requests past the end of
	// the array get none). Invalid specs fail loudly with a 400 so a typo'd
	// scenario cannot silently test nothing.
	var active []FaultSpec
	active = append(active, cfg.Faults...)
	if idx := int(attempt - 1); idx >= 0 && idx < len(cfg.AttemptFaults) {
		active = append(active, cfg.AttemptFaults[idx]...)
	}
	var preBody, stream []FaultSpec
	for i := range active {
		if err := validateFaultSpec(&active[i]); err != nil {
			writeErrorResponse(w, cfg, 400, provider, "invalid_request_error", err.Error())
			return true
		}
		if active[i].streamPhase() {
			stream = append(stream, active[i])
		} else {
			preBody = append(preBody, active[i])
		}
	}
	cfg.streamFaults = stream

	// 1. Rate limiting
	if cfg.RateLimitRPM > 0 && limiter != nil {
		allowed, retryAfter := limiter.Allow()
		if !allowed {
			w.Header().Set("Retry-After", strconv.Itoa(retryAfter))
			writeErrorResponse(w, cfg, 429, provider, "rate_limit_error", "Rate limit exceeded. Please retry after the specified time.")
			return true
		}
	}

	// 2a. Deterministic fail-first-N: the first N requests per provider fail
	// with error_status, then requests succeed. The counter lives on
	// ServerState (reset on config update/reset) and is shared by header-
	// and config-driven fail_first_n.
	if cfg.FailFirstN > 0 && attempt <= int64(cfg.FailFirstN) {
		status, errType := faultErrorStatus(cfg, provider)
		writeErrorResponse(w, cfg, status, provider, errType, fmt.Sprintf("Simulated deterministic error %d/%d (status %d)", attempt, cfg.FailFirstN, status))
		return true
	}

	// 2b. Random error
	if cfg.ErrorRate > 0 && rand.Float64() < cfg.ErrorRate {
		status, errType := faultErrorStatus(cfg, provider)
		writeErrorResponse(w, cfg, status, provider, errType, fmt.Sprintf("Simulated error (status %d)", status))
		return true
	}

	// 2c. Pre-body fault specs ("error" and un-WHEN'd "disconnect"): the
	// first one wins — the request is over once it fires. after_ms holds
	// cancel-aware before firing.
	if len(preBody) > 0 {
		f := &preBody[0]
		if !waitCancelable(ctx, f.AfterMs) {
			return true // client gone; nothing left to serve
		}
		switch f.Mode {
		case "error":
			status := f.ErrorStatus
			if status == 0 {
				status, _ = faultErrorStatus(cfg, provider)
			}
			errType := f.ErrorType
			if errType == "" {
				errType = errorTypeForStatus(status, provider)
			}
			msg := f.ErrorMessage
			if msg == "" {
				msg = fmt.Sprintf("Simulated fault error (attempt %d, status %d)", attempt, status)
			}
			if f.RetryAfter != "" {
				w.Header().Set("Retry-After", f.RetryAfter)
			}
			writeErrorResponse(w, cfg, status, provider, errType, msg)
		case "non_json_body":
			// C9: a 200 whose body is not JSON — what an intermediary
			// proxy's error page looks like to a JSON-expecting client.
			w.Header().Set("Content-Type", "text/html")
			w.WriteHeader(http.StatusOK)
			fmt.Fprint(w, "<html><head><title>502 Bad Gateway</title></head><body><h1>Bad Gateway</h1><p>mock upstream proxy error page</p></body></html>\n")
		case "disconnect":
			hijackAndClose(w)
		}
		return true
	}

	// 3. Timeout (hold then close)
	if cfg.TimeoutMs > 0 {
		waitCancelable(ctx, cfg.TimeoutMs)
		hijackAndClose(w)
		return true
	}

	// 4. Latency
	if cfg.LatencyMs > 0 {
		if !waitCancelable(ctx, cfg.LatencyMs) {
			return true
		}
	}

	return false
}

// checkStreamingFault checks for mid-stream faults. Returns true if the connection should be terminated.
func checkStreamingFault(w http.ResponseWriter, cfg *ProviderConfig, chunkIndex int, totalChunks int) bool {
	// Disconnect after N chunks
	if cfg.DisconnectAfterChunks > 0 && chunkIndex >= cfg.DisconnectAfterChunks {
		hijackAndClose(w)
		return true
	}

	// Malformed chunk at midpoint
	if cfg.MalformedChunk && chunkIndex == totalChunks/2 {
		sse := newSSEWriter(w)
		sse.writeData("{INVALID JSON CORRUPT")
		return false // Continue streaming after malformed chunk
	}

	return false
}

// frameWriter writes one SSE frame through the owning handler's normal
// pipeline (including validate_responses when enabled). For Anthropic
// streams `event` is the SSE event name; OpenAI data-only frames ignore it.
// Returns true when the stream must terminate.
type frameWriter func(event, data string) bool

// streamFaultInjector fires the stream-phase FaultSpecs resolved by
// checkFaults as the stream progresses. The handler calls afterFrame after
// each real frame it writes; injected frames are written through the same
// validating frameWriter — deliberately off-vocabulary payloads
// (unknown_event / unknown_block / stream_error) therefore require the
// scenario to set validate_responses:false. A nil injector is a no-op.
type streamFaultInjector struct {
	ctx    context.Context
	w      http.ResponseWriter
	sse    *sseWriter
	write  frameWriter
	specs  []FaultSpec
	fired  []bool
	start  time.Time
	frames int
	// blockIndex numbers injected content blocks well clear of the real
	// blocks' indexes so an injected block never collides with one the
	// handler is streaming.
	blockIndex int
}

func newStreamFaultInjector(ctx context.Context, specs []FaultSpec, w http.ResponseWriter, sse *sseWriter, write frameWriter) *streamFaultInjector {
	if len(specs) == 0 {
		return nil
	}
	return &streamFaultInjector{
		ctx:        ctx,
		w:          w,
		sse:        sse,
		write:      write,
		specs:      specs,
		fired:      make([]bool, len(specs)),
		start:      time.Now(),
		blockIndex: 50,
	}
}

// afterFrame must be called after every real frame the handler writes.
// Returns true when a fired fault terminated the stream.
func (inj *streamFaultInjector) afterFrame(event string) bool {
	if inj == nil {
		return false
	}
	inj.frames++
	return inj.matchAndFire(event)
}

// matchAndFire fires every armed spec whose WHEN matches. Injected frames
// re-enter matching (with their own event name, not counted as real
// frames), so specs can compose — e.g. disconnect after an injected
// `error` event. The fired flags make each spec one-shot, which bounds the
// recursion.
func (inj *streamFaultInjector) matchAndFire(event string) bool {
	for i := range inj.specs {
		if inj.fired[i] || !inj.matched(&inj.specs[i], event) {
			continue
		}
		inj.fired[i] = true
		if inj.fire(&inj.specs[i]) {
			return true
		}
	}
	return false
}

func (inj *streamFaultInjector) matched(f *FaultSpec, event string) bool {
	if f.AfterEvent != "" && f.AfterEvent != event {
		return false
	}
	if f.AfterN > 0 && inj.frames < f.AfterN {
		return false
	}
	if f.AfterBytes > 0 && inj.sse.bytes < f.AfterBytes {
		return false
	}
	if f.AfterMs > 0 && time.Since(inj.start) < time.Duration(f.AfterMs)*time.Millisecond {
		return false
	}
	return true
}

func (inj *streamFaultInjector) fire(f *FaultSpec) bool {
	repeat := f.Repeat
	if repeat < 1 {
		repeat = 1
	}
	switch f.Mode {
	case "disconnect":
		hijackAndClose(inj.w)
		return true

	case "stall":
		// A7: go silent while keeping the connection open — no bytes, no
		// FIN, no RST. The only way out is the client giving up (context
		// cancellation); then the stream is simply over.
		if inj.ctx != nil {
			<-inj.ctx.Done()
		}
		return true

	case "malformed_chunk":
		// Raw corrupt bytes by definition — never routed through
		// validation (being invalid is the fault's purpose).
		inj.sse.writeData("{INVALID JSON CORRUPT")

	case "unknown_event":
		// B1: a top-level event type outside the pinned MessageStreamEvent
		// union. Well-formed JSON whose type matches the SSE event name,
		// like a future API addition would look.
		eventType := f.EventType
		if eventType == "" {
			eventType = "message_future"
		}
		data, _ := json.Marshal(map[string]any{
			"type":   eventType,
			"future": map[string]any{"note": "mock unknown top-level event"},
		})
		for i := 0; i < repeat; i++ {
			if inj.emitInjected(eventType, string(data)) {
				return true
			}
		}

	case "unknown_block":
		// B2: a complete, well-formed content block (start + stop) of a
		// type the decoder does not map.
		for i := 0; i < repeat; i++ {
			idx := inj.blockIndex
			inj.blockIndex++
			start, _ := json.Marshal(map[string]any{
				"type":          "content_block_start",
				"index":         idx,
				"content_block": mockContentBlock(f.BlockType),
			})
			stop, _ := json.Marshal(map[string]any{
				"type":  "content_block_stop",
				"index": idx,
			})
			if inj.emitInjected("content_block_start", string(start)) {
				return true
			}
			if inj.emitInjected("content_block_stop", string(stop)) {
				return true
			}
		}

	case "stream_error":
		// B5: a mid-stream Anthropic `event: error`. The stream continues
		// afterwards; compose a disconnect spec keyed on after_event
		// "error" to cut it here instead.
		errType := f.ErrorType
		if errType == "" {
			errType = "overloaded_error"
		}
		msg := f.ErrorMessage
		if msg == "" {
			msg = "Simulated mid-stream provider error"
		}
		data, _ := json.Marshal(map[string]any{
			"type": "error",
			"error": map[string]any{
				"type":    errType,
				"message": msg,
			},
		})
		if inj.emitInjected("error", string(data)) {
			return true
		}
	}
	return false
}

// emitInjected writes an injected frame through the handler's validating
// writer, then lets other specs match on it (composition).
func (inj *streamFaultInjector) emitInjected(event, data string) bool {
	if inj.write(event, data) {
		return true
	}
	return inj.matchAndFire(event)
}

// mockContentBlock builds the content_block payload for an unknown_block
// fault: spec-accurate shapes for the real suppressed types nanollm
// allowlists (KNOWN_SUPPRESSED_BLOCK_TYPES), a generic probe otherwise.
func mockContentBlock(blockType string) map[string]any {
	if blockType == "" {
		blockType = "x_mock_unknown_block"
	}
	switch blockType {
	case "redacted_thinking":
		return map[string]any{
			"type": blockType,
			"data": "bW9ja19yZWRhY3RlZF90aGlua2luZw==",
		}
	case "server_tool_use":
		return map[string]any{
			"type":   blockType,
			"id":     "srvtoolu_mock01",
			"name":   "web_search",
			"input":  map[string]any{},
			"caller": map[string]any{"type": "direct"},
		}
	default:
		return map[string]any{
			"type":           blockType,
			"x_mock_payload": "unknown-block-tolerance-probe",
		}
	}
}

func hijackAndClose(w http.ResponseWriter) {
	if hj, ok := w.(http.Hijacker); ok {
		conn, _, err := hj.Hijack()
		if err == nil {
			if tc, ok := conn.(*net.TCPConn); ok {
				_ = tc.SetLinger(0)
			}
			conn.Close()
			return
		}
	}
	// Fallback for non-hijackable connections (e.g. HTTP/2):
	// close the response to simulate a broken connection as best we can.
	w.WriteHeader(http.StatusBadGateway)
}

// writeErrorResponse writes a provider-shaped error envelope carrying every
// key the pinned specs require: Anthropic ErrorResponse.required is
// {type, error, request_id}; OpenAI Error.required is
// {type, message, param, code} (param/code nullable).
//
// When the config snapshot (nil = env default) enables validate_responses,
// the envelope is checked against the provider's pinned ErrorResponse
// schema before writing — injected errors are part of the contract too.
// The "admin" pseudo-provider is not a spec surface and skips validation.
func writeErrorResponse(w http.ResponseWriter, cfg *ProviderConfig, status int, provider, errType, message string) {
	var body any
	if provider == "anthropic" {
		body = map[string]any{
			"type": "error",
			"error": map[string]any{
				"type":    errType,
				"message": message,
			},
			"request_id": fmt.Sprintf("req_mock_%d", time.Now().UnixNano()),
		}
	} else {
		body = map[string]any{
			"error": map[string]any{
				"message": message,
				"type":    errType,
				"param":   nil,
				"code":    errType,
			},
		}
	}

	data, err := marshalBody(body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if provider != "admin" && shouldValidate(cfg) {
		kind := kindOpenAIError
		if provider == "anthropic" {
			kind = kindAnthropicError
		}
		if verr := validateEmittedBody(kind, data); verr != nil {
			reportValidationFailure(provider+" error envelope", data, verr)
			writeValidationFailure(w, kind, provider+" error envelope", verr)
			return
		}
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	w.Write(data)
}
