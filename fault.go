package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"strconv"
	"sync"
	"time"
)

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
	if provider == "anthropic" {
		return status, anthropicErrorType(status)
	}
	switch {
	case status == 429:
		return status, "rate_limit_error"
	case status >= 500:
		return status, "server_error"
	default:
		return status, "invalid_request_error"
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
	// 0. Per-request fault targeting: the header overlays cfg, so knobs it
	// sets also steer everything downstream of checkFaults (stream faults,
	// stop reasons, strict mode, ...).
	if err := applyFaultHeader(r, cfg); err != nil {
		writeErrorResponse(w, 400, provider, "invalid_request_error", err.Error())
		return true
	}

	// 1. Rate limiting
	if cfg.RateLimitRPM > 0 && limiter != nil {
		allowed, retryAfter := limiter.Allow()
		if !allowed {
			w.Header().Set("Retry-After", strconv.Itoa(retryAfter))
			writeErrorResponse(w, 429, provider, "rate_limit_error", "Rate limit exceeded. Please retry after the specified time.")
			return true
		}
	}

	// 2a. Deterministic fail-first-N: the first N requests per provider fail
	// with error_status, then requests succeed. The counter lives on
	// ServerState (reset on config update/reset) and is shared by header-
	// and config-driven fail_first_n.
	if cfg.FailFirstN > 0 && state != nil {
		if seq := state.NextFailSeq(provider); seq <= int64(cfg.FailFirstN) {
			status, errType := faultErrorStatus(cfg, provider)
			writeErrorResponse(w, status, provider, errType, fmt.Sprintf("Simulated deterministic error %d/%d (status %d)", seq, cfg.FailFirstN, status))
			return true
		}
	}

	// 2b. Random error
	if cfg.ErrorRate > 0 && rand.Float64() < cfg.ErrorRate {
		status, errType := faultErrorStatus(cfg, provider)
		writeErrorResponse(w, status, provider, errType, fmt.Sprintf("Simulated error (status %d)", status))
		return true
	}

	// 3. Timeout (hold then close)
	if cfg.TimeoutMs > 0 {
		time.Sleep(time.Duration(cfg.TimeoutMs) * time.Millisecond)
		hijackAndClose(w)
		return true
	}

	// 4. Latency
	if cfg.LatencyMs > 0 {
		time.Sleep(time.Duration(cfg.LatencyMs) * time.Millisecond)
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
func writeErrorResponse(w http.ResponseWriter, status int, provider, errType, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)

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

	json.NewEncoder(w).Encode(body)
}
