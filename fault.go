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

// checkFaults runs fault injection checks. Returns true if the request was handled (caller should return).
func checkFaults(w http.ResponseWriter, _ *http.Request, cfg *ProviderConfig, limiter *RateLimiter, provider string) bool {
	// 1. Rate limiting
	if cfg.RateLimitRPM > 0 && limiter != nil {
		allowed, retryAfter := limiter.Allow()
		if !allowed {
			w.Header().Set("Retry-After", strconv.Itoa(retryAfter))
			writeErrorResponse(w, 429, provider, "rate_limit_error", "Rate limit exceeded. Please retry after the specified time.")
			return true
		}
	}

	// 2. Random error
	if cfg.ErrorRate > 0 && rand.Float64() < cfg.ErrorRate {
		errType := "server_error"
		if cfg.ErrorStatus == 429 {
			errType = "rate_limit_error"
		}
		writeErrorResponse(w, cfg.ErrorStatus, provider, errType, fmt.Sprintf("Simulated error (status %d)", cfg.ErrorStatus))
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
		}
	} else {
		body = map[string]any{
			"error": map[string]any{
				"message": message,
				"type":    errType,
				"code":    errType,
			},
		}
	}

	json.NewEncoder(w).Encode(body)
}
