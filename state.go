package main

import (
	"encoding/json"
	"sync"
	"sync/atomic"
	"time"
)

// RecordedRequest holds a captured upstream request.
type RecordedRequest struct {
	Timestamp time.Time              `json:"timestamp"`
	Provider  string                 `json:"provider"`
	Method    string                 `json:"method"`
	Path      string                 `json:"path"`
	Headers   map[string]string      `json:"headers"`
	Body      json.RawMessage        `json:"body"`
}

// ServerState holds the mutable server configuration protected by a RWMutex.
// Handlers read per-request snapshots via OpenAI()/Anthropic(); the admin API
// swaps the whole config atomically via Update().
type ServerState struct {
	mu           sync.RWMutex
	cfg          Config
	initialCfg   Config
	openaiRL     *RateLimiter
	anthropicRL  *RateLimiter
	activePreset string

	// Request recording
	recMu    sync.Mutex
	records  []RecordedRequest

	// Concurrency tracking
	openaiActive    atomic.Int32
	anthropicActive atomic.Int32
}

// NewServerState creates a ServerState from the initial config and builds rate limiters.
func NewServerState(cfg *Config) *ServerState {
	s := &ServerState{
		cfg:        *cfg,
		initialCfg: *cfg,
	}
	s.rebuildLimiters()

	return s
}

// OpenAI returns a snapshot of the OpenAI provider config and its rate limiter.
func (s *ServerState) OpenAI() (ProviderConfig, *RateLimiter) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.cfg.OpenAI, s.openaiRL
}

// Anthropic returns a snapshot of the Anthropic provider config and its rate limiter.
func (s *ServerState) Anthropic() (ProviderConfig, *RateLimiter) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.cfg.Anthropic, s.anthropicRL
}

// Config returns a snapshot of the full config and the active preset name.
func (s *ServerState) Config() (Config, string) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.cfg, s.activePreset
}

// Update replaces the provider configs, rebuilds rate limiters, and records the preset name.
func (s *ServerState) Update(openai, anthropic ProviderConfig, presetName string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.cfg.OpenAI = openai
	s.cfg.Anthropic = anthropic
	s.activePreset = presetName
	s.rebuildLimiters()
}

// Reset restores the initial startup config.
func (s *ServerState) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.cfg.OpenAI = s.initialCfg.OpenAI
	s.cfg.Anthropic = s.initialCfg.Anthropic
	s.activePreset = ""
	s.rebuildLimiters()
}

// RecordRequest stores a captured request.
func (s *ServerState) RecordRequest(rec RecordedRequest) {
	s.recMu.Lock()
	defer s.recMu.Unlock()
	s.records = append(s.records, rec)
}

// Requests returns all recorded requests.
func (s *ServerState) Requests() []RecordedRequest {
	s.recMu.Lock()
	defer s.recMu.Unlock()
	out := make([]RecordedRequest, len(s.records))
	copy(out, s.records)
	return out
}

// ClearRequests removes all recorded requests.
func (s *ServerState) ClearRequests() {
	s.recMu.Lock()
	defer s.recMu.Unlock()
	s.records = nil
}

// rebuildLimiters must be called under write lock.
func (s *ServerState) rebuildLimiters() {
	if s.cfg.OpenAI.RateLimitRPM > 0 {
		s.openaiRL = newRateLimiter(s.cfg.OpenAI.RateLimitRPM)
	} else {
		s.openaiRL = nil
	}

	if s.cfg.Anthropic.RateLimitRPM > 0 {
		s.anthropicRL = newRateLimiter(s.cfg.Anthropic.RateLimitRPM)
	} else {
		s.anthropicRL = nil
	}
}

// AcquireConcurrency attempts to acquire a concurrency slot for the provider.
// Returns (allowed, acquired): allowed=false means 503; acquired=true means
// ReleaseConcurrency must be called when done.
func (s *ServerState) AcquireConcurrency(provider string) (allowed bool, acquired bool) {
	cfg, _ := s.providerConfig(provider)
	if cfg.MaxConcurrent <= 0 {
		return true, false
	}
	counter := s.counterFor(provider)
	for {
		current := counter.Load()
		if int(current) >= cfg.MaxConcurrent {
			return false, false
		}
		if counter.CompareAndSwap(current, current+1) {
			return true, true
		}
	}
}

// ReleaseConcurrency releases a concurrency slot for the provider.
func (s *ServerState) ReleaseConcurrency(provider string) {
	counter := s.counterFor(provider)
	counter.Add(-1)
}

func (s *ServerState) counterFor(provider string) *atomic.Int32 {
	if provider == "openai" {
		return &s.openaiActive
	}
	return &s.anthropicActive
}

func (s *ServerState) providerConfig(provider string) (ProviderConfig, *RateLimiter) {
	if provider == "openai" {
		return s.OpenAI()
	}
	return s.Anthropic()
}
