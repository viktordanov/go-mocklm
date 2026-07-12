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
	// Proto is r.Proto — "HTTP/2.0" when the client negotiated h2 over the
	// TLS lane's ALPN, "HTTP/1.1" otherwise. The transport-observation
	// oracle: assert the gateway actually spoke the protocol it claims.
	// Additive key; consumers ignoring unknown keys are unaffected.
	Proto   string            `json:"proto,omitempty"`
	Headers map[string]string `json:"headers"`
	Body    json.RawMessage   `json:"body"`
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
	bedrockRL    *RateLimiter
	activePreset string

	// Request recording
	recMu    sync.Mutex
	records  []RecordedRequest

	// Concurrency tracking
	openaiActive    atomic.Int32
	anthropicActive atomic.Int32
	bedrockActive   atomic.Int32

	// Per-provider request (attempt) counters: back the
	// /admin/request-count introspection oracle (counting ALL traffic,
	// scenario-matched included) and drive fail_first_n / attempt_faults
	// indexing for provider- and header-level configs. Matched scenarios
	// index their own per-scenario counters instead (scenario.go). Reset
	// on Update/Reset so a fresh provider fault config starts counting
	// from zero.
	openaiAttempts    atomic.Int64
	anthropicAttempts atomic.Int64
	bedrockAttempts   atomic.Int64

	// scenarios is the scenario registry. Deliberately NOT touched by
	// Update/Reset — scenarios are test fixtures with independent
	// lifetimes (DELETE /admin/scenarios clears them), and their
	// per-scenario fault-attempt counters survive a provider reset too
	// (R6): /admin/reset never re-arms a scenario's attempt faults.
	scenarios *ScenarioStore
}

// NewServerState creates a ServerState from the initial config and builds rate limiters.
func NewServerState(cfg *Config) *ServerState {
	s := &ServerState{
		cfg:        *cfg,
		initialCfg: *cfg,
		scenarios:  newScenarioStore(),
	}
	s.rebuildLimiters()

	return s
}

// Scenarios returns the scenario registry.
func (s *ServerState) Scenarios() *ScenarioStore {
	return s.scenarios
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

// Bedrock returns a snapshot of the Bedrock provider config and its rate limiter.
func (s *ServerState) Bedrock() (ProviderConfig, *RateLimiter) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.cfg.Bedrock, s.bedrockRL
}

// Config returns a snapshot of the full config and the active preset name.
func (s *ServerState) Config() (Config, string) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.cfg, s.activePreset
}

// Update replaces the provider configs, rebuilds rate limiters, and records
// the preset name. Scenarios (the scenario registry) survive untouched —
// only the provider attempt counters are zeroed; per-scenario fault-attempt
// counters are the scenarios' own state.
func (s *ServerState) Update(openai, anthropic, bedrock ProviderConfig, presetName string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.cfg.OpenAI = openai
	s.cfg.Anthropic = anthropic
	s.cfg.Bedrock = bedrock
	s.activePreset = presetName
	s.rebuildLimiters()
	s.ResetAttempts()
}

// Reset restores the initial startup config.
func (s *ServerState) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.cfg.OpenAI = s.initialCfg.OpenAI
	s.cfg.Anthropic = s.initialCfg.Anthropic
	s.cfg.Bedrock = s.initialCfg.Bedrock
	s.activePreset = ""
	s.rebuildLimiters()
	s.ResetAttempts()
}

// attemptsFor resolves the provider's attempt counter. Unknown providers
// panic loudly (R2): a silent else-branch here once aliased a third
// provider onto anthropic's counter — never again.
func (s *ServerState) attemptsFor(provider string) *atomic.Int64 {
	switch provider {
	case "openai":
		return &s.openaiAttempts
	case "anthropic":
		return &s.anthropicAttempts
	case "bedrock":
		return &s.bedrockAttempts
	}
	panic("mocklm: unknown provider " + provider)
}

// NextAttempt returns the 1-based sequence number of this request for the
// provider's attempt counter — the /admin/request-count oracle, and the
// fail_first_n / attempt_faults index for provider- and header-level
// configs (matched scenarios index their own counter, scenario.go).
func (s *ServerState) NextAttempt(provider string) int64 {
	return s.attemptsFor(provider).Add(1)
}

// AttemptCounts returns the per-provider request counts since the last
// reset/config update.
func (s *ServerState) AttemptCounts() (openai, anthropic, bedrock int64) {
	return s.openaiAttempts.Load(), s.anthropicAttempts.Load(), s.bedrockAttempts.Load()
}

// ResetAttempts zeroes the per-provider request counters.
func (s *ServerState) ResetAttempts() {
	s.openaiAttempts.Store(0)
	s.anthropicAttempts.Store(0)
	s.bedrockAttempts.Store(0)
}

// maxRecordedRequests caps the in-memory recording buffer; the oldest
// entries are dropped first. Prevents unbounded growth in long-lived runs.
const maxRecordedRequests = 1000

// RecordRequest stores a captured request.
func (s *ServerState) RecordRequest(rec RecordedRequest) {
	s.recMu.Lock()
	defer s.recMu.Unlock()
	s.records = append(s.records, rec)
	if len(s.records) > maxRecordedRequests {
		s.records = s.records[len(s.records)-maxRecordedRequests:]
	}
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

	if s.cfg.Bedrock.RateLimitRPM > 0 {
		s.bedrockRL = newRateLimiter(s.cfg.Bedrock.RateLimitRPM)
	} else {
		s.bedrockRL = nil
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

// counterFor resolves the provider's active-concurrency counter. Unknown
// providers panic loudly (R2): the old openai-else-anthropic branch would
// have silently aliased bedrock onto anthropic's concurrency gate.
func (s *ServerState) counterFor(provider string) *atomic.Int32 {
	switch provider {
	case "openai":
		return &s.openaiActive
	case "anthropic":
		return &s.anthropicActive
	case "bedrock":
		return &s.bedrockActive
	}
	panic("mocklm: unknown provider " + provider)
}

func (s *ServerState) providerConfig(provider string) (ProviderConfig, *RateLimiter) {
	switch provider {
	case "openai":
		return s.OpenAI()
	case "anthropic":
		return s.Anthropic()
	case "bedrock":
		return s.Bedrock()
	}
	panic("mocklm: unknown provider " + provider)
}
