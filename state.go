package main

import "sync"

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
