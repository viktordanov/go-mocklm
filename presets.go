package main

// Preset is a named pair of provider configs with a human-readable description.
type Preset struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	OpenAI      ProviderConfig `json:"openai"`
	Anthropic   ProviderConfig `json:"anthropic"`
}

func builtinPresets() map[string]Preset {
	healthy := ProviderConfig{Tokens: 20, ErrorStatus: 500}

	return map[string]Preset{
		"healthy": {
			Name:        "healthy",
			Description: "No faults, happy path",
			OpenAI:      healthy,
			Anthropic:   healthy,
		},
		"openai-disconnect": {
			Name:        "openai-disconnect",
			Description: "OpenAI drops after 3 chunks; Anthropic healthy (stream resume)",
			OpenAI:      ProviderConfig{Tokens: 20, ErrorStatus: 500, DisconnectAfterChunks: 3},
			Anthropic:   healthy,
		},
		"anthropic-disconnect": {
			Name:        "anthropic-disconnect",
			Description: "Anthropic drops after 3 chunks; OpenAI healthy",
			OpenAI:      healthy,
			Anthropic:   ProviderConfig{Tokens: 20, ErrorStatus: 500, DisconnectAfterChunks: 3},
		},
		"openai-errors": {
			Name:        "openai-errors",
			Description: "OpenAI 503 always (fallback chain)",
			OpenAI:      ProviderConfig{Tokens: 20, ErrorRate: 1.0, ErrorStatus: 503},
			Anthropic:   healthy,
		},
		"anthropic-errors": {
			Name:        "anthropic-errors",
			Description: "Anthropic 529 always",
			OpenAI:      healthy,
			Anthropic:   ProviderConfig{Tokens: 20, ErrorRate: 1.0, ErrorStatus: 529},
		},
		"openai-rate-limited": {
			Name:        "openai-rate-limited",
			Description: "OpenAI 1 RPM (Retry-After handling)",
			OpenAI:      ProviderConfig{Tokens: 20, ErrorStatus: 500, RateLimitRPM: 1},
			Anthropic:   healthy,
		},
		"both-rate-limited": {
			Name:        "both-rate-limited",
			Description: "Both providers at 2 RPM",
			OpenAI:      ProviderConfig{Tokens: 20, ErrorStatus: 500, RateLimitRPM: 2},
			Anthropic:   ProviderConfig{Tokens: 20, ErrorStatus: 500, RateLimitRPM: 2},
		},
		"openai-slow": {
			Name:        "openai-slow",
			Description: "OpenAI 500ms latency, 200ms chunks (timeout detection)",
			OpenAI:      ProviderConfig{Tokens: 20, ErrorStatus: 500, LatencyMs: 500, StreamDelayMs: 200},
			Anthropic:   healthy,
		},
		"openai-timeout": {
			Name:        "openai-timeout",
			Description: "OpenAI 5s hold then drop (connection timeout)",
			OpenAI:      ProviderConfig{Tokens: 20, ErrorStatus: 500, TimeoutMs: 5000},
			Anthropic:   healthy,
		},
		"malformed-streams": {
			Name:        "malformed-streams",
			Description: "Both inject corrupt JSON (parse resilience)",
			OpenAI:      ProviderConfig{Tokens: 20, ErrorStatus: 500, MalformedChunk: true},
			Anthropic:   ProviderConfig{Tokens: 20, ErrorStatus: 500, MalformedChunk: true},
		},
		"flaky-openai": {
			Name:        "flaky-openai",
			Description: "OpenAI 50% error rate (probabilistic retry)",
			OpenAI:      ProviderConfig{Tokens: 20, ErrorRate: 0.5, ErrorStatus: 500},
			Anthropic:   healthy,
		},
	}
}
