package main

import (
	"fmt"
	"os"

	"github.com/BurntSushi/toml"
)

type ServerConfig struct {
	Host string `toml:"host"`
	Port int    `toml:"port"`
}

type ProviderConfig struct {
	LatencyMs              int     `toml:"latency_ms" json:"latency_ms"`
	Tokens                 int     `toml:"tokens" json:"tokens"`
	StreamDelayMs          int     `toml:"stream_delay_ms" json:"stream_delay_ms"`
	ErrorRate              float64 `toml:"error_rate" json:"error_rate"`
	ErrorStatus            int     `toml:"error_status" json:"error_status"`
	TimeoutMs              int     `toml:"timeout_ms" json:"timeout_ms"`
	DisconnectAfterChunks  int     `toml:"disconnect_after_chunks" json:"disconnect_after_chunks"`
	MalformedChunk         bool    `toml:"malformed_chunk" json:"malformed_chunk"`
	RateLimitRPM           int     `toml:"rate_limit_rpm" json:"rate_limit_rpm"`
	ReasoningTokens        int     `toml:"reasoning_tokens" json:"reasoning_tokens"`
	ThinkingDelayMs        int     `toml:"thinking_delay_ms" json:"thinking_delay_ms"`
	Deterministic          bool    `toml:"deterministic" json:"deterministic"`
	ToolUseResponse        bool    `toml:"tool_use_response" json:"tool_use_response"`
	HonorMaxTokens         bool    `toml:"honor_max_tokens" json:"honor_max_tokens"`
	MinTokens              int     `toml:"min_tokens" json:"min_tokens"`
	TtftMs                 int     `toml:"ttft_ms" json:"ttft_ms"`
	StreamDelayJitterMs    int     `toml:"stream_delay_jitter_ms" json:"stream_delay_jitter_ms"`
	SlowHeaderMs           int     `toml:"slow_header_ms" json:"slow_header_ms"`
	MaxConcurrent          int     `toml:"max_concurrent" json:"max_concurrent"`
	SseKeepaliveIntervalMs int     `toml:"sse_keepalive_interval_ms" json:"sse_keepalive_interval_ms"`
	// StopReason overrides the emitted stop reason. Anthropic responses use
	// it for stop_reason (e.g. "pause_turn", "refusal"); OpenAI responses
	// use it for finish_reason (e.g. "content_filter"). Empty = default.
	StopReason string `toml:"stop_reason" json:"stop_reason"`
	// Strict enables contract-oracle request validation (Anthropic):
	// unknown top-level fields, missing required fields, and out-of-range
	// values are rejected with 400 like the real API (strict.go).
	Strict bool `toml:"strict" json:"strict"`
	// CacheReadTokens / CacheCreationTokens drive prompt-cache usage fields:
	// Anthropic cache_read_input_tokens / cache_creation_input_tokens, and
	// OpenAI prompt_tokens_details.cached_tokens.
	CacheReadTokens     int `toml:"cache_read_tokens" json:"cache_read_tokens"`
	CacheCreationTokens int `toml:"cache_creation_tokens" json:"cache_creation_tokens"`
	// FailFirstN deterministically fails the first N requests seen by the
	// provider (with error_status), then succeeds. The counter resets on
	// config update/reset. Deterministic alternative to error_rate.
	FailFirstN int `toml:"fail_first_n" json:"fail_first_n"`
	// DisconnectAfterEvent cuts the Anthropic stream (TCP RST) immediately
	// after writing the named SSE event type, e.g. "message_delta" leaves
	// the client with a stop_reason but no message_stop, and
	// "content_block_start" cuts before any delta arrives.
	DisconnectAfterEvent string `toml:"disconnect_after_event" json:"disconnect_after_event"`
	// EmitNonstandardFields injects genuinely-unknown fields that are NOT in
	// the pinned Anthropic spec (x_mock_unknown_field on message shapes and
	// message_delta.delta, x_mock_unknown_usage_field on usage objects) as a
	// deliberate unknown-field-tolerance fault. Off by default: responses
	// carry only spec fields. Note the spec-required nullable fields
	// (stop_details, usage.inference_geo, usage.output_tokens_details) are
	// always emitted — they are part of Message.required / Usage.required in
	// nanollm's pinned spec, not fabrications.
	EmitNonstandardFields bool `toml:"emit_nonstandard_fields" json:"emit_nonstandard_fields"`
	// SuppressPingEvents omits the typed Anthropic ping event after
	// message_start. The real API sends it (and it is emitted by default),
	// but the pinned spec's MessageStreamEvent union has no ping arm — a
	// validator checking every data: payload against that union needs either
	// a ping exception or this knob.
	SuppressPingEvents bool `toml:"suppress_ping_events" json:"suppress_ping_events"`
	// LegacyStreamUsage restores the pre-fidelity OpenAI streaming usage
	// shape: usage rides the finish chunk unconditionally and
	// stream_options.include_usage is ignored. Off by default: the real API
	// shape (usage: null on chunks + trailing choices:[] usage chunk, only
	// when include_usage is requested) is emitted.
	LegacyStreamUsage bool `toml:"legacy_stream_usage" json:"legacy_stream_usage"`
	// ValidateResponses checks the bodies covered by the vendored
	// response-side closure of nanollm's pinned specs before writing them
	// (validator.go): OpenAI chat-completion and Anthropic messages bodies
	// (non-stream responses and each SSE data payload) plus provider error
	// envelopes on every endpoint. Success bodies of /v1/completions,
	// /v1/embeddings, /v1/responses, and /v1/models are outside the
	// extracted closure and are NOT validated. Violations fail loudly: 500
	// before headers, RST mid-stream, or panic with MOCKLM_VALIDATE_PANIC.
	// Tri-state: unset defers to the MOCKLM_VALIDATE_RESPONSES env
	// default, so a harness can force it on globally while a
	// deliberate-fault request opts out with X-MockLM-Fault
	// {"validate_responses": false}. Deliberately-off-spec output
	// (malformed_chunk, emit_nonstandard_fields) bypasses validation by
	// design. The typed ping event validates against a local ping arm,
	// since the pinned MessageStreamEvent union has none.
	ValidateResponses *bool `toml:"validate_responses" json:"validate_responses,omitempty"`
}

type Config struct {
	Server    ServerConfig   `toml:"server"`
	OpenAI    ProviderConfig `toml:"openai"`
	Anthropic ProviderConfig `toml:"anthropic"`
}

func loadConfig() (*Config, error) {
	path := os.Getenv("CONFIG_PATH")
	if path == "" {
		path = "config.toml"
	}

	var cfg Config
	if _, err := toml.DecodeFile(path, &cfg); err != nil {
		return nil, fmt.Errorf("loading config from %s: %w", path, err)
	}

	if cfg.Server.Port == 0 {
		cfg.Server.Port = 9999
	}
	if cfg.Server.Host == "" {
		cfg.Server.Host = "0.0.0.0"
	}

	applyProviderDefaults(&cfg.OpenAI)
	applyProviderDefaults(&cfg.Anthropic)

	return &cfg, nil
}

func applyProviderDefaults(p *ProviderConfig) {
	if p.Tokens == 0 {
		p.Tokens = 20
	}

	if p.ErrorStatus == 0 {
		p.ErrorStatus = 500
	}

	if p.MinTokens == 0 {
		p.MinTokens = 1
	}
}

func (c *Config) summary() string {
	return fmt.Sprintf(
		"Server: %s:%d\n"+
			"OpenAI:    tokens=%d latency=%dms stream_delay=%dms error_rate=%.2f error_status=%d timeout=%dms disconnect_after=%d malformed=%v rate_limit=%drpm reasoning_tokens=%d thinking_delay=%dms\n"+
			"Anthropic: tokens=%d latency=%dms stream_delay=%dms error_rate=%.2f error_status=%d timeout=%dms disconnect_after=%d malformed=%v rate_limit=%drpm reasoning_tokens=%d thinking_delay=%dms",
		c.Server.Host, c.Server.Port,
		c.OpenAI.Tokens, c.OpenAI.LatencyMs, c.OpenAI.StreamDelayMs, c.OpenAI.ErrorRate, c.OpenAI.ErrorStatus, c.OpenAI.TimeoutMs, c.OpenAI.DisconnectAfterChunks, c.OpenAI.MalformedChunk, c.OpenAI.RateLimitRPM, c.OpenAI.ReasoningTokens, c.OpenAI.ThinkingDelayMs,
		c.Anthropic.Tokens, c.Anthropic.LatencyMs, c.Anthropic.StreamDelayMs, c.Anthropic.ErrorRate, c.Anthropic.ErrorStatus, c.Anthropic.TimeoutMs, c.Anthropic.DisconnectAfterChunks, c.Anthropic.MalformedChunk, c.Anthropic.RateLimitRPM, c.Anthropic.ReasoningTokens, c.Anthropic.ThinkingDelayMs,
	)
}
