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
	LatencyMs             int     `toml:"latency_ms" json:"latency_ms"`
	Tokens                int     `toml:"tokens" json:"tokens"`
	StreamDelayMs         int     `toml:"stream_delay_ms" json:"stream_delay_ms"`
	ErrorRate             float64 `toml:"error_rate" json:"error_rate"`
	ErrorStatus           int     `toml:"error_status" json:"error_status"`
	TimeoutMs             int     `toml:"timeout_ms" json:"timeout_ms"`
	DisconnectAfterChunks int     `toml:"disconnect_after_chunks" json:"disconnect_after_chunks"`
	MalformedChunk        bool    `toml:"malformed_chunk" json:"malformed_chunk"`
	RateLimitRPM          int     `toml:"rate_limit_rpm" json:"rate_limit_rpm"`
	ReasoningTokens        int     `toml:"reasoning_tokens" json:"reasoning_tokens"`
	ThinkingDelayMs        int     `toml:"thinking_delay_ms" json:"thinking_delay_ms"`
	Deterministic          bool    `toml:"deterministic" json:"deterministic"`
	ToolUseResponse        bool    `toml:"tool_use_response" json:"tool_use_response"`
	HonorMaxTokens         bool   `toml:"honor_max_tokens" json:"honor_max_tokens"`
	MinTokens              int    `toml:"min_tokens" json:"min_tokens"`
	TtftMs                 int    `toml:"ttft_ms" json:"ttft_ms"`
	StreamDelayJitterMs    int    `toml:"stream_delay_jitter_ms" json:"stream_delay_jitter_ms"`
	SlowHeaderMs           int    `toml:"slow_header_ms" json:"slow_header_ms"`
	MaxConcurrent          int    `toml:"max_concurrent" json:"max_concurrent"`
	SseKeepaliveIntervalMs int    `toml:"sse_keepalive_interval_ms" json:"sse_keepalive_interval_ms"`
	// StopReason overrides the emitted stop reason. Anthropic responses use
	// it for stop_reason (e.g. "pause_turn", "refusal"); OpenAI responses
	// use it for finish_reason (e.g. "content_filter"). Empty = default.
	StopReason string `toml:"stop_reason" json:"stop_reason"`
	// CacheReadTokens / CacheCreationTokens add Anthropic prompt-cache usage
	// fields (cache_read_input_tokens / cache_creation_input_tokens) to
	// responses when > 0.
	CacheReadTokens     int `toml:"cache_read_tokens" json:"cache_read_tokens"`
	CacheCreationTokens int `toml:"cache_creation_tokens" json:"cache_creation_tokens"`
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
