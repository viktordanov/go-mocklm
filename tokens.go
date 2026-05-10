package main

import (
	"fmt"
	"net/http"
	"strconv"
)

// resolveTokenCount determines the output token count based on priority:
// 1. X-MockLM-Tokens header (error if present but invalid)
// 2. body max_tokens (only if honor_max_tokens enabled)
// 3. config tokens
func resolveTokenCount(r *http.Request, cfg *ProviderConfig, bodyMaxTokens int) (int, error) {
	// Priority 1: X-MockLM-Tokens header
	if h := r.Header.Get("X-MockLM-Tokens"); h != "" {
		v, err := strconv.Atoi(h)
		if err != nil {
			return 0, fmt.Errorf("invalid X-MockLM-Tokens header value: %q", h)
		}
		if v < cfg.MinTokens {
			return cfg.MinTokens, nil
		}
		return v, nil
	}

	// Priority 2: body max_tokens (only if honor_max_tokens enabled)
	if cfg.HonorMaxTokens && bodyMaxTokens > 0 {
		if bodyMaxTokens < cfg.MinTokens {
			return cfg.MinTokens, nil
		}
		return bodyMaxTokens, nil
	}

	// Priority 3: config tokens
	return cfg.Tokens, nil
}
