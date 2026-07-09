package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// handleOpenAICompletions emulates the legacy OpenAI completions API
// (POST /v1/completions). Mirrors the chat handler: records the raw request,
// honors fault injection and token resolution, supports streaming.
func handleOpenAICompletions(state *ServerState) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		cfg, limiter := state.OpenAI()

		// R4: scenarios are not honored on the legacy completions surface —
		// a targeting header is a loud 400, not a silent no-op.
		if rejectScenarioHeaderUnwired(w, r, &cfg, "openai") {
			return
		}

		allowed, acquired := state.AcquireConcurrency("openai")
		if !allowed {
			writeErrorResponse(w, &cfg, 503, "openai", "server_error", "Too many concurrent requests")
			return
		}
		if acquired {
			defer state.ReleaseConcurrency("openai")
		}

		rawBody, err := io.ReadAll(r.Body)
		if err != nil {
			writeErrorResponse(w, &cfg, 400, "openai", "invalid_request_error", "Failed to read body: "+err.Error())
			return
		}

		headers := make(map[string]string)
		for k := range r.Header {
			headers[k] = r.Header.Get(k)
		}
		state.RecordRequest(RecordedRequest{
			Timestamp: time.Now(),
			Provider:  "openai",
			Method:    r.Method,
			Path:      r.URL.Path,
			Headers:   headers,
			Body:      json.RawMessage(rawBody),
		})

		var req struct {
			Model     string `json:"model"`
			Stream    bool   `json:"stream"`
			MaxTokens int    `json:"max_tokens"`
		}
		if err := json.NewDecoder(bytes.NewReader(rawBody)).Decode(&req); err != nil {
			writeErrorResponse(w, &cfg, 400, "openai", "invalid_request_error", "Invalid JSON: "+err.Error())
			return
		}

		if checkFaults(w, r, &cfg, limiter, state, "openai") {
			return
		}

		outputTokens, err := resolveTokenCount(r, &cfg, req.MaxTokens)
		if err != nil {
			writeErrorResponse(w, &cfg, 400, "openai", "invalid_request_error", err.Error())
			return
		}

		words := generateWords(outputTokens)
		if cfg.Deterministic {
			words = generateDeterministicWords(outputTokens)
		}
		model := req.Model
		if model == "" {
			model = "gpt-3.5-turbo-instruct"
		}
		id := fmt.Sprintf("cmpl-mock-%d", time.Now().UnixNano())
		if cfg.Deterministic {
			id = "cmpl-mock-deterministic"
		}

		if cfg.SlowHeaderMs > 0 {
			if !waitCancelable(r.Context(), cfg.SlowHeaderMs) {
				return
			}
		}

		if req.Stream {
			handleCompletionsStream(r.Context(), w, &cfg, id, model, words, outputTokens)
		} else {
			handleCompletionsNonStream(w, &cfg, id, model, words, outputTokens)
		}
	}
}

func handleCompletionsNonStream(w http.ResponseWriter, _ *ProviderConfig, id, model string, words []string, outputTokens int) {
	resp := map[string]any{
		"id":      id,
		"object":  "text_completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []map[string]any{
			{
				"text":          joinContent(words),
				"index":         0,
				"logprobs":      nil,
				"finish_reason": "stop",
			},
		},
		"usage": map[string]any{
			"prompt_tokens":     1,
			"completion_tokens": outputTokens,
			"total_tokens":      1 + outputTokens,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func handleCompletionsStream(ctx context.Context, w http.ResponseWriter, cfg *ProviderConfig, id, model string, words []string, outputTokens int) {
	sse := newSSEWriter(w)

	if cfg.TtftMs > 0 {
		if !sleepWithPings(ctx, sse, cfg.TtftMs, cfg.SseKeepaliveIntervalMs) {
			return
		}
	}

	// Content chunk (single chunk keeps the legacy emulation simple)
	chunk, _ := json.Marshal(map[string]any{
		"id":      id,
		"object":  "text_completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []map[string]any{
			{"text": joinContent(words), "index": 0, "logprobs": nil, "finish_reason": nil},
		},
	})
	sse.writeData(string(chunk))

	// Finish chunk with usage
	finish, _ := json.Marshal(map[string]any{
		"id":      id,
		"object":  "text_completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []map[string]any{
			{"text": "", "index": 0, "logprobs": nil, "finish_reason": "stop"},
		},
		"usage": map[string]any{
			"prompt_tokens":     1,
			"completion_tokens": outputTokens,
			"total_tokens":      1 + outputTokens,
		},
	})
	sse.writeData(string(finish))
	sse.writeData("[DONE]")
}
