package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"strings"
	"time"
)

func handleOpenAIChat(state *ServerState) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		cfg, limiter := state.OpenAI()

		// Max concurrent check
		allowed, acquired := state.AcquireConcurrency("openai")
		if !allowed {
			writeErrorResponse(w, 503, "openai", "overloaded", "Too many concurrent requests")
			return
		}
		if acquired {
			defer state.ReleaseConcurrency("openai")
		}

		// Buffer body for recording
		rawBody, err := io.ReadAll(r.Body)
		if err != nil {
			writeErrorResponse(w, 400, "openai", "invalid_request_error", "Failed to read body: "+err.Error())
			return
		}

		// Record request
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
			Messages  []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(bytes.NewReader(rawBody)).Decode(&req); err != nil {
			writeErrorResponse(w, 400, "openai", "invalid_request_error", "Invalid JSON: "+err.Error())
			return
		}

		if checkFaults(w, r, &cfg, limiter, "openai") {
			return
		}

		if cfg.ThinkingDelayMs > 0 {
			time.Sleep(time.Duration(cfg.ThinkingDelayMs) * time.Millisecond)
		}

		// Resolve output tokens: header > body > config
		outputTokens, err := resolveTokenCount(r, &cfg, req.MaxTokens)
		if err != nil {
			writeErrorResponse(w, 400, "openai", "invalid_request_error", err.Error())
			return
		}

		totalChars := 0
		for _, m := range req.Messages {
			totalChars += len(m.Content)
		}
		promptTokens := totalChars / 4
		if promptTokens < 1 {
			promptTokens = 1
		}

		words := generateWords(outputTokens)
		model := req.Model
		if model == "" {
			model = "gpt-4"
		}
		id := fmt.Sprintf("chatcmpl-mock-%d", time.Now().UnixNano())

		// slow_header_ms delay
		if cfg.SlowHeaderMs > 0 {
			time.Sleep(time.Duration(cfg.SlowHeaderMs) * time.Millisecond)
		}

		if req.Stream {
			handleOpenAIChatStream(w, &cfg, id, model, words, promptTokens, outputTokens)
		} else {
			handleOpenAIChatNonStream(w, &cfg, id, model, words, promptTokens, outputTokens)
		}
	}
}

func handleOpenAIChatNonStream(w http.ResponseWriter, cfg *ProviderConfig, id, model string, words []string, promptTokens, completionTokens int) {
	if cfg.ReasoningTokens > 0 {
		completionTokens = len(words) + cfg.ReasoningTokens
	}

	content := joinContent(words)
	usage := map[string]any{
		"prompt_tokens":     promptTokens,
		"completion_tokens": completionTokens,
		"total_tokens":      promptTokens + completionTokens,
	}
	if cfg.ReasoningTokens > 0 {
		usage["completion_tokens_details"] = map[string]any{
			"reasoning_tokens":           cfg.ReasoningTokens,
			"accepted_prediction_tokens": 0,
			"rejected_prediction_tokens": 0,
		}
		usage["prompt_tokens_details"] = map[string]any{
			"cached_tokens": 0,
		}
	}

	resp := map[string]any{
		"id":                 id,
		"object":             "chat.completion",
		"created":            time.Now().Unix(),
		"model":              model,
		"system_fingerprint": "fp_mock",
		"choices": []map[string]any{
			{
				"index": 0,
				"message": map[string]any{
					"role":    "assistant",
					"content": content,
				},
				"logprobs":      nil,
				"finish_reason": "stop",
			},
		},
		"usage": usage,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func handleOpenAIChatStream(w http.ResponseWriter, cfg *ProviderConfig, id, model string, words []string, promptTokens, completionTokens int) {
	sse := newSSEWriter(w)

	// TTFT delay
	if cfg.TtftMs > 0 {
		sleepWithPings(sse, cfg.TtftMs, cfg.SseKeepaliveIntervalMs)
	}

	// Role chunk
	roleChunk := map[string]any{
		"id":                 id,
		"object":             "chat.completion.chunk",
		"created":            time.Now().Unix(),
		"model":              model,
		"system_fingerprint": "fp_mock",
		"choices": []map[string]any{
			{
				"index": 0,
				"delta": map[string]any{
					"role":    "assistant",
					"content": "",
				},
				"logprobs":      nil,
				"finish_reason": nil,
			},
		},
	}
	data, _ := json.Marshal(roleChunk)
	sse.writeData(string(data))

	// Content chunks
	for i, word := range words {
		if checkStreamingFault(w, cfg, i, len(words)) {
			return
		}

		delay := cfg.StreamDelayMs
		if cfg.StreamDelayJitterMs > 0 {
			jitter := rand.Intn(cfg.StreamDelayJitterMs*2+1) - cfg.StreamDelayJitterMs
			delay += jitter
			if delay < 0 {
				delay = 0
			}
		}
		if delay > 0 {
			sleepWithPings(sse, delay, cfg.SseKeepaliveIntervalMs)
		}

		token := word
		if i == 0 {
			// Capitalize first word
			token = capitalize(token)
		}
		if i == len(words)-1 {
			token += "."
		} else {
			token += " "
		}

		contentChunk := map[string]any{
			"id":                 id,
			"object":             "chat.completion.chunk",
			"created":            time.Now().Unix(),
			"model":              model,
			"system_fingerprint": "fp_mock",
			"choices": []map[string]any{
				{
					"index": 0,
					"delta": map[string]any{
						"content": token,
					},
					"logprobs":      nil,
					"finish_reason": nil,
				},
			},
		}
		data, _ := json.Marshal(contentChunk)
		sse.writeData(string(data))
	}

	// Adjust completion tokens for reasoning
	if cfg.ReasoningTokens > 0 {
		completionTokens = len(words) + cfg.ReasoningTokens
	}

	// Finish chunk with usage
	streamUsage := map[string]any{
		"prompt_tokens":     promptTokens,
		"completion_tokens": completionTokens,
		"total_tokens":      promptTokens + completionTokens,
	}
	if cfg.ReasoningTokens > 0 {
		streamUsage["completion_tokens_details"] = map[string]any{
			"reasoning_tokens":           cfg.ReasoningTokens,
			"accepted_prediction_tokens": 0,
			"rejected_prediction_tokens": 0,
		}
		streamUsage["prompt_tokens_details"] = map[string]any{
			"cached_tokens": 0,
		}
	}
	finishChunk := map[string]any{
		"id":                 id,
		"object":             "chat.completion.chunk",
		"created":            time.Now().Unix(),
		"model":              model,
		"system_fingerprint": "fp_mock",
		"choices": []map[string]any{
			{
				"index":         0,
				"delta":         map[string]any{},
				"logprobs":      nil,
				"finish_reason": "stop",
			},
		},
		"usage": streamUsage,
	}
	data, _ = json.Marshal(finishChunk)
	sse.writeData(string(data))

	sse.writeDone()
}

func handleOpenAIModels() http.HandlerFunc {
	models := []map[string]any{
		{"id": "gpt-4", "object": "model", "owned_by": "openai"},
		{"id": "gpt-4o", "object": "model", "owned_by": "openai"},
		{"id": "gpt-3.5-turbo", "object": "model", "owned_by": "openai"},
	}

	resp := map[string]any{
		"object": "list",
		"data":   models,
	}

	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}
}

func capitalize(s string) string {
	if s == "" {
		return s
	}
	r := []rune(s)
	return strings.ToUpper(string(r[0])) + string(r[1:])
}

// sleepWithPings sleeps for totalMs, emitting SSE pings at keepaliveMs intervals.
func sleepWithPings(sse *sseWriter, totalMs, keepaliveMs int) {
	if keepaliveMs <= 0 || totalMs <= keepaliveMs {
		time.Sleep(time.Duration(totalMs) * time.Millisecond)
		return
	}
	remaining := totalMs
	for remaining > 0 {
		chunk := keepaliveMs
		if chunk > remaining {
			chunk = remaining
		}
		time.Sleep(time.Duration(chunk) * time.Millisecond)
		remaining -= chunk
		if remaining > 0 {
			sse.writePing()
		}
	}
}
