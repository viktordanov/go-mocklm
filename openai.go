package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

func handleOpenAIChat(state *ServerState) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		cfg, limiter := state.OpenAI()

		var req struct {
			Model    string `json:"model"`
			Stream   bool   `json:"stream"`
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeErrorResponse(w, 400, "openai", "invalid_request_error", "Invalid JSON: "+err.Error())
			return
		}

		if checkFaults(w, r, &cfg, limiter, "openai") {
			return
		}

		if cfg.ThinkingDelayMs > 0 {
			time.Sleep(time.Duration(cfg.ThinkingDelayMs) * time.Millisecond)
		}

		// Calculate mock token counts
		totalChars := 0
		for _, m := range req.Messages {
			totalChars += len(m.Content)
		}
		promptTokens := totalChars / 4
		if promptTokens < 1 {
			promptTokens = 1
		}
		completionTokens := cfg.Tokens

		words := generateWords(cfg.Tokens)
		model := req.Model
		if model == "" {
			model = "gpt-4"
		}
		id := fmt.Sprintf("chatcmpl-mock-%d", time.Now().UnixNano())

		if req.Stream {
			handleOpenAIChatStream(w, &cfg, id, model, words, promptTokens, completionTokens)
		} else {
			handleOpenAIChatNonStream(w, &cfg, id, model, words, promptTokens, completionTokens)
		}
	}
}

func handleOpenAIChatNonStream(w http.ResponseWriter, cfg *ProviderConfig, id, model string, words []string, promptTokens, completionTokens int) {
	if cfg.ReasoningTokens > 0 {
		completionTokens = cfg.Tokens + cfg.ReasoningTokens
	}

	content := joinContent(words)
	usage := map[string]any{
		"prompt_tokens":     promptTokens,
		"completion_tokens": completionTokens,
		"total_tokens":      promptTokens + completionTokens,
	}
	if cfg.ReasoningTokens > 0 {
		usage["completion_tokens_details"] = map[string]any{
			"reasoning_tokens":            cfg.ReasoningTokens,
			"accepted_prediction_tokens":  0,
			"rejected_prediction_tokens":  0,
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

		if cfg.StreamDelayMs > 0 {
			time.Sleep(time.Duration(cfg.StreamDelayMs) * time.Millisecond)
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
		completionTokens = cfg.Tokens + cfg.ReasoningTokens
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
