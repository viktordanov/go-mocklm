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
			writeErrorResponse(w, &cfg, 503, "openai", "server_error", "Too many concurrent requests")
			return
		}
		if acquired {
			defer state.ReleaseConcurrency("openai")
		}

		// Buffer body for recording
		rawBody, err := io.ReadAll(r.Body)
		if err != nil {
			writeErrorResponse(w, &cfg, 400, "openai", "invalid_request_error", "Failed to read body: "+err.Error())
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
			Model         string          `json:"model"`
			Stream        bool            `json:"stream"`
			MaxTokens     int             `json:"max_tokens"`
			Messages      []chatMessage   `json:"messages"`
			Tools         json.RawMessage `json:"tools"`
			StreamOptions *struct {
				IncludeUsage bool `json:"include_usage"`
			} `json:"stream_options"`
		}
		if err := json.NewDecoder(bytes.NewReader(rawBody)).Decode(&req); err != nil {
			writeErrorResponse(w, &cfg, 400, "openai", "invalid_request_error", "Invalid JSON: "+err.Error())
			return
		}

		if checkFaults(w, r, &cfg, limiter, state, "openai") {
			return
		}

		if cfg.ThinkingDelayMs > 0 {
			time.Sleep(time.Duration(cfg.ThinkingDelayMs) * time.Millisecond)
		}

		// Resolve output tokens: header > body > config
		outputTokens, err := resolveTokenCount(r, &cfg, req.MaxTokens)
		if err != nil {
			writeErrorResponse(w, &cfg, 400, "openai", "invalid_request_error", err.Error())
			return
		}

		// Tool echo: respond with the first tool the caller offered.
		toolName := firstRequestedToolName(req.Tools)
		toolInput := map[string]any{"input": "mock-input"}
		if toolName == "" {
			toolName = "get_weather"
			toolInput = map[string]any{"location": "San Francisco", "unit": "celsius"}
		}

		// Content may be a string or an array of content parts.
		totalChars := 0
		for _, m := range req.Messages {
			totalChars += m.contentChars()
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
			includeUsage := req.StreamOptions != nil && req.StreamOptions.IncludeUsage
			handleOpenAIChatStream(w, &cfg, id, model, words, promptTokens, outputTokens, toolName, toolInput, includeUsage)
		} else {
			handleOpenAIChatNonStream(w, &cfg, id, model, words, promptTokens, outputTokens, toolName, toolInput)
		}
	}
}

// openaiFinishReason resolves the finish_reason to emit: explicit config
// override first, then tool_calls for tool responses, then "stop".
func openaiFinishReason(cfg *ProviderConfig) string {
	if cfg.StopReason != "" {
		return cfg.StopReason
	}
	if cfg.ToolUseResponse {
		return "tool_calls"
	}
	return "stop"
}

// openaiUsage builds the chat usage object. The real API sends the
// prompt/completion *_details sub-objects unconditionally, not only when
// reasoning is in play.
func openaiUsage(cfg *ProviderConfig, promptTokens, completionTokens int) map[string]any {
	return map[string]any{
		"prompt_tokens":     promptTokens,
		"completion_tokens": completionTokens,
		"total_tokens":      promptTokens + completionTokens,
		"prompt_tokens_details": map[string]any{
			"cached_tokens": cfg.CacheReadTokens,
			"audio_tokens":  0,
		},
		"completion_tokens_details": map[string]any{
			"reasoning_tokens":           cfg.ReasoningTokens,
			"audio_tokens":               0,
			"accepted_prediction_tokens": 0,
			"rejected_prediction_tokens": 0,
		},
	}
}

// mockToolCalls builds an OpenAI tool_calls array echoing the requested tool.
func mockToolCalls(toolName string, toolInput map[string]any) []map[string]any {
	args, _ := json.Marshal(toolInput)
	return []map[string]any{
		{
			"index": 0,
			"id":    "call_mock_123",
			"type":  "function",
			"function": map[string]any{
				"name":      toolName,
				"arguments": string(args),
			},
		},
	}
}

func handleOpenAIChatNonStream(w http.ResponseWriter, cfg *ProviderConfig, id, model string, words []string, promptTokens, completionTokens int, toolName string, toolInput map[string]any) {
	if cfg.ReasoningTokens > 0 {
		completionTokens = len(words) + cfg.ReasoningTokens
	}

	content := joinContent(words)
	usage := openaiUsage(cfg, promptTokens, completionTokens)

	message := map[string]any{
		"role":        "assistant",
		"content":     content,
		"refusal":     nil,
		"annotations": []any{},
	}
	if cfg.ToolUseResponse {
		message["content"] = nil
		// Strip the streaming-only "index" key from the non-stream shape.
		calls := mockToolCalls(toolName, toolInput)
		delete(calls[0], "index")
		message["tool_calls"] = calls
	}

	resp := map[string]any{
		"id":                 id,
		"object":             "chat.completion",
		"created":            time.Now().Unix(),
		"model":              model,
		"system_fingerprint": "fp_mock",
		"service_tier":       "default",
		"choices": []map[string]any{
			{
				"index":         0,
				"message":       message,
				"logprobs":      nil,
				"finish_reason": openaiFinishReason(cfg),
			},
		},
		"usage": usage,
	}

	writeValidatedJSON(w, cfg, kindOpenAIChat, "openai chat completion", resp)
}

func handleOpenAIChatStream(w http.ResponseWriter, cfg *ProviderConfig, id, model string, words []string, promptTokens, completionTokens int, toolName string, toolInput map[string]any, includeUsage bool) {
	sse := newSSEWriter(w)
	validate := shouldValidate(cfg) && !bypassesValidation(cfg)

	// emitChunk validates each data payload against the pinned
	// CreateChatCompletionStreamResponse root when validate_responses is on
	// (a violation severs the stream), then writes the SSE frame. Returns
	// false when the stream must abort.
	emitChunk := func(chunk map[string]any) bool {
		data, _ := json.Marshal(chunk)
		if validate {
			if err := validateEmittedBody(kindOpenAIChunk, data); err != nil {
				failStreamValidation(w, "openai stream chunk", data, err)
				return false
			}
		}
		sse.writeData(string(data))
		return true
	}

	// Streaming usage shape (real API): usage appears only when the request
	// set stream_options.include_usage — "usage": null on every chunk, then
	// one trailing chunk with empty choices carrying the totals. The old
	// mock shape (usage riding the finish chunk unconditionally) stays
	// available behind the legacy_stream_usage compat flag.
	usageNullOnChunks := includeUsage && !cfg.LegacyStreamUsage

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
		"service_tier":       "default",
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
	if usageNullOnChunks {
		roleChunk["usage"] = nil
	}
	if !emitChunk(roleChunk) {
		return
	}

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
			"service_tier":       "default",
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
		if usageNullOnChunks {
			contentChunk["usage"] = nil
		}
		if !emitChunk(contentChunk) {
			return
		}
	}

	// Tool call delta: a single chunk carrying the full call (valid per the
	// OpenAI streaming spec; finer-grained argument deltas can come later).
	if cfg.ToolUseResponse {
		toolChunk := map[string]any{
			"id":                 id,
			"object":             "chat.completion.chunk",
			"created":            time.Now().Unix(),
			"model":              model,
			"system_fingerprint": "fp_mock",
			"service_tier":       "default",
			"choices": []map[string]any{
				{
					"index": 0,
					"delta": map[string]any{
						"tool_calls": mockToolCalls(toolName, toolInput),
					},
					"logprobs":      nil,
					"finish_reason": nil,
				},
			},
		}
		if usageNullOnChunks {
			toolChunk["usage"] = nil
		}
		if !emitChunk(toolChunk) {
			return
		}
	}

	// Adjust completion tokens for reasoning
	if cfg.ReasoningTokens > 0 {
		completionTokens = len(words) + cfg.ReasoningTokens
	}

	// Finish chunk
	finishChunk := map[string]any{
		"id":                 id,
		"object":             "chat.completion.chunk",
		"created":            time.Now().Unix(),
		"model":              model,
		"system_fingerprint": "fp_mock",
		"service_tier":       "default",
		"choices": []map[string]any{
			{
				"index":         0,
				"delta":         map[string]any{},
				"logprobs":      nil,
				"finish_reason": openaiFinishReason(cfg),
			},
		},
	}
	if cfg.LegacyStreamUsage {
		finishChunk["usage"] = openaiUsage(cfg, promptTokens, completionTokens)
	} else if includeUsage {
		finishChunk["usage"] = nil
	}
	if !emitChunk(finishChunk) {
		return
	}

	// Trailing usage chunk: empty choices carrying the totals.
	if usageNullOnChunks {
		usageChunk := map[string]any{
			"id":                 id,
			"object":             "chat.completion.chunk",
			"created":            time.Now().Unix(),
			"model":              model,
			"system_fingerprint": "fp_mock",
			"service_tier":       "default",
			"choices":            []any{},
			"usage":              openaiUsage(cfg, promptTokens, completionTokens),
		}
		if !emitChunk(usageChunk) {
			return
		}
	}

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
