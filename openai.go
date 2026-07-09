package main

import (
	"bytes"
	"context"
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

		// Scenario match — after body read + model decode (K1: the
		// provider-global concurrency gate and limiter fetch above already
		// ran; scenarios scope content+faults+capture only).
		sc, scStatus, scMsg := matchScenario(state.Scenarios(), r, "openai", "chat", req.Model)
		if scMsg != "" {
			writeErrorResponse(w, &cfg, scStatus, "openai", errorTypeForStatus(scStatus, "openai"), scMsg)
			return
		}
		var exact *ExactOutput
		if sc != nil {
			cfg = applyScenario(sc, r, rawBody, &cfg)
			exact = sc.Output
		}

		if checkFaults(w, r, &cfg, limiter, state, "openai") {
			return
		}

		if cfg.ThinkingDelayMs > 0 {
			if !waitCancelable(r.Context(), cfg.ThinkingDelayMs) {
				return
			}
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
		if cfg.ContentText != "" {
			words = strings.Fields(cfg.ContentText)
		}
		model := req.Model
		if model == "" {
			model = "gpt-4"
		}
		id := fmt.Sprintf("chatcmpl-mock-%d", time.Now().UnixNano())

		// slow_header_ms delay
		if cfg.SlowHeaderMs > 0 {
			if !waitCancelable(r.Context(), cfg.SlowHeaderMs) {
				return
			}
		}

		if req.Stream {
			includeUsage := req.StreamOptions != nil && req.StreamOptions.IncludeUsage
			handleOpenAIChatStream(r.Context(), w, &cfg, id, model, words, promptTokens, outputTokens, toolName, toolInput, includeUsage, exact)
		} else {
			handleOpenAIChatNonStream(w, &cfg, id, model, words, promptTokens, outputTokens, toolName, toolInput, exact)
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

// openaiUsageWithFault resolves the usage object under the usage_fault knob:
// D2 "partial" strips it down to prompt_tokens only (off-spec — needs
// validate_responses:false); everything else gets the full spec shape.
// Callers handle "omit" themselves (there is no object to build).
func openaiUsageWithFault(cfg *ProviderConfig, promptTokens, completionTokens int) map[string]any {
	if cfg.UsageFault == "partial" {
		return map[string]any{"prompt_tokens": promptTokens}
	}
	return openaiUsage(cfg, promptTokens, completionTokens)
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

func handleOpenAIChatNonStream(w http.ResponseWriter, cfg *ProviderConfig, id, model string, words []string, promptTokens, completionTokens int, toolName string, toolInput map[string]any, exact *ExactOutput) {
	if cfg.ReasoningTokens > 0 {
		completionTokens = len(words) + cfg.ReasoningTokens
	}

	content := joinContent(words)
	if cfg.ContentText != "" {
		content = strings.Join(words, " ")
	}
	if exact != nil {
		// Exact output: Text verbatim, usage per the R9 rule.
		content = exact.Text
		completionTokens = exactOutputTokens(exact)
	}

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
	}
	// D1 "omit" leaves usage out entirely — spec-valid, since the pinned
	// response root doesn't require it.
	if cfg.UsageFault != "omit" {
		resp["usage"] = openaiUsageWithFault(cfg, promptTokens, completionTokens)
	}

	writeValidatedJSON(w, cfg, kindOpenAIChat, "openai chat completion", resp)
}

func handleOpenAIChatStream(ctx context.Context, w http.ResponseWriter, cfg *ProviderConfig, id, model string, words []string, promptTokens, completionTokens int, toolName string, toolInput map[string]any, includeUsage bool, exact *ExactOutput) {
	if exact != nil {
		completionTokens = exactOutputTokens(exact)
	}
	sse := newSSEWriter(w)
	sse.applyTransportFaults(ctx, cfg)
	validate := shouldValidate(cfg) && !bypassesValidation(cfg)

	// Usage faults (D1-D3): "trailer" forces the real include_usage wire
	// shape even when the request didn't ask; "omit" suppresses usage
	// everywhere, even when it did.
	switch cfg.UsageFault {
	case "trailer":
		includeUsage = true
	case "omit":
		includeUsage = false
	}

	// writeFrame validates one data payload against the pinned
	// CreateChatCompletionStreamResponse root when validate_responses is on
	// (a violation severs the stream), then writes the SSE frame. OpenAI
	// frames are data-only, so the injector's event name is ignored.
	// Returns true when the stream must abort.
	writeFrame := func(_, data string) bool {
		if validate {
			if err := validateEmittedBody(kindOpenAIChunk, []byte(data)); err != nil {
				failStreamValidation(w, "openai stream chunk", []byte(data), err)
				return true
			}
		}
		sse.writeData(data)
		return false
	}
	inj := newStreamFaultInjector(ctx, cfg.streamFaults, w, sse, writeFrame)

	// emitChunk writes one real chunk, then gives the fault injector its
	// shot. Returns false when the stream must abort.
	emitChunk := func(chunk map[string]any) bool {
		data, _ := json.Marshal(chunk)
		if writeFrame("", string(data)) {
			return false
		}
		return !inj.afterFrame("")
	}

	// Streaming usage shape (real API): usage appears only when the request
	// set stream_options.include_usage — "usage": null on every chunk, then
	// one trailing chunk with empty choices carrying the totals. The old
	// mock shape (usage riding the finish chunk unconditionally) stays
	// available behind the legacy_stream_usage compat flag.
	usageNullOnChunks := includeUsage && !cfg.LegacyStreamUsage

	// TTFT delay
	if cfg.TtftMs > 0 {
		if !sleepWithPings(ctx, sse, cfg.TtftMs, cfg.SseKeepaliveIntervalMs) {
			return
		}
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

	// Content chunks: exact-output scenarios stream chunkExact slices of
	// Output.Text verbatim (the delta boundary a decoder reassembles);
	// generated words keep the historical decoration.
	deltas := streamDeltas(cfg, words, exact)
	for i, token := range deltas {
		if checkStreamingFault(w, cfg, i, len(deltas)) {
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
			if !sleepWithPings(ctx, sse, delay, cfg.SseKeepaliveIntervalMs) {
				return
			}
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

	// Adjust completion tokens for reasoning (exact output already pinned
	// its usage via the R9 rule)
	if cfg.ReasoningTokens > 0 && exact == nil {
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
	if cfg.LegacyStreamUsage && cfg.UsageFault != "omit" {
		finishChunk["usage"] = openaiUsageWithFault(cfg, promptTokens, completionTokens)
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
			"usage":              openaiUsageWithFault(cfg, promptTokens, completionTokens),
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

// sleepWithPings sleeps for totalMs, emitting SSE pings at keepaliveMs
// intervals. Cancel-aware: returns false as soon as the request context is
// done so fault-injected slow streams cannot outlive their client.
func sleepWithPings(ctx context.Context, sse *sseWriter, totalMs, keepaliveMs int) bool {
	if keepaliveMs <= 0 || totalMs <= keepaliveMs {
		return waitCancelable(ctx, totalMs)
	}
	remaining := totalMs
	for remaining > 0 {
		chunk := keepaliveMs
		if chunk > remaining {
			chunk = remaining
		}
		if !waitCancelable(ctx, chunk) {
			return false
		}
		remaining -= chunk
		if remaining > 0 {
			sse.writePing()
		}
	}
	return true
}
