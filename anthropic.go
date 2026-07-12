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

func handleAnthropicMessages(state *ServerState) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		cfg, limiter := state.Anthropic()

		// Validate required headers
		if r.Header.Get("x-api-key") == "" {
			writeErrorResponse(w, &cfg, 401, "anthropic", "authentication_error", "Missing x-api-key header")
			return
		}
		if r.Header.Get("anthropic-version") == "" {
			writeErrorResponse(w, &cfg, 401, "anthropic", "authentication_error", "Missing anthropic-version header")
			return
		}

		// Max concurrent check
		allowed, acquired := state.AcquireConcurrency("anthropic")
		if !allowed {
			writeErrorResponse(w, &cfg, 503, "anthropic", "overloaded_error", "Too many concurrent requests")
			return
		}
		if acquired {
			defer state.ReleaseConcurrency("anthropic")
		}

		// Capture raw body for request recording
		rawBody, err := io.ReadAll(r.Body)
		if err != nil {
			writeErrorResponse(w, &cfg, 400, "anthropic", "invalid_request_error", "Failed to read body: "+err.Error())
			return
		}

		// Record the request
		headers := make(map[string]string)
		for k := range r.Header {
			headers[k] = r.Header.Get(k)
		}
		state.RecordRequest(RecordedRequest{
			Timestamp: time.Now(),
			Provider:  "anthropic",
			Method:    r.Method,
			Path:      r.URL.Path,
			Proto:     r.Proto,
			Headers:   headers,
			Body:      json.RawMessage(rawBody),
		})

		var req struct {
			Model     string          `json:"model"`
			Stream    bool            `json:"stream"`
			MaxTokens int             `json:"max_tokens"`
			Messages  []chatMessage   `json:"messages"`
			Tools     json.RawMessage `json:"tools"`
			Thinking  *struct {
				Type         string `json:"type"`
				BudgetTokens int    `json:"budget_tokens"`
			} `json:"thinking"`
		}
		if err := json.NewDecoder(bytes.NewReader(rawBody)).Decode(&req); err != nil {
			writeErrorResponse(w, &cfg, 400, "anthropic", "invalid_request_error", "Invalid JSON: "+err.Error())
			return
		}

		// Scenario match — after body read + model decode (K1: the
		// provider-global concurrency gate and limiter fetch above already
		// ran; scenarios scope content+faults+capture only).
		sc, scStatus, scMsg := matchScenario(state.Scenarios(), r, "anthropic", "messages", req.Model)
		if scMsg != "" {
			writeErrorResponse(w, &cfg, scStatus, "anthropic", errorTypeForStatus(scStatus, "anthropic"), scMsg)
			return
		}
		var exact *ExactOutput
		if sc != nil {
			cfg = applyScenario(sc, r, rawBody, &cfg)
			exact = sc.Output
		}

		if rejectLeakedCacheControl(w, &cfg, "anthropic", rawBody) {
			return
		}
		if checkFaults(w, r, &cfg, limiter, state, "anthropic") {
			return
		}

		// Strict mode: the Anthropic request-shape checker (bounded) —
		// a manual-allowlist shape check, not a full request-schema
		// validator (after fault injection, like a real gateway).
		if cfg.Strict {
			if msg := validateAnthropicStrict(rawBody); msg != "" {
				writeErrorResponse(w, &cfg, 400, "anthropic", "invalid_request_error", msg)
				return
			}
		}

		if cfg.ThinkingDelayMs > 0 {
			if !waitCancelable(r.Context(), cfg.ThinkingDelayMs) {
				return
			}
		}

		// A request with thinking enabled gets a thinking block even when the
		// config doesn't force one (cfg is a per-request copy).
		if cfg.ReasoningTokens == 0 && req.Thinking != nil && req.Thinking.Type == "enabled" {
			cfg.ReasoningTokens = 45
		}

		// Tool echo: respond with the first tool the caller offered, falling
		// back to the legacy fixed tool when none was offered.
		toolName := firstRequestedToolName(req.Tools)
		toolInput := map[string]any{"input": "mock-input"}
		if toolName == "" {
			toolName = "get_weather"
			toolInput = map[string]any{"location": "San Francisco", "unit": "celsius"}
		}

		// Calculate mock token counts (content may be string or block array)
		totalChars := 0
		for _, m := range req.Messages {
			totalChars += m.contentChars()
		}
		inputTokens := totalChars / 4
		if inputTokens < 1 {
			inputTokens = 1
		}

		// Resolve output tokens: header > body > config
		outputTokens, err := resolveTokenCount(r, &cfg, req.MaxTokens)
		if err != nil {
			writeErrorResponse(w, &cfg, 400, "anthropic", "invalid_request_error", err.Error())
			return
		}

		words := generateWords(outputTokens)
		if cfg.Deterministic {
			words = generateDeterministicWords(outputTokens)
		}
		if cfg.ContentText != "" {
			words = strings.Fields(cfg.ContentText)
		}
		model := req.Model
		if model == "" {
			model = "claude-3-haiku-20240307"
		}
		id := fmt.Sprintf("msg_mock_%d", time.Now().UnixNano())
		if cfg.Deterministic {
			id = "msg_mock_deterministic"
		}

		// slow_header_ms delay
		if cfg.SlowHeaderMs > 0 {
			if !waitCancelable(r.Context(), cfg.SlowHeaderMs) {
				return
			}
		}

		if req.Stream {
			handleAnthropicStream(r.Context(), w, &cfg, id, model, words, inputTokens, outputTokens, toolName, toolInput, exact)
		} else {
			handleAnthropicNonStream(w, &cfg, id, model, words, inputTokens, outputTokens, toolName, toolInput, exact)
		}
	}
}

// anthropicUsage builds the usage object with the full field set the real
// API always sends: all 9 spec Usage.required keys, including the
// required-but-nullable inference_geo and output_tokens_details (emitted as
// null, their spec default). Cache fields are driven by config knobs. The
// emit_nonstandard_fields fault knob adds a genuinely-unknown extra key for
// unknown-field tolerance testing.
func anthropicUsage(cfg *ProviderConfig, inputTokens, outputTokens int) map[string]any {
	u := map[string]any{
		"input_tokens":                inputTokens,
		"output_tokens":               outputTokens,
		"cache_read_input_tokens":     cfg.CacheReadTokens,
		"cache_creation_input_tokens": cfg.CacheCreationTokens,
		"cache_creation": map[string]any{
			"ephemeral_5m_input_tokens": cfg.CacheCreationTokens,
			"ephemeral_1h_input_tokens": 0,
		},
		"server_tool_use":       nil,
		"service_tier":          "standard",
		"inference_geo":         nil,
		"output_tokens_details": nil,
	}
	if cfg.EmitNonstandardFields {
		u["x_mock_unknown_usage_field"] = 0
	}
	return u
}

// anthropicStopReason resolves the stop_reason to emit: explicit config
// override first, then tool_use for tool responses, then end_turn.
func anthropicStopReason(cfg *ProviderConfig) string {
	if cfg.StopReason != "" {
		return cfg.StopReason
	}
	if cfg.ToolUseResponse {
		return "tool_use"
	}
	return "end_turn"
}

func handleAnthropicNonStream(w http.ResponseWriter, cfg *ProviderConfig, id, model string, words []string, inputTokens, outputTokens int, toolName string, toolInput map[string]any, exact *ExactOutput) {
	// Tool use response mode: return text + tool_use content blocks echoing
	// the first requested tool.
	if cfg.ToolUseResponse {
		resp := map[string]any{
			"id":    id,
			"type":  "message",
			"role":  "assistant",
			"model": model,
			"content": []map[string]any{
				{"type": "text", "text": "I'll look that up for you.", "citations": nil},
				{"type": "tool_use", "id": "toolu_mock_123", "name": toolName,
					"input": toolInput, "caller": map[string]any{"type": "direct"}},
			},
			"stop_reason":   anthropicStopReason(cfg),
			"stop_sequence": nil,
			"stop_details":  nil,
			"container":     nil,
			"usage":         anthropicUsage(cfg, inputTokens, outputTokens),
		}
		if cfg.EmitNonstandardFields {
			resp["x_mock_unknown_field"] = "unknown-field-tolerance-probe"
		}
		writeValidatedJSON(w, cfg, kindAnthropicMessage, "anthropic tool-use message", resp)
		return
	}

	content := joinContent(words)
	if cfg.ContentText != "" {
		content = strings.Join(words, " ")
	}
	if exact != nil {
		// Exact output: Text verbatim, usage per the R9 rule.
		content = exact.Text
		outputTokens = exactOutputTokens(exact)
	}

	contentBlocks := []map[string]any{}

	if exact != nil && exact.Thinking != "" {
		// Exact thinking (K17 opt-in): the scenario owns the thinking block
		// verbatim; reasoning_tokens does not add a second one.
		contentBlocks = append(contentBlocks, map[string]any{
			"type":      "thinking",
			"thinking":  exact.Thinking,
			"signature": "sig_mock",
		})
	} else if cfg.ReasoningTokens > 0 {
		thinkingWordCount := cfg.ReasoningTokens / 3
		if thinkingWordCount < 5 {
			thinkingWordCount = 5
		}
		thinkingText := joinContent(generateWords(thinkingWordCount))
		contentBlocks = append(contentBlocks, map[string]any{
			"type":      "thinking",
			"thinking":  thinkingText,
			"signature": "sig_mock",
		})
	}

	contentBlocks = append(contentBlocks, map[string]any{
		"type":      "text",
		"text":      content,
		"citations": nil,
	})

	resp := map[string]any{
		"id":            id,
		"type":          "message",
		"role":          "assistant",
		"content":       contentBlocks,
		"model":         model,
		"stop_reason":   anthropicStopReason(cfg),
		"stop_sequence": nil,
		"stop_details":  nil,
		"container":     nil,
		"usage":         anthropicUsage(cfg, inputTokens, outputTokens),
	}
	if cfg.EmitNonstandardFields {
		resp["x_mock_unknown_field"] = "unknown-field-tolerance-probe"
	}

	writeValidatedJSON(w, cfg, kindAnthropicMessage, "anthropic message", resp)
}

func handleAnthropicStream(ctx context.Context, w http.ResponseWriter, cfg *ProviderConfig, id, model string, words []string, inputTokens, outputTokens int, toolName string, toolInput map[string]any, exact *ExactOutput) {
	if exact != nil {
		outputTokens = exactOutputTokens(exact)
	}
	sse := newSSEWriter(w)
	sse.applyTransportFaults(ctx, cfg)
	validate := shouldValidate(cfg) && !bypassesValidation(cfg)

	// writeFrame validates the SSE data payload against the pinned
	// MessageStreamEvent union (ping via the local arm) when
	// validate_responses is on — a violation severs the stream — then
	// writes the event and applies the disconnect_after_event fault: when
	// the just-written event type matches the knob, the connection is cut
	// (RST) and the handler must stop. Injected fault frames go through
	// here too, so deliberately off-vocabulary faults (unknown_event,
	// unknown_block, stream_error) need validate_responses:false.
	writeFrame := func(event, data string) bool {
		if validate {
			if err := validateEmittedBody(kindAnthropicEvent, []byte(data)); err != nil {
				failStreamValidation(w, "anthropic stream event "+event, []byte(data), err)
				return true
			}
		}
		sse.writeEvent(event, data)
		if cfg.DisconnectAfterEvent != "" && cfg.DisconnectAfterEvent == event {
			hijackAndClose(w)
			return true
		}
		return false
	}
	inj := newStreamFaultInjector(ctx, cfg.streamFaults, w, sse, writeFrame)

	// emit writes one real stream event, then gives the fault injector its
	// shot at the just-written frame. Returns true when the stream is over.
	emit := func(event, data string) bool {
		if writeFrame(event, data) {
			return true
		}
		return inj.afterFrame(event)
	}

	// TTFT delay
	if cfg.TtftMs > 0 {
		if !sleepWithPings(ctx, sse, cfg.TtftMs, cfg.SseKeepaliveIntervalMs) {
			return
		}
	}

	// message_start
	startMsg := map[string]any{
		"id":            id,
		"type":          "message",
		"role":          "assistant",
		"content":       []any{},
		"model":         model,
		"stop_reason":   nil,
		"stop_sequence": nil,
		"stop_details":  nil,
		"container":     nil,
		"usage":         anthropicUsage(cfg, inputTokens, 1),
	}
	if cfg.EmitNonstandardFields {
		startMsg["x_mock_unknown_field"] = "unknown-field-tolerance-probe"
	}
	msgStart, _ := json.Marshal(map[string]any{
		"type":    "message_start",
		"message": startMsg,
	})
	if emit("message_start", string(msgStart)) {
		return
	}

	// Typed ping event right after message_start, like the real API. Not a
	// MessageStreamEvent union member in the pinned spec, so it can be
	// suppressed for strict schema-validation harnesses.
	if !cfg.SuppressPingEvents {
		if emit("ping", `{"type":"ping"}`) {
			return
		}
	}

	textBlockIndex := 0

	// Thinking block: exact thinking (K17 opt-in — the scenario owns the
	// block verbatim, sliced by chunkExact) or generated from
	// reasoning_tokens.
	exactThinking := exact != nil && exact.Thinking != ""
	if exactThinking || cfg.ReasoningTokens > 0 {
		var thinkChunks []string
		if exactThinking {
			thinkChunks = chunkExact(exact.Thinking, exact.Chunking)
		} else {
			thinkingWordCount := cfg.ReasoningTokens / 3
			if thinkingWordCount < 5 {
				thinkingWordCount = 5
			}
			thinkingWords := generateWords(thinkingWordCount)
			for i := 0; i < len(thinkingWords); i += 5 {
				end := i + 5
				if end > len(thinkingWords) {
					end = len(thinkingWords)
				}
				chunk := strings.Join(thinkingWords[i:end], " ")
				if i+5 < len(thinkingWords) {
					chunk += " "
				}
				thinkChunks = append(thinkChunks, chunk)
			}
		}

		// content_block_start for thinking
		thinkStart, _ := json.Marshal(map[string]any{
			"type":  "content_block_start",
			"index": 0,
			"content_block": map[string]any{
				"type": "thinking", "thinking": "", "signature": "",
			},
		})
		if emit("content_block_start", string(thinkStart)) {
			return
		}

		for _, chunk := range thinkChunks {
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

			thinkDelta, _ := json.Marshal(map[string]any{
				"type":  "content_block_delta",
				"index": 0,
				"delta": map[string]any{
					"type":     "thinking_delta",
					"thinking": chunk,
				},
			})
			if emit("content_block_delta", string(thinkDelta)) {
				return
			}
		}

		// signature_delta closes the thinking block like the real API
		sigDelta, _ := json.Marshal(map[string]any{
			"type":  "content_block_delta",
			"index": 0,
			"delta": map[string]any{
				"type":      "signature_delta",
				"signature": "sig_mock",
			},
		})
		if emit("content_block_delta", string(sigDelta)) {
			return
		}

		// content_block_stop for thinking
		thinkStop, _ := json.Marshal(map[string]any{
			"type":  "content_block_stop",
			"index": 0,
		})
		if emit("content_block_stop", string(thinkStop)) {
			return
		}

		textBlockIndex = 1
	}

	// content_block_start for text
	blockStart, _ := json.Marshal(map[string]any{
		"type":          "content_block_start",
		"index":         textBlockIndex,
		"content_block": map[string]any{"type": "text", "text": "", "citations": nil},
	})
	if emit("content_block_start", string(blockStart)) {
		return
	}

	// content_block_delta: exact-output scenarios stream chunkExact slices
	// of Output.Text verbatim; generated words keep the historical
	// decoration.
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

		delta, _ := json.Marshal(map[string]any{
			"type":  "content_block_delta",
			"index": textBlockIndex,
			"delta": map[string]any{
				"type": "text_delta",
				"text": token,
			},
		})
		if emit("content_block_delta", string(delta)) {
			return
		}
	}

	// content_block_stop for text
	blockStop, _ := json.Marshal(map[string]any{
		"type":  "content_block_stop",
		"index": textBlockIndex,
	})
	if emit("content_block_stop", string(blockStop)) {
		return
	}

	// Tool use block: content_block_start + input_json_delta fragments +
	// content_block_stop, like the real API streams tool calls.
	if cfg.ToolUseResponse {
		toolBlockIndex := textBlockIndex + 1

		toolStart, _ := json.Marshal(map[string]any{
			"type":  "content_block_start",
			"index": toolBlockIndex,
			"content_block": map[string]any{
				"type":   "tool_use",
				"id":     "toolu_mock_123",
				"name":   toolName,
				"input":  map[string]any{},
				"caller": map[string]any{"type": "direct"},
			},
		})
		if emit("content_block_start", string(toolStart)) {
			return
		}

		inputJSON, _ := json.Marshal(toolInput)
		for _, fragment := range splitJSONFragments(string(inputJSON), 3) {
			toolDelta, _ := json.Marshal(map[string]any{
				"type":  "content_block_delta",
				"index": toolBlockIndex,
				"delta": map[string]any{
					"type":         "input_json_delta",
					"partial_json": fragment,
				},
			})
			if emit("content_block_delta", string(toolDelta)) {
				return
			}
		}

		toolStop, _ := json.Marshal(map[string]any{
			"type":  "content_block_stop",
			"index": toolBlockIndex,
		})
		if emit("content_block_stop", string(toolStop)) {
			return
		}
	}

	// message_delta — full MessageDelta + MessageDeltaUsage shapes per the
	// spec: delta carries container and the required-nullable stop_details;
	// usage carries the full MessageDeltaUsage.required set including the
	// required-nullable output_tokens_details.
	deltaObj := map[string]any{
		"stop_reason":   anthropicStopReason(cfg),
		"stop_sequence": nil,
		"stop_details":  nil,
		"container":     nil,
	}
	deltaUsage := map[string]any{
		"input_tokens":                inputTokens,
		"output_tokens":               outputTokens,
		"cache_read_input_tokens":     cfg.CacheReadTokens,
		"cache_creation_input_tokens": cfg.CacheCreationTokens,
		"server_tool_use":             nil,
		"output_tokens_details":       nil,
	}
	if cfg.EmitNonstandardFields {
		deltaObj["x_mock_unknown_field"] = "unknown-field-tolerance-probe"
		deltaUsage["x_mock_unknown_usage_field"] = 0
	}
	msgDelta, _ := json.Marshal(map[string]any{
		"type":  "message_delta",
		"delta": deltaObj,
		"usage": deltaUsage,
	})
	if emit("message_delta", string(msgDelta)) {
		return
	}

	// message_stop
	msgStop, _ := json.Marshal(map[string]any{
		"type": "message_stop",
	})
	emit("message_stop", string(msgStop))
	// Coalesced-frame mode (A3): the buffered tail goes out with the stream.
	sse.flushPending()
}

// splitJSONFragments splits a JSON string into n roughly equal pieces for
// streaming as input_json_delta partial_json fragments.
func splitJSONFragments(s string, n int) []string {
	if n < 1 || len(s) <= n {
		return []string{s}
	}
	size := (len(s) + n - 1) / n
	fragments := make([]string, 0, n)
	for start := 0; start < len(s); start += size {
		end := start + size
		if end > len(s) {
			end = len(s)
		}
		fragments = append(fragments, s[start:end])
	}
	return fragments
}
