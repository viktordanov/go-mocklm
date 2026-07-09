package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"math/rand"
	"net/http"
	"strings"
	"time"
)

// AWS Bedrock Runtime Converse / ConverseStream mock (Phase 3c) — the third
// provider:
//
//	POST /model/{modelId}/converse          -> JSON response
//	POST /model/{modelId}/converse-stream   -> application/vnd.amazon.eventstream
//
// Model routing (D2): plain model IDs only. Go's {modelId} wildcard matches
// a single slash-free path segment — colons are fine
// (anthropic.claude-3-5-sonnet-20240620-v1:0 routes), but ARNs and
// cross-region inference-profile IDs contain "/" (or %2F) and do NOT route;
// they would need a custom RequestURI parser. Documented limitation.
//
// Validation honesty (§3.6): nanollm's pinned specs are the OpenAI and
// Anthropic OpenAPI documents — there is NO Bedrock schema in the spec-sync
// closure, so validate_responses cannot cover these bodies. Bedrock
// fidelity is hand-rolled and bounded like strict.go; the SDK-decode tests
// (bedrock_test.go) are its tripwire instead. Auth is accept-any: SigV4
// signatures are ignored like every other provider's credentials.
//
// v1 scope: text responses (+ exact-output scenarios). No toolUse blocks,
// no reasoningContent — ExactOutput.Thinking is an Anthropic-surface knob
// and is not emitted here.

// bedrockStopReason resolves the stopReason to emit: explicit config
// override first, then end_turn.
func bedrockStopReason(cfg *ProviderConfig) string {
	if cfg.StopReason != "" {
		return cfg.StopReason
	}
	return "end_turn"
}

func handleBedrockConverse(state *ServerState, stream bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		cfg, limiter := state.Bedrock()

		// Max concurrent check (provider-global, before body read — K1)
		allowed, acquired := state.AcquireConcurrency("bedrock")
		if !allowed {
			writeErrorResponse(w, &cfg, 503, "bedrock", "ServiceUnavailableException", "Too many concurrent requests")
			return
		}
		if acquired {
			defer state.ReleaseConcurrency("bedrock")
		}

		// Buffer body for recording
		rawBody, err := io.ReadAll(r.Body)
		if err != nil {
			writeErrorResponse(w, &cfg, 400, "bedrock", "ValidationException", "Failed to read body: "+err.Error())
			return
		}

		// Record request
		headers := make(map[string]string)
		for k := range r.Header {
			headers[k] = r.Header.Get(k)
		}
		state.RecordRequest(RecordedRequest{
			Timestamp: time.Now(),
			Provider:  "bedrock",
			Method:    r.Method,
			Path:      r.URL.Path,
			Headers:   headers,
			Body:      json.RawMessage(rawBody),
		})

		// The model comes from the path, not the body (plain IDs only, D2).
		modelID := r.PathValue("modelId")

		var req struct {
			Messages []struct {
				Role    string `json:"role"`
				Content []struct {
					Text string `json:"text"`
				} `json:"content"`
			} `json:"messages"`
			System []struct {
				Text string `json:"text"`
			} `json:"system"`
			InferenceConfig *struct {
				MaxTokens int `json:"maxTokens"`
			} `json:"inferenceConfig"`
		}
		if err := json.NewDecoder(bytes.NewReader(rawBody)).Decode(&req); err != nil {
			writeErrorResponse(w, &cfg, 400, "bedrock", "ValidationException", "Invalid JSON: "+err.Error())
			return
		}

		// Scenario match — after body read + model decode (K1). Both
		// converse and converse-stream are the "converse" surface.
		sc, scStatus, scMsg := matchScenario(state.Scenarios(), r, "bedrock", "converse", modelID)
		if scMsg != "" {
			writeErrorResponse(w, &cfg, scStatus, "bedrock", errorTypeForStatus(scStatus, "bedrock"), scMsg)
			return
		}
		var exact *ExactOutput
		if sc != nil {
			cfg = applyScenario(sc, r, rawBody, &cfg)
			exact = sc.Output
		}

		if checkFaults(w, r, &cfg, limiter, state, "bedrock") {
			return
		}

		if cfg.ThinkingDelayMs > 0 {
			if !waitCancelable(r.Context(), cfg.ThinkingDelayMs) {
				return
			}
		}

		// Resolve output tokens: header > body inferenceConfig.maxTokens >
		// config
		bodyMax := 0
		if req.InferenceConfig != nil {
			bodyMax = req.InferenceConfig.MaxTokens
		}
		outputTokens, err := resolveTokenCount(r, &cfg, bodyMax)
		if err != nil {
			writeErrorResponse(w, &cfg, 400, "bedrock", "ValidationException", err.Error())
			return
		}

		// Mock input-token estimate from message + system text lengths.
		totalChars := 0
		for _, m := range req.Messages {
			for _, c := range m.Content {
				totalChars += len(c.Text)
			}
		}
		for _, s := range req.System {
			totalChars += len(s.Text)
		}
		inputTokens := totalChars / 4
		if inputTokens < 1 {
			inputTokens = 1
		}

		words := generateWords(outputTokens)
		if cfg.Deterministic {
			words = generateDeterministicWords(outputTokens)
		}
		if cfg.ContentText != "" {
			words = strings.Fields(cfg.ContentText)
		}

		// slow_header_ms delay
		if cfg.SlowHeaderMs > 0 {
			if !waitCancelable(r.Context(), cfg.SlowHeaderMs) {
				return
			}
		}

		if stream {
			handleBedrockConverseStream(r.Context(), w, &cfg, words, inputTokens, outputTokens, exact, start)
		} else {
			handleBedrockConverseNonStream(w, &cfg, words, inputTokens, outputTokens, exact, start)
		}
	}
}

// latencyMsSince derives the metrics.latencyMs value: real elapsed handler
// time, floored to 1 like the live service reports.
func latencyMsSince(start time.Time) int64 {
	ms := time.Since(start).Milliseconds()
	if ms < 1 {
		ms = 1
	}
	return ms
}

// bedrockUsage builds the usage object with the member casing the SDK
// decodes (inputTokens/outputTokens/totalTokens).
func bedrockUsage(inputTokens, outputTokens int) map[string]any {
	return map[string]any{
		"inputTokens":  inputTokens,
		"outputTokens": outputTokens,
		"totalTokens":  inputTokens + outputTokens,
	}
}

func handleBedrockConverseNonStream(w http.ResponseWriter, cfg *ProviderConfig, words []string, inputTokens, outputTokens int, exact *ExactOutput, start time.Time) {
	content := joinContent(words)
	if cfg.ContentText != "" {
		content = strings.Join(words, " ")
	}
	if exact != nil {
		// Exact output: Text verbatim, usage per the R9 rule.
		content = exact.Text
		outputTokens = exactOutputTokens(exact)
	}

	resp := map[string]any{
		"output": map[string]any{
			"message": map[string]any{
				"role": "assistant",
				"content": []map[string]any{
					{"text": content},
				},
			},
		},
		"stopReason": bedrockStopReason(cfg),
		"usage":      bedrockUsage(inputTokens, outputTokens),
		"metrics": map[string]any{
			"latencyMs": latencyMsSince(start),
		},
	}

	// No validate_responses here: no Bedrock schema exists in the spec-sync
	// closure (§3.6) — the SDK-decode tests are this surface's tripwire.
	data, err := marshalBody(resp)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(data)
}

// checkBedrockStreamingFault mirrors checkStreamingFault for the
// eventstream dialect: the SSE-based helper would write SSE bytes into a
// binary stream, so the legacy malformed_chunk knob corrupts a frame CRC
// here instead.
func checkBedrockStreamingFault(w http.ResponseWriter, esw *eventStreamWriter, cfg *ProviderConfig, chunkIndex, totalChunks int) bool {
	if cfg.DisconnectAfterChunks > 0 && chunkIndex >= cfg.DisconnectAfterChunks {
		hijackAndClose(w)
		return true
	}
	if cfg.MalformedChunk && chunkIndex == totalChunks/2 {
		esw.writeCorrupt()
	}
	return false
}

func handleBedrockConverseStream(ctx context.Context, w http.ResponseWriter, cfg *ProviderConfig, words []string, inputTokens, outputTokens int, exact *ExactOutput, start time.Time) {
	if exact != nil {
		outputTokens = exactOutputTokens(exact)
	}
	esw := newEventStreamWriter(w)

	// The 4th stream emit path (K15): exact chunks ride
	// contentBlockDelta.delta.text inside eventstream frames. No
	// validate_responses (no pinned Bedrock schema, §3.6) and no SSE
	// transport faults (different framing); disconnect / stall / after_* /
	// malformed_chunk and the Bedrock-dialect decoder probes come from the
	// eventstream fault injector.
	writeFrame := func(event, data string) bool {
		esw.writeEvent(event, []byte(data))
		if cfg.DisconnectAfterEvent != "" && cfg.DisconnectAfterEvent == event {
			hijackAndClose(w)
			return true
		}
		return false
	}
	inj := newEventStreamFaultInjector(ctx, cfg.streamFaults, w, esw, writeFrame)

	// emit writes one real event, then gives the fault injector its shot.
	// Returns true when the stream is over.
	emit := func(event string, payload map[string]any) bool {
		data, _ := json.Marshal(payload)
		if writeFrame(event, string(data)) {
			return true
		}
		return inj.afterFrame(event)
	}

	// TTFT delay (no SSE ping concept on eventstream — plain wait)
	if cfg.TtftMs > 0 {
		if !waitCancelable(ctx, cfg.TtftMs) {
			return
		}
	}

	// messageStart
	if emit("messageStart", map[string]any{"role": "assistant"}) {
		return
	}

	// contentBlockDelta per chunk (the real service emits no
	// contentBlockStart for text blocks)
	deltas := streamDeltas(cfg, words, exact)
	for i, token := range deltas {
		if checkBedrockStreamingFault(w, esw, cfg, i, len(deltas)) {
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
			if !waitCancelable(ctx, delay) {
				return
			}
		}

		if emit("contentBlockDelta", map[string]any{
			"contentBlockIndex": 0,
			"delta":             map[string]any{"text": token},
		}) {
			return
		}
	}

	// contentBlockStop
	if emit("contentBlockStop", map[string]any{"contentBlockIndex": 0}) {
		return
	}

	// messageStop
	if emit("messageStop", map[string]any{"stopReason": bedrockStopReason(cfg)}) {
		return
	}

	// metadata (usage + metrics)
	emit("metadata", map[string]any{
		"usage": bedrockUsage(inputTokens, outputTokens),
		"metrics": map[string]any{
			"latencyMs": latencyMsSince(start),
		},
	})
}
