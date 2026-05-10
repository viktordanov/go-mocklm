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

func handleAnthropicMessages(state *ServerState) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Validate required headers
		if r.Header.Get("x-api-key") == "" {
			writeErrorResponse(w, 401, "anthropic", "authentication_error", "Missing x-api-key header")
			return
		}
		if r.Header.Get("anthropic-version") == "" {
			writeErrorResponse(w, 401, "anthropic", "authentication_error", "Missing anthropic-version header")
			return
		}

		cfg, limiter := state.Anthropic()

		// Max concurrent check
		allowed, acquired := state.AcquireConcurrency("anthropic")
		if !allowed {
			writeErrorResponse(w, 503, "anthropic", "overloaded", "Too many concurrent requests")
			return
		}
		if acquired {
			defer state.ReleaseConcurrency("anthropic")
		}

		// Capture raw body for request recording
		rawBody, err := io.ReadAll(r.Body)
		if err != nil {
			writeErrorResponse(w, 400, "anthropic", "invalid_request_error", "Failed to read body: "+err.Error())
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
			writeErrorResponse(w, 400, "anthropic", "invalid_request_error", "Invalid JSON: "+err.Error())
			return
		}

		if checkFaults(w, r, &cfg, limiter, "anthropic") {
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
		inputTokens := totalChars / 4
		if inputTokens < 1 {
			inputTokens = 1
		}

		// Resolve output tokens: header > body > config
		outputTokens, err := resolveTokenCount(r, &cfg, req.MaxTokens)
		if err != nil {
			writeErrorResponse(w, 400, "anthropic", "invalid_request_error", err.Error())
			return
		}

		words := generateWords(outputTokens)
		if cfg.Deterministic {
			words = generateDeterministicWords(outputTokens)
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
			time.Sleep(time.Duration(cfg.SlowHeaderMs) * time.Millisecond)
		}

		if req.Stream {
			handleAnthropicStream(w, &cfg, id, model, words, inputTokens, outputTokens)
		} else {
			handleAnthropicNonStream(w, &cfg, id, model, words, inputTokens, outputTokens)
		}
	}
}

func handleAnthropicNonStream(w http.ResponseWriter, cfg *ProviderConfig, id, model string, words []string, inputTokens, outputTokens int) {
	// Tool use response mode: return text + tool_use content blocks
	if cfg.ToolUseResponse {
		resp := map[string]any{
			"id":    id,
			"type":  "message",
			"role":  "assistant",
			"model": model,
			"content": []map[string]any{
				{"type": "text", "text": "I'll look that up for you."},
				{"type": "tool_use", "id": "toolu_mock_123", "name": "get_weather", "input": map[string]any{
					"location": "San Francisco",
					"unit":     "celsius",
				}},
			},
			"stop_reason":   "tool_use",
			"stop_sequence": nil,
			"usage": map[string]any{
				"input_tokens":  inputTokens,
				"output_tokens": outputTokens,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
		return
	}

	content := joinContent(words)

	contentBlocks := []map[string]any{}

	if cfg.ReasoningTokens > 0 {
		thinkingWordCount := cfg.ReasoningTokens / 3
		if thinkingWordCount < 5 {
			thinkingWordCount = 5
		}
		thinkingText := joinContent(generateWords(thinkingWordCount))
		contentBlocks = append(contentBlocks, map[string]any{
			"type":     "thinking",
			"thinking": thinkingText,
		})
	}

	contentBlocks = append(contentBlocks, map[string]any{
		"type": "text",
		"text": content,
	})

	resp := map[string]any{
		"id":            id,
		"type":          "message",
		"role":          "assistant",
		"content":       contentBlocks,
		"model":         model,
		"stop_reason":   "end_turn",
		"stop_sequence": nil,
		"usage": map[string]any{
			"input_tokens":  inputTokens,
			"output_tokens": outputTokens,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func handleAnthropicStream(w http.ResponseWriter, cfg *ProviderConfig, id, model string, words []string, inputTokens, outputTokens int) {
	sse := newSSEWriter(w)

	// TTFT delay
	if cfg.TtftMs > 0 {
		sleepWithPings(sse, cfg.TtftMs, cfg.SseKeepaliveIntervalMs)
	}

	// message_start
	msgStart, _ := json.Marshal(map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":            id,
			"type":          "message",
			"role":          "assistant",
			"content":       []any{},
			"model":         model,
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage": map[string]any{
				"input_tokens":  inputTokens,
				"output_tokens": 1,
			},
		},
	})
	sse.writeEvent("message_start", string(msgStart))

	textBlockIndex := 0

	// Thinking block (if reasoning tokens configured)
	if cfg.ReasoningTokens > 0 {
		thinkingWordCount := cfg.ReasoningTokens / 3
		if thinkingWordCount < 5 {
			thinkingWordCount = 5
		}
		thinkingWords := generateWords(thinkingWordCount)

		// content_block_start for thinking
		thinkStart, _ := json.Marshal(map[string]any{
			"type":          "content_block_start",
			"index":         0,
			"content_block": map[string]any{"type": "thinking", "thinking": ""},
		})
		sse.writeEvent("content_block_start", string(thinkStart))

		// Stream thinking in ~5 word chunks
		for i := 0; i < len(thinkingWords); i += 5 {
			end := i + 5
			if end > len(thinkingWords) {
				end = len(thinkingWords)
			}
			chunk := strings.Join(thinkingWords[i:end], " ")
			if i+5 < len(thinkingWords) {
				chunk += " "
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

			thinkDelta, _ := json.Marshal(map[string]any{
				"type":  "content_block_delta",
				"index": 0,
				"delta": map[string]any{
					"type":     "thinking_delta",
					"thinking": chunk,
				},
			})
			sse.writeEvent("content_block_delta", string(thinkDelta))
		}

		// content_block_stop for thinking
		thinkStop, _ := json.Marshal(map[string]any{
			"type":  "content_block_stop",
			"index": 0,
		})
		sse.writeEvent("content_block_stop", string(thinkStop))

		textBlockIndex = 1
	}

	// content_block_start for text
	blockStart, _ := json.Marshal(map[string]any{
		"type":          "content_block_start",
		"index":         textBlockIndex,
		"content_block": map[string]any{"type": "text", "text": ""},
	})
	sse.writeEvent("content_block_start", string(blockStart))

	// content_block_delta (word by word)
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
			token = capitalize(token)
		}
		if i == len(words)-1 {
			token += "."
		} else {
			token += " "
		}

		delta, _ := json.Marshal(map[string]any{
			"type":  "content_block_delta",
			"index": textBlockIndex,
			"delta": map[string]any{
				"type": "text_delta",
				"text": token,
			},
		})
		sse.writeEvent("content_block_delta", string(delta))
	}

	// content_block_stop for text
	blockStop, _ := json.Marshal(map[string]any{
		"type":  "content_block_stop",
		"index": textBlockIndex,
	})
	sse.writeEvent("content_block_stop", string(blockStop))

	// message_delta
	msgDelta, _ := json.Marshal(map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason":   "end_turn",
			"stop_sequence": nil,
		},
		"usage": map[string]any{
			"output_tokens": outputTokens,
		},
	})
	sse.writeEvent("message_delta", string(msgDelta))

	// message_stop
	msgStop, _ := json.Marshal(map[string]any{
		"type": "message_stop",
	})
	sse.writeEvent("message_stop", string(msgStop))
}
