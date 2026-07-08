package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"time"
)

func handleOpenAIResponses(state *ServerState) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		cfg, limiter := state.OpenAI()

		// Max concurrent check
		allowed, acquired := state.AcquireConcurrency("openai")
		if !allowed {
			writeErrorResponse(w, 503, "openai", "server_error", "Too many concurrent requests")
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
			Model           string `json:"model"`
			Stream          bool   `json:"stream"`
			MaxOutputTokens int    `json:"max_output_tokens"`
			Input           []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"input"`
		}
		if err := json.NewDecoder(bytes.NewReader(rawBody)).Decode(&req); err != nil {
			writeErrorResponse(w, 400, "openai", "invalid_request_error", "Invalid JSON: "+err.Error())
			return
		}

		if checkFaults(w, r, &cfg, limiter, state, "openai") {
			return
		}

		if cfg.ThinkingDelayMs > 0 {
			time.Sleep(time.Duration(cfg.ThinkingDelayMs) * time.Millisecond)
		}

		// Resolve output tokens: header > body > config
		outputTokens, err := resolveTokenCount(r, &cfg, req.MaxOutputTokens)
		if err != nil {
			writeErrorResponse(w, 400, "openai", "invalid_request_error", err.Error())
			return
		}

		totalChars := 0
		for _, m := range req.Input {
			totalChars += len(m.Content)
		}
		inputTokens := totalChars / 4
		if inputTokens < 1 {
			inputTokens = 1
		}
		completionTokens := outputTokens

		words := generateWords(outputTokens)
		model := req.Model
		if model == "" {
			model = "gpt-4"
		}

		// slow_header_ms delay
		if cfg.SlowHeaderMs > 0 {
			time.Sleep(time.Duration(cfg.SlowHeaderMs) * time.Millisecond)
		}

		if req.Stream {
			handleResponsesStream(w, &cfg, model, words, inputTokens, completionTokens)
		} else {
			handleResponsesNonStream(w, &cfg, model, words, inputTokens, completionTokens)
		}
	}
}

func handleResponsesNonStream(w http.ResponseWriter, cfg *ProviderConfig, model string, words []string, inputTokens, completionTokens int) {
	if cfg.ReasoningTokens > 0 {
		completionTokens = len(words) + cfg.ReasoningTokens
	}

	outputItems := buildOutputItems(cfg, words)

	resp := map[string]any{
		"id":         fmt.Sprintf("resp_mock_%d", time.Now().UnixNano()),
		"object":     "response",
		"created_at": time.Now().Unix(),
		"status":     "completed",
		"model":      model,
		"output":     outputItems,
		"usage": map[string]any{
			"input_tokens":  inputTokens,
			"output_tokens": completionTokens,
			"total_tokens":  inputTokens + completionTokens,
			"output_tokens_details": map[string]any{
				"reasoning_tokens": cfg.ReasoningTokens,
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func handleResponsesStream(w http.ResponseWriter, cfg *ProviderConfig, model string, words []string, inputTokens, completionTokens int) {
	if cfg.ReasoningTokens > 0 {
		completionTokens = len(words) + cfg.ReasoningTokens
	}

	sse := newSSEWriter(w)

	// TTFT delay
	if cfg.TtftMs > 0 {
		sleepWithPings(sse, cfg.TtftMs, cfg.SseKeepaliveIntervalMs)
	}

	respID := fmt.Sprintf("resp_mock_%d", time.Now().UnixNano())
	msgID := fmt.Sprintf("msg_mock_%d", time.Now().UnixNano())

	// response.created — in_progress with empty output
	createdResp := map[string]any{
		"id":         respID,
		"object":     "response",
		"created_at": time.Now().Unix(),
		"status":     "in_progress",
		"model":      model,
		"output":     []any{},
	}
	data, _ := json.Marshal(createdResp)
	sse.writeEvent("response.created", string(data))

	// response.output_item.added — message item
	msgItem := map[string]any{
		"type":    "message",
		"id":      msgID,
		"status":  "in_progress",
		"role":    "assistant",
		"content": []any{},
	}
	itemAdded := map[string]any{
		"type":         "response.output_item.added",
		"output_index": 0,
		"item":         msgItem,
	}
	data, _ = json.Marshal(itemAdded)
	sse.writeEvent("response.output_item.added", string(data))

	// response.content_part.added
	contentPart := map[string]any{
		"type": "output_text",
		"text": "",
	}
	partAdded := map[string]any{
		"type":          "response.content_part.added",
		"output_index":  0,
		"content_index": 0,
		"part":          contentPart,
	}
	data, _ = json.Marshal(partAdded)
	sse.writeEvent("response.content_part.added", string(data))

	// response.output_text.delta — word by word
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

		delta := map[string]any{
			"type":          "response.output_text.delta",
			"output_index":  0,
			"content_index": 0,
			"delta":         token,
		}
		data, _ = json.Marshal(delta)
		sse.writeEvent("response.output_text.delta", string(data))
	}

	// response.output_text.done
	fullText := joinContent(words)
	textDone := map[string]any{
		"type":          "response.output_text.done",
		"output_index":  0,
		"content_index": 0,
		"text":          fullText,
	}
	data, _ = json.Marshal(textDone)
	sse.writeEvent("response.output_text.done", string(data))

	// response.content_part.done
	partDone := map[string]any{
		"type":          "response.content_part.done",
		"output_index":  0,
		"content_index": 0,
		"part": map[string]any{
			"type": "output_text",
			"text": fullText,
		},
	}
	data, _ = json.Marshal(partDone)
	sse.writeEvent("response.content_part.done", string(data))

	// response.output_item.done
	doneItem := map[string]any{
		"type":         "response.output_item.done",
		"output_index": 0,
		"item": map[string]any{
			"type":   "message",
			"id":     msgID,
			"status": "completed",
			"role":   "assistant",
			"content": []map[string]any{
				{"type": "output_text", "text": fullText},
			},
		},
	}
	data, _ = json.Marshal(doneItem)
	sse.writeEvent("response.output_item.done", string(data))

	// response.completed — full response with usage
	outputItems := buildOutputItems(cfg, words)
	completedResp := map[string]any{
		"id":         respID,
		"object":     "response",
		"created_at": time.Now().Unix(),
		"status":     "completed",
		"model":      model,
		"output":     outputItems,
		"usage": map[string]any{
			"input_tokens":  inputTokens,
			"output_tokens": completionTokens,
			"total_tokens":  inputTokens + completionTokens,
			"output_tokens_details": map[string]any{
				"reasoning_tokens": cfg.ReasoningTokens,
			},
		},
	}
	data, _ = json.Marshal(completedResp)
	sse.writeEvent("response.completed", string(data))
}

func buildOutputItems(cfg *ProviderConfig, words []string) []map[string]any {
	outputItems := []map[string]any{}

	if cfg.ReasoningTokens > 0 {
		thinkingWords := generateWords(cfg.ReasoningTokens / 3)
		if len(thinkingWords) < 5 {
			thinkingWords = generateWords(5)
		}
		outputItems = append(outputItems, map[string]any{
			"type": "reasoning",
			"id":   fmt.Sprintf("rs_mock_%d", time.Now().UnixNano()),
			"summary": []map[string]any{
				{"type": "summary_text", "text": joinContent(thinkingWords)},
			},
		})
	}

	outputItems = append(outputItems, map[string]any{
		"type":   "message",
		"id":     fmt.Sprintf("msg_mock_%d", time.Now().UnixNano()),
		"status": "completed",
		"role":   "assistant",
		"content": []map[string]any{
			{"type": "output_text", "text": joinContent(words)},
		},
	})

	return outputItems
}
