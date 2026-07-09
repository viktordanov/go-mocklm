package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"time"
)

// handleOpenAIEmbeddings emulates the OpenAI embeddings API
// (POST /v1/embeddings). Records the raw request, honors fault injection,
// and returns one deterministic 8-dimension vector per input item.
func handleOpenAIEmbeddings(state *ServerState) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		cfg, limiter := state.OpenAI()

		// R4: scenarios are not honored on the embeddings surface — a
		// targeting header is a loud 400, not a silent no-op.
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
			Model string          `json:"model"`
			Input json.RawMessage `json:"input"`
		}
		if err := json.NewDecoder(bytes.NewReader(rawBody)).Decode(&req); err != nil {
			writeErrorResponse(w, &cfg, 400, "openai", "invalid_request_error", "Invalid JSON: "+err.Error())
			return
		}

		if checkFaults(w, r, &cfg, limiter, state, "openai") {
			return
		}

		// Input is a string or an array of strings; one vector per item.
		inputCount := 1
		var inputs []string
		if len(req.Input) > 0 {
			if err := json.Unmarshal(req.Input, &inputs); err == nil {
				inputCount = len(inputs)
			}
		}

		model := req.Model
		if model == "" {
			model = "text-embedding-3-small"
		}

		data := make([]map[string]any, 0, inputCount)
		for i := 0; i < inputCount; i++ {
			embedding := make([]float64, 8)
			for d := range embedding {
				embedding[d] = float64(i+1) * 0.1
			}
			data = append(data, map[string]any{
				"object":    "embedding",
				"embedding": embedding,
				"index":     i,
			})
		}

		resp := map[string]any{
			"object": "list",
			"data":   data,
			"model":  model,
			"usage": map[string]any{
				"prompt_tokens": inputCount * 4,
				"total_tokens":  inputCount * 4,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}
}
