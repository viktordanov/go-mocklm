package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
)

func main() {
	cfg, err := loadConfig()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	state := NewServerState(cfg)
	mux := buildMux(state)

	addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
	log.Printf("mocklm starting on %s", addr)
	log.Printf("Config:\n%s", cfg.summary())

	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}

// buildMux wires all routes including admin endpoints.
func buildMux(state *ServerState) *http.ServeMux {
	mux := http.NewServeMux()

	// Provider endpoints
	mux.HandleFunc("POST /v1/chat/completions", handleOpenAIChat(state))
	mux.HandleFunc("POST /v1/completions", handleOpenAICompletions(state))
	mux.HandleFunc("POST /v1/embeddings", handleOpenAIEmbeddings(state))
	mux.HandleFunc("GET /v1/models", handleOpenAIModels())
	mux.HandleFunc("POST /v1/messages", handleAnthropicMessages(state))
	mux.HandleFunc("POST /v1/responses", handleOpenAIResponses(state))

	// Bedrock Runtime Converse/ConverseStream. Plain model IDs only (D2):
	// {modelId} matches one slash-free segment, so ARNs / inference-profile
	// IDs containing "/" do not route (documented in bedrock.go).
	mux.HandleFunc("POST /model/{modelId}/converse", handleBedrockConverse(state, false))
	mux.HandleFunc("POST /model/{modelId}/converse-stream", handleBedrockConverse(state, true))

	// Health check
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	// Admin endpoints
	mux.HandleFunc("GET /admin/config", handleAdminGetConfig(state))
	mux.HandleFunc("PUT /admin/config", handleAdminPutConfig(state))
	mux.HandleFunc("PUT /admin/preset/{name}", handleAdminPutPreset(state))
	mux.HandleFunc("POST /admin/reset", handleAdminReset(state))
	mux.HandleFunc("GET /admin/presets", handleAdminGetPresets())
	mux.HandleFunc("GET /admin/requests", handleAdminGetRequests(state))
	mux.HandleFunc("POST /admin/requests/clear", handleAdminClearRequests(state))
	mux.HandleFunc("GET /admin/request-count", handleAdminGetRequestCount(state))
	mux.HandleFunc("POST /admin/request-count/reset", handleAdminResetRequestCount(state))
	mux.HandleFunc("GET /admin/faults", handleAdminGetFaults())
	mux.HandleFunc("GET /admin/fault-presets", handleAdminGetFaultPresets())

	// Scenario registry (Phase 3a). Scenarios survive POST /admin/reset;
	// DELETE /admin/scenarios is their lifecycle boundary.
	mux.HandleFunc("POST /admin/scenarios", handleAdminPostScenario(state))
	mux.HandleFunc("GET /admin/scenarios", handleAdminGetScenarios(state))
	mux.HandleFunc("GET /admin/scenarios/{id}", handleAdminGetScenario(state))
	mux.HandleFunc("DELETE /admin/scenarios/{id}", handleAdminDeleteScenario(state))
	mux.HandleFunc("DELETE /admin/scenarios", handleAdminClearScenarios(state))
	mux.HandleFunc("GET /admin/scenarios/{id}/last-request", handleAdminScenarioLastRequest(state))
	mux.HandleFunc("GET /admin/scenarios/{id}/request-count", handleAdminScenarioRequestCount(state))
	mux.HandleFunc("POST /admin/scenarios/{id}/request-count/reset", handleAdminScenarioResetRequestCount(state))

	return mux
}
