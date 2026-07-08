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

	return mux
}
