package main

import (
	"encoding/json"
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

	// Dual-listener startup: the clear-text lane and the TLS/h2 observation
	// lane run side by side, each independently enable/disable-able via
	// MOCKLM_HTTP_*/MOCKLM_TLS_* env vars (see tls.go).
	spec, err := resolveListenerSpec(&cfg.Server)
	if err != nil {
		log.Fatalf("Listener config error: %v", err)
	}
	log.Printf("Config:\n%s", cfg.summary())

	errCh := make(chan error, 2)
	if spec.httpEnabled {
		log.Printf("mocklm HTTP listening on %s", spec.httpAddr)
		go func() {
			errCh <- http.ListenAndServe(spec.httpAddr, mux)
		}()
	}
	if spec.tlsEnabled {
		cert, err := loadOrGenerateCert(spec.certFile, spec.keyFile)
		if err != nil {
			log.Fatalf("TLS setup error: %v", err)
		}
		certSource := "in-memory self-signed"
		if spec.certFile != "" {
			certSource = spec.certFile
		}
		log.Printf("mocklm HTTPS (h2 via ALPN) listening on %s (cert: %s)", spec.tlsAddr, certSource)
		srv := newTLSServer(spec.tlsAddr, mux, cert)
		go func() {
			errCh <- srv.ListenAndServeTLS("", "")
		}()
	}
	log.Fatalf("Server error: %v", <-errCh)
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

	// Scenario registry. Scenarios survive POST /admin/reset;
	// DELETE /admin/scenarios is their lifecycle boundary.
	mux.HandleFunc("POST /admin/scenarios", handleAdminPostScenario(state))
	mux.HandleFunc("GET /admin/scenarios", handleAdminGetScenarios(state))
	mux.HandleFunc("GET /admin/scenarios/{id}", handleAdminGetScenario(state))
	mux.HandleFunc("DELETE /admin/scenarios/{id}", handleAdminDeleteScenario(state))
	mux.HandleFunc("DELETE /admin/scenarios", handleAdminClearScenarios(state))
	mux.HandleFunc("GET /admin/scenarios/{id}/last-request", handleAdminScenarioLastRequest(state))
	mux.HandleFunc("GET /admin/scenarios/{id}/request-count", handleAdminScenarioRequestCount(state))
	mux.HandleFunc("POST /admin/scenarios/{id}/request-count/reset", handleAdminScenarioResetRequestCount(state))
	mux.HandleFunc("GET /admin/scenarios/{id}/attempt-count", handleAdminScenarioAttemptCount(state))
	mux.HandleFunc("POST /admin/scenarios/{id}/attempt-count/reset", handleAdminScenarioResetAttemptCount(state))

	return mux
}
