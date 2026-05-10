package main

import (
	"encoding/json"
	"net/http"
)

// handleAdminGetConfig returns the current config and active preset.
func handleAdminGetConfig(state *ServerState) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		cfg, preset := state.Config()

		resp := map[string]any{
			"active_preset": preset,
			"openai":        cfg.OpenAI,
			"anthropic":     cfg.Anthropic,
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}
}

// handleAdminPutConfig replaces both provider configs from the request body.
func handleAdminPutConfig(state *ServerState) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			OpenAI    ProviderConfig `json:"openai"`
			Anthropic ProviderConfig `json:"anthropic"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			writeErrorResponse(w, 400, "admin", "invalid_request_error", "Invalid JSON: "+err.Error())
			return
		}

		applyProviderDefaults(&body.OpenAI)
		applyProviderDefaults(&body.Anthropic)
		state.Update(body.OpenAI, body.Anthropic, "custom")

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"status":        "updated",
			"active_preset": "custom",
		})
	}
}

// handleAdminPutPreset activates a named preset.
func handleAdminPutPreset(state *ServerState) http.HandlerFunc {
	presets := builtinPresets()

	return func(w http.ResponseWriter, r *http.Request) {
		name := r.PathValue("name")

		preset, ok := presets[name]
		if !ok {
			writeErrorResponse(w, 404, "admin", "not_found", "Unknown preset: "+name)
			return
		}

		openai := preset.OpenAI
		anthropic := preset.Anthropic
		applyProviderDefaults(&openai)
		applyProviderDefaults(&anthropic)
		state.Update(openai, anthropic, name)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"status":        "activated",
			"active_preset": name,
			"description":   preset.Description,
		})
	}
}

// handleAdminReset reverts to the startup config.
func handleAdminReset(state *ServerState) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		state.Reset()

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"status":        "reset",
			"active_preset": "",
		})
	}
}

// handleAdminGetRequests returns all recorded requests.
func handleAdminGetRequests(state *ServerState) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		records := state.Requests()

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"requests": records,
		})
	}
}

// handleAdminClearRequests clears all recorded requests.
func handleAdminClearRequests(state *ServerState) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		state.ClearRequests()

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"status": "cleared",
		})
	}
}

// handleAdminGetPresets lists all available presets.
func handleAdminGetPresets() http.HandlerFunc {
	presets := builtinPresets()

	type presetInfo struct {
		Name        string `json:"name"`
		Description string `json:"description"`
	}

	// Build a stable list sorted by name.
	list := make([]presetInfo, 0, len(presets))
	for _, p := range presets {
		list = append(list, presetInfo{Name: p.Name, Description: p.Description})
	}

	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"presets": list,
		})
	}
}
