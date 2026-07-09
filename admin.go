package main

import (
	"encoding/json"
	"net/http"
	"sort"
)

// handleAdminGetConfig returns the current config and active preset.
func handleAdminGetConfig(state *ServerState) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		cfg, preset := state.Config()

		resp := map[string]any{
			"active_preset": preset,
			"openai":        cfg.OpenAI,
			"anthropic":     cfg.Anthropic,
			"bedrock":       cfg.Bedrock,
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
			Bedrock   ProviderConfig `json:"bedrock"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			writeErrorResponse(w, nil, 400, "admin", "invalid_request_error", "Invalid JSON: "+err.Error())
			return
		}

		applyProviderDefaults(&body.OpenAI)
		applyProviderDefaults(&body.Anthropic)
		applyProviderDefaults(&body.Bedrock)
		state.Update(body.OpenAI, body.Anthropic, body.Bedrock, "custom")

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
			writeErrorResponse(w, nil, 404, "admin", "not_found", "Unknown preset: "+name)
			return
		}

		openai := preset.OpenAI
		anthropic := preset.Anthropic
		bedrock := preset.Bedrock
		applyProviderDefaults(&openai)
		applyProviderDefaults(&anthropic)
		applyProviderDefaults(&bedrock)
		state.Update(openai, anthropic, bedrock, name)

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

// handleAdminGetRequestCount returns the per-provider request counts since
// the last reset/config update — the attempt-counter oracle: a test can
// assert exactly how many attempts the proxy made (retries, fallbacks)
// without parsing recorded bodies. The same counter indexes fail_first_n
// and attempt_faults.
func handleAdminGetRequestCount(state *ServerState) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		openai, anthropic, bedrock := state.AttemptCounts()

		// The bedrock key is ADDITIVE to the historical {openai, anthropic}
		// response shape (K8): consumers ignoring unknown keys are
		// unaffected; a consumer asserting the exact two-key set must relax
		// that assertion.
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"openai":    openai,
			"anthropic": anthropic,
			"bedrock":   bedrock,
		})
	}
}

// handleAdminResetRequestCount zeroes the per-provider request counters
// (they also reset on every config update/reset), so one scenario's
// attempts never leak into the next assertion.
func handleAdminResetRequestCount(state *ServerState) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		state.ResetAttempts()

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"status": "reset",
		})
	}
}

// handleAdminGetFaults returns the machine-readable fault-mode catalog:
// per mode, its phase, WHEN knobs, params, whether it needs
// validate_responses:false, and its Bedrock dialect where it differs.
func handleAdminGetFaults() http.HandlerFunc {
	catalog := faultCatalog()

	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"faults": catalog,
		})
	}
}

// handleAdminGetFaultPresets lists the named fault presets — ProviderConfig
// fragments carrying only fault-surface knobs, registerable straight into a
// scenario's config (POST /admin/scenarios {"fault_preset": name}).
func handleAdminGetFaultPresets() http.HandlerFunc {
	presets := builtinFaultPresets()

	list := make([]FaultPreset, 0, len(presets))
	for _, p := range presets {
		list = append(list, p)
	}
	sort.Slice(list, func(i, j int) bool { return list[i].Name < list[j].Name })

	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"fault_presets": list,
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
