package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Scenario registry: a scenario is a named exact-output spec +
// fault set + its own capture/counter slot, matched per request by the
// X-MockLM-Scenario header or by (provider, model).
//
// Scoping (K1/D1, deliberate): a scenario scopes CONTENT + FAULTS +
// CAPTURE + its own FAULT-ATTEMPT SEQUENCE. A matched scenario's
// fail_first_n / attempt_faults index the scenario's own counter
// (faultAttempts, exposed at /admin/scenarios/{id}/attempt-count), so two
// scenarios on one provider — and any non-scenario background traffic —
// have fully independent fault sequences under parallel load. Only
// rate_limit_rpm and max_concurrent stay PROVIDER-GLOBAL: two scenarios
// sharing one provider share the provider's concurrency gate and RPM
// limiter, and registering a scenario whose config sets either is REJECTED
// with 400 (R5) rather than silently ignored. The provider request counter
// (/admin/request-count) still counts scenario traffic — it is an
// observability oracle, not the fault index.
//
// Lifecycle (R6, deliberate): scenarios are test fixtures with independent
// lifetimes — POST /admin/reset does NOT wipe them (use DELETE
// /admin/scenarios) and does NOT touch their fault-attempt counters, so a
// provider config reset never re-arms a scenario's attempt faults.
// Replacement (re-POST the same id) swaps in a brand-new *Scenario with
// fresh counters — that re-arms; a request already holding the old pointer
// finishes against the old counter (the registry's immutable-swap model).

// Chunking controls how exact output text is sliced into stream deltas —
// the application boundary a decoder reassembles. It is a different layer
// from the transport-level fragment_* knobs (sse.go), which split one frame
// across writes; those still compose on top.
type Chunking struct {
	// Mode: "whole" | "runes" | "words" (default "runes"). Deliberately no
	// byte-split mode — chunks must round-trip through the JSON delta
	// marshal, and invalid-UTF-8 testing belongs to fragment_split "rune".
	Mode string `json:"mode,omitempty"`
	// Size is the chunk width in runes/words; <= 0 means whole.
	Size int `json:"size,omitempty"`
}

// ExactOutput pins the assistant output byte-for-byte.
type ExactOutput struct {
	// Text is the verbatim assistant text (may contain \n, runs of spaces,
	// unicode). Preserved exactly — never strings.Fields'd or re-joined.
	Text string `json:"text"`
	// Thinking, when non-empty, owns the thinking/reasoning block verbatim
	// (Anthropic messages surface; other surfaces have no thinking channel
	// in v1 — see the emit paths). Empty = thinking generated from
	// config.reasoning_tokens as usual.
	Thinking string `json:"thinking,omitempty"`
	// Chunking slices Text (and Thinking) into stream deltas.
	Chunking Chunking `json:"chunking"`
	// OutputTokens pins the emitted output-token usage. 0 derives it
	// deterministically via the documented rule (R9): max(1,
	// (runes(Text)+runes(Thinking))/4).
	OutputTokens int `json:"output_tokens,omitempty"`
}

// exactOutputTokens resolves the output-token usage for an exact-output
// response — the R9 rule: honor an explicit output_tokens; otherwise derive
// deterministically as max(1, (runes(Text)+runes(Thinking))/4). The same
// rule feeds OpenAI chat completion_tokens, Anthropic output_tokens, and
// Bedrock usage.outputTokens.
func exactOutputTokens(out *ExactOutput) int {
	if out.OutputTokens > 0 {
		return out.OutputTokens
	}
	derived := (len([]rune(out.Text)) + len([]rune(out.Thinking))) / 4
	if derived < 1 {
		derived = 1
	}
	return derived
}

// ScenarioDef is the wire shape of a scenario definition (runtime state
// omitted).
type ScenarioDef struct {
	// ID is the registry key; also matchable via X-MockLM-Scenario.
	ID string `json:"id"`
	// Provider is "openai" | "anthropic" | "bedrock" and MUST match the
	// route the request arrived on (K2): a scenario cannot change the wire
	// shape of a route it did not route to. The header overrides model
	// selection, not provider selection.
	Provider string `json:"provider"`
	// Model is the model-name match key (optional if only header-matched).
	// Bedrock models come from the request path.
	Model string `json:"model,omitempty"`
	// Surface pins the v1 allow-list route: "chat" (/v1/chat/completions),
	// "messages" (/v1/messages), or "converse" (Bedrock converse +
	// converse-stream). Defaults from Provider. /v1/responses and legacy
	// /v1/completions are deliberately NOT scenario surfaces in v1 — they
	// word-generate and would silently lose exact content.
	Surface string `json:"surface,omitempty"`
	// Output pins exact content; nil falls back to generated words per
	// Config.Tokens etc.
	Output *ExactOutput `json:"output,omitempty"`
	// Config carries the scenario's faults, delays, usage knobs,
	// validate_responses, attempt_faults — the ProviderConfig surface
	// reused verbatim, EXCEPT rate_limit_rpm / max_concurrent which are
	// provider-global in v1 and rejected at register (R5).
	Config ProviderConfig `json:"config"`
}

// surfaceForProvider maps a provider to its single wired v1 surface.
func surfaceForProvider(provider string) string {
	switch provider {
	case "openai":
		return "chat"
	case "anthropic":
		return "messages"
	case "bedrock":
		return "converse"
	}
	return ""
}

// CapturedRequest is a per-scenario snapshot of the last matched request.
// It is BODY-exact, not wire-exact (K13/D3): RawBody is byte-for-byte what
// arrived, but Headers collapses repeated values and ordering and Path
// drops the query string — adequate for body-diff oracles, not full
// wire-fidelity assertions. Protocol (r.Proto) IS meaningful: on the TLS
// lane it reports the ALPN-negotiated protocol ("HTTP/2.0" vs "HTTP/1.1",
// see tls.go); on the clear-text lane it is always "HTTP/1.1".
type CapturedRequest struct {
	RawBody   []byte
	Method    string
	Path      string
	Headers   map[string]string
	Protocol  string
	Timestamp time.Time
}

// Scenario is a registered scenario plus its runtime state. Published
// scenarios are IMMUTABLE except for the atomics: replacement swaps a
// brand-new *Scenario into the store (K7), so a request holding this
// pointer never races an admin write.
type Scenario struct {
	ScenarioDef

	// captureCount counts matched-raw-requests: every request that matched
	// this scenario, incremented exactly once at the capture point. NOT the
	// fault-attempt counter below — capture happens before checkFaults, so
	// the two can differ (e.g. a malformed X-MockLM-Fault header is
	// captured but rejected before attempt accounting).
	captureCount atomic.Int64
	// faultAttempts is the scenario's own fault-attempt sequence: bumped by
	// checkFaults (via the request-local config's faultAttemptCounter) for
	// every matched request that reaches fault processing, and the ONLY
	// counter this scenario's fail_first_n / attempt_faults index. Zeroed
	// by POST /admin/scenarios/{id}/attempt-count/reset and implicitly by
	// replacement (a fresh *Scenario); deliberately NOT by /admin/reset.
	// Same-scenario concurrent requests are ordered by the atomic increment
	// — linearizable, but which caller draws attempt 1 is scheduler-
	// dependent; per-scenario isolation does not promise caller identity.
	faultAttempts atomic.Int64
	lastRequest   atomic.Pointer[CapturedRequest]
}

// capture records a matched request (R7: called right after the scenario
// match, from the already-buffered raw body) and bumps the matched counter.
func (sc *Scenario) capture(r *http.Request, rawBody []byte) {
	headers := make(map[string]string)
	for k := range r.Header {
		headers[k] = r.Header.Get(k)
	}
	// Store a private copy: rawBody's backing array belongs to the handler.
	body := make([]byte, len(rawBody))
	copy(body, rawBody)
	sc.lastRequest.Store(&CapturedRequest{
		RawBody:   body,
		Method:    r.Method,
		Path:      r.URL.Path,
		Headers:   headers,
		Protocol:  r.Proto,
		Timestamp: time.Now(),
	})
	sc.captureCount.Add(1)
}

// ScenarioStore holds registered scenarios. Readers take request-local
// copies of Config under the read lock; writers only ever swap whole
// *Scenario values — in-place mutation of a published scenario's definition
// is forbidden (the atomics are the only post-publication writes).
type ScenarioStore struct {
	mu   sync.RWMutex
	byID map[string]*Scenario
	// byKey indexes "provider\x00model" -> id for model-keyed matching.
	byKey map[string]string
}

func newScenarioStore() *ScenarioStore {
	return &ScenarioStore{
		byID:  make(map[string]*Scenario),
		byKey: make(map[string]string),
	}
}

func scenarioKey(provider, model string) string {
	return provider + "\x00" + model
}

// Register publishes a scenario. A (provider, model) pair already claimed
// by a DIFFERENT id is a conflict unless replace is set (a typo'd model
// must not silently steer another test); re-POSTing the same id always
// replaces (and resets that scenario's capture state).
func (st *ScenarioStore) Register(sc *Scenario, replace bool) (conflictWith string, err error) {
	st.mu.Lock()
	defer st.mu.Unlock()

	key := ""
	if sc.Model != "" {
		key = scenarioKey(sc.Provider, sc.Model)
		if otherID, ok := st.byKey[key]; ok && otherID != sc.ID && !replace {
			return otherID, fmt.Errorf("scenario %q already claims (provider=%s, model=%s); re-POST with ?replace=1 to take it over", otherID, sc.Provider, sc.Model)
		}
	}

	// Replacing an existing id: drop its old model index entry so a model
	// change doesn't leave a stale mapping behind.
	if old, ok := st.byID[sc.ID]; ok && old.Model != "" {
		if st.byKey[scenarioKey(old.Provider, old.Model)] == sc.ID {
			delete(st.byKey, scenarioKey(old.Provider, old.Model))
		}
	}

	st.byID[sc.ID] = sc
	if key != "" {
		st.byKey[key] = sc.ID
	}
	return "", nil
}

// Lookup returns the scenario registered under id.
func (st *ScenarioStore) Lookup(id string) (*Scenario, bool) {
	st.mu.RLock()
	defer st.mu.RUnlock()
	sc, ok := st.byID[id]
	return sc, ok
}

// MatchModel returns the scenario claiming (provider, model), if any.
func (st *ScenarioStore) MatchModel(provider, model string) (*Scenario, bool) {
	st.mu.RLock()
	defer st.mu.RUnlock()
	id, ok := st.byKey[scenarioKey(provider, model)]
	if !ok {
		return nil, false
	}
	sc, ok := st.byID[id]
	return sc, ok
}

// Delete removes one scenario.
func (st *ScenarioStore) Delete(id string) bool {
	st.mu.Lock()
	defer st.mu.Unlock()
	sc, ok := st.byID[id]
	if !ok {
		return false
	}
	delete(st.byID, id)
	if sc.Model != "" && st.byKey[scenarioKey(sc.Provider, sc.Model)] == id {
		delete(st.byKey, scenarioKey(sc.Provider, sc.Model))
	}
	return true
}

// Clear removes all scenarios.
func (st *ScenarioStore) Clear() {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.byID = make(map[string]*Scenario)
	st.byKey = make(map[string]string)
}

// List returns all definitions sorted by id (runtime state omitted).
func (st *ScenarioStore) List() []ScenarioDef {
	st.mu.RLock()
	defer st.mu.RUnlock()
	defs := make([]ScenarioDef, 0, len(st.byID))
	for _, sc := range st.byID {
		defs = append(defs, sc.ScenarioDef)
	}
	sort.Slice(defs, func(i, j int) bool { return defs[i].ID < defs[j].ID })
	return defs
}

// validateScenarioDef rejects definitions that cannot work as registered —
// loud at register time, never a silent no-op at request time.
func validateScenarioDef(def *ScenarioDef) error {
	if def.ID == "" {
		return fmt.Errorf("scenario id is required")
	}
	if strings.ContainsAny(def.ID, "/ \t\n") {
		return fmt.Errorf("scenario id %q must not contain slashes or whitespace (it is a path segment of the admin routes)", def.ID)
	}
	switch def.Provider {
	case "openai", "anthropic", "bedrock":
	default:
		return fmt.Errorf("unknown scenario provider %q (want openai, anthropic, or bedrock)", def.Provider)
	}
	want := surfaceForProvider(def.Provider)
	if def.Surface == "" {
		def.Surface = want
	} else if def.Surface != want {
		return fmt.Errorf("surface %q does not match provider %q (its v1 surface is %q; /v1/responses and /v1/completions are not scenario surfaces in v1)", def.Surface, def.Provider, want)
	}
	// R5: provider-global knobs are rejected at register, not silently
	// ignored — a scenario cannot carry a private rate limiter or
	// concurrency budget in v1 (D1).
	if def.Config.RateLimitRPM != 0 {
		return fmt.Errorf("rate_limit_rpm is provider-global in v1 and cannot be set per-scenario (set it on the provider config); rejected rather than silently ignored")
	}
	if def.Config.MaxConcurrent != 0 {
		return fmt.Errorf("max_concurrent is provider-global in v1 and cannot be set per-scenario (set it on the provider config); rejected rather than silently ignored")
	}
	if out := def.Output; out != nil {
		// Exact tool-call outputs are deferred in v1: the tool_use branch
		// would silently replace the exact output.text on non-stream
		// responses while the stream still carried it.
		if def.Config.ToolUseResponse {
			return fmt.Errorf("output cannot be combined with tool_use_response in v1 (exact tool-call outputs are not supported; the tool_use response would discard output.text); rejected rather than silently ignored")
		}
		switch out.Chunking.Mode {
		case "", "whole", "runes", "words":
		default:
			return fmt.Errorf("unknown chunking mode %q (want whole, runes, or words — there is deliberately no byte-split mode; use fragment_split \"rune\" for partial-rune transport tests)", out.Chunking.Mode)
		}
		if out.Chunking.Size < 0 {
			return fmt.Errorf("chunking size must be >= 0")
		}
	}
	// Fault specs fail loudly here instead of on the first matched request.
	for i := range def.Config.Faults {
		if err := validateFaultSpec(&def.Config.Faults[i]); err != nil {
			return fmt.Errorf("faults[%d]: %w", i, err)
		}
	}
	for ai := range def.Config.AttemptFaults {
		for i := range def.Config.AttemptFaults[ai] {
			if err := validateFaultSpec(&def.Config.AttemptFaults[ai][i]); err != nil {
				return fmt.Errorf("attempt_faults[%d][%d]: %w", ai, i, err)
			}
		}
	}
	return nil
}

// matchScenario resolves the scenario for a request AFTER body read + model
// decode (K1: there is no scenario identity before the body exists, so the
// provider-global concurrency gate and limiter have already run). Precedence:
//  1. X-MockLM-Scenario header — exact id lookup; wins over model matching,
//     but the scenario's provider must equal the route's (K2): mismatch is
//     a loud 409, an unknown id a loud 404.
//  2. (provider, model) index.
//  3. nil — fall through to the provider-global config.
//
// Returns (scenario, 0, "") on match or no-match; (nil, status, msg) when
// the request must be rejected.
func matchScenario(store *ScenarioStore, r *http.Request, provider, surface, model string) (*Scenario, int, string) {
	if id := r.Header.Get("X-MockLM-Scenario"); id != "" {
		sc, ok := store.Lookup(id)
		if !ok {
			return nil, 404, fmt.Sprintf("unknown scenario %q (X-MockLM-Scenario)", id)
		}
		if sc.Provider != provider || sc.Surface != surface {
			return nil, 409, fmt.Sprintf("scenario %q is bound to provider %q surface %q and cannot be served on the %s %s route — a scenario cannot change the wire shape of a route it did not route to", id, sc.Provider, sc.Surface, provider, surface)
		}
		return sc, 0, ""
	}
	if model != "" {
		if sc, ok := store.MatchModel(provider, model); ok && sc.Surface == surface {
			return sc, 0, ""
		}
	}
	return nil, 0, ""
}

// rejectScenarioHeaderUnwired makes a present X-MockLM-Scenario header on a
// non-scenario surface (/v1/responses, /v1/completions, /v1/embeddings) a
// loud 400 instead of a silent no-op (R4) — consistent with the provider-
// mismatch loudness.
func rejectScenarioHeaderUnwired(w http.ResponseWriter, r *http.Request, cfg *ProviderConfig, provider string) bool {
	if r.Header.Get("X-MockLM-Scenario") == "" {
		return false
	}
	writeErrorResponse(w, cfg, 400, provider, "invalid_request_error",
		"scenarios are not supported on this surface in v1 (X-MockLM-Scenario is honored on /v1/chat/completions, /v1/messages, and Bedrock converse/converse-stream only)")
	return true
}

// applyScenario folds a matched scenario into the request: captures the raw
// body (R7), and returns the request-local config copy — the scenario's
// Config with the PROVIDER-GLOBAL knobs (rate_limit_rpm / max_concurrent)
// carried over from the provider snapshot so the shared limiter and the
// introspection surface keep working (K1; scenario configs cannot set them,
// R5). The scenario's fault-attempt counter rides along so checkFaults
// indexes fail_first_n / attempt_faults off THIS scenario's sequence
// instead of the provider counter.
func applyScenario(sc *Scenario, r *http.Request, rawBody []byte, providerCfg *ProviderConfig) ProviderConfig {
	sc.capture(r, rawBody)
	cfg := sc.Config
	cfg.RateLimitRPM = providerCfg.RateLimitRPM
	cfg.MaxConcurrent = providerCfg.MaxConcurrent
	cfg.faultAttemptCounter = &sc.faultAttempts
	return cfg
}

// --- Admin handlers (native routes, §1.9) ---

// handleAdminPostScenario registers or replaces a scenario.
// Body: {id, provider, model?, surface?, output?, config?, fault_preset?}.
// fault_preset names a builtin fault preset whose config fragment becomes
// the base the scenario's own config overlays (absent keys keep preset
// values). (provider, model) collisions with another id return 409 unless
// ?replace=1.
func handleAdminPostScenario(state *ServerState) http.HandlerFunc {
	faultPresets := builtinFaultPresets()

	return func(w http.ResponseWriter, r *http.Request) {
		var reg struct {
			ID          string          `json:"id"`
			Provider    string          `json:"provider"`
			Model       string          `json:"model"`
			Surface     string          `json:"surface"`
			Output      *ExactOutput    `json:"output"`
			Config      json.RawMessage `json:"config"`
			FaultPreset string          `json:"fault_preset"`
		}
		if err := json.NewDecoder(r.Body).Decode(&reg); err != nil {
			writeErrorResponse(w, nil, 400, "admin", "invalid_request_error", "Invalid JSON: "+err.Error())
			return
		}

		def := ScenarioDef{
			ID:       reg.ID,
			Provider: reg.Provider,
			Model:    reg.Model,
			Surface:  reg.Surface,
			Output:   reg.Output,
		}
		if reg.FaultPreset != "" {
			preset, ok := faultPresets[reg.FaultPreset]
			if !ok {
				writeErrorResponse(w, nil, 404, "admin", "not_found", "Unknown fault preset: "+reg.FaultPreset)
				return
			}
			def.Config = preset.Config
		}
		if len(reg.Config) > 0 {
			if err := json.Unmarshal(reg.Config, &def.Config); err != nil {
				writeErrorResponse(w, nil, 400, "admin", "invalid_request_error", "Invalid scenario config: "+err.Error())
				return
			}
		}
		if err := validateScenarioDef(&def); err != nil {
			writeErrorResponse(w, nil, 400, "admin", "invalid_request_error", err.Error())
			return
		}
		applyProviderDefaults(&def.Config)

		sc := &Scenario{ScenarioDef: def}
		replace := r.URL.Query().Get("replace") == "1"
		if _, err := state.Scenarios().Register(sc, replace); err != nil {
			writeErrorResponse(w, nil, 409, "admin", "invalid_request_error", err.Error())
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"status":   "registered",
			"scenario": def,
		})
	}
}

// handleAdminGetScenarios lists all scenario definitions.
func handleAdminGetScenarios(state *ServerState) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"scenarios": state.Scenarios().List(),
		})
	}
}

// handleAdminGetScenario returns one definition.
func handleAdminGetScenario(state *ServerState) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		sc, ok := state.Scenarios().Lookup(r.PathValue("id"))
		if !ok {
			writeErrorResponse(w, nil, 404, "admin", "not_found", "Unknown scenario: "+r.PathValue("id"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(sc.ScenarioDef)
	}
}

// handleAdminDeleteScenario removes one scenario.
func handleAdminDeleteScenario(state *ServerState) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !state.Scenarios().Delete(r.PathValue("id")) {
			writeErrorResponse(w, nil, 404, "admin", "not_found", "Unknown scenario: "+r.PathValue("id"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"status": "deleted"})
	}
}

// handleAdminClearScenarios removes all scenarios (the register-per-test /
// clear lifecycle; POST /admin/reset deliberately does NOT do this).
func handleAdminClearScenarios(state *ServerState) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		state.Scenarios().Clear()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"status": "cleared"})
	}
}

// handleAdminScenarioLastRequest returns the raw bytes of the last matched
// request BODY — written directly (not re-marshaled), so a byte-diff
// against what the client sent is exact. Method/path/protocol/time ride
// response headers; body-exact, not wire-exact (K13/D3).
func handleAdminScenarioLastRequest(state *ServerState) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		sc, ok := state.Scenarios().Lookup(r.PathValue("id"))
		if !ok {
			writeErrorResponse(w, nil, 404, "admin", "not_found", "Unknown scenario: "+r.PathValue("id"))
			return
		}
		cap := sc.lastRequest.Load()
		if cap == nil {
			writeErrorResponse(w, nil, 404, "admin", "not_found", "Scenario "+sc.ID+" has not matched any request yet")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-MockLM-Captured-Method", cap.Method)
		w.Header().Set("X-MockLM-Captured-Path", cap.Path)
		w.Header().Set("X-MockLM-Captured-Protocol", cap.Protocol)
		w.Header().Set("X-MockLM-Captured-At", cap.Timestamp.UTC().Format(time.RFC3339Nano))
		w.Write(cap.RawBody)
	}
}

// handleAdminScenarioRequestCount returns {"count": N} where N counts
// matched-raw-requests (every request that matched this scenario id) — NOT
// "attempts that reached fault processing": that is the scenario's own
// fault-attempt counter at /admin/scenarios/{id}/attempt-count, and the
// provider-wide one at /admin/request-count (K6).
func handleAdminScenarioRequestCount(state *ServerState) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		sc, ok := state.Scenarios().Lookup(r.PathValue("id"))
		if !ok {
			writeErrorResponse(w, nil, 404, "admin", "not_found", "Unknown scenario: "+r.PathValue("id"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"count": sc.captureCount.Load()})
	}
}

// handleAdminScenarioResetRequestCount zeroes one scenario's matched
// counter (capture-only; the fault-attempt counter has its own reset at
// /admin/scenarios/{id}/attempt-count/reset).
func handleAdminScenarioResetRequestCount(state *ServerState) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		sc, ok := state.Scenarios().Lookup(r.PathValue("id"))
		if !ok {
			writeErrorResponse(w, nil, 404, "admin", "not_found", "Unknown scenario: "+r.PathValue("id"))
			return
		}
		sc.captureCount.Store(0)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"status": "reset"})
	}
}

// handleAdminScenarioAttemptCount returns {"count": N} where N is the
// scenario's fault-attempt sequence position — how many matched requests
// reached fault processing since registration/replacement or the last
// attempt-count reset. This is the counter the scenario's fail_first_n /
// attempt_faults index, and the retry-test oracle: after a client's
// retry loop, assert exactly how many attempts THIS scenario absorbed,
// independent of any other traffic on the provider.
func handleAdminScenarioAttemptCount(state *ServerState) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		sc, ok := state.Scenarios().Lookup(r.PathValue("id"))
		if !ok {
			writeErrorResponse(w, nil, 404, "admin", "not_found", "Unknown scenario: "+r.PathValue("id"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"count": sc.faultAttempts.Load()})
	}
}

// handleAdminScenarioResetAttemptCount zeroes one scenario's fault-attempt
// counter, re-arming its fail_first_n / attempt_faults sequence from
// attempt 1 — without touching its capture state, any other scenario, or
// the provider counters.
func handleAdminScenarioResetAttemptCount(state *ServerState) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		sc, ok := state.Scenarios().Lookup(r.PathValue("id"))
		if !ok {
			writeErrorResponse(w, nil, 404, "admin", "not_found", "Unknown scenario: "+r.PathValue("id"))
			return
		}
		sc.faultAttempts.Store(0)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"status": "reset"})
	}
}
