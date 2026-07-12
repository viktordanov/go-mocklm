package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"sort"
	"strconv"
)

// Cache-control request-rejection oracle (reject_cache_control).
//
// A gateway that forwards a client's cache_control blocks to the upstream
// provider is leaking a caller-facing knob across the trust boundary. With
// reject_cache_control enabled, the mock asserts the negative: any JSON
// object in the inbound body whose "cache_control" member itself contains a
// "type" member (the wire shape of a real cache_control block, e.g.
// {"type":"ephemeral"}) draws a provider-shaped 400 naming the leak's JSON
// path. A cache_control member WITHOUT a "type" member does not match — the
// predicate is the consumer contract, not the key name alone.
//
// One shared raw-JSON walker serves every body-bearing surface (OpenAI
// chat/responses/completions/embeddings, Anthropic messages, Bedrock
// converse). Handlers call it AFTER their own JSON decode succeeded
// (malformed bodies keep the normal invalid-JSON error) and BEFORE
// checkFaults, so a configured fault can never mask the assertion.

// rejectLeakedCacheControl runs the oracle when the resolved config enables
// it. Returns true when the request was rejected (caller must return).
func rejectLeakedCacheControl(w http.ResponseWriter, cfg *ProviderConfig, provider string, rawBody []byte) bool {
	if !cfg.RejectCacheControl {
		return false
	}
	var v any
	if err := json.Unmarshal(rawBody, &v); err != nil {
		// The handler's own decode governs malformed-JSON behavior; the
		// oracle only inspects bodies that parsed.
		return false
	}
	path, found := findCacheControlLeak(v, "")
	if !found {
		return false
	}
	writeErrorResponse(w, cfg, 400, provider, errorTypeForStatus(400, provider),
		fmt.Sprintf("cache_control must not reach the provider: found cache_control.type at %s (reject_cache_control oracle — the gateway under test should strip cache_control before forwarding)", path))
	return true
}

// findCacheControlLeak walks decoded JSON depth-first and returns the JSON
// path of the first object whose "cache_control" member is an object
// containing a "type" member. Object members are visited in lexicographic
// key order (Go maps have no document order), so the reported path is
// deterministic; an object's own match wins over matches in its children.
func findCacheControlLeak(v any, path string) (string, bool) {
	switch node := v.(type) {
	case map[string]any:
		if cc, ok := node["cache_control"].(map[string]any); ok {
			if _, ok := cc["type"]; ok {
				return joinPath(path, "cache_control"), true
			}
		}
		keys := make([]string, 0, len(node))
		for k := range node {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			if p, found := findCacheControlLeak(node[k], joinPath(path, k)); found {
				return p, true
			}
		}
	case []any:
		for i, item := range node {
			if p, found := findCacheControlLeak(item, path+"["+strconv.Itoa(i)+"]"); found {
				return p, true
			}
		}
	}
	return "", false
}

// plainKey matches keys that read cleanly in dotted-path notation.
var plainKey = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

// joinPath appends an object key to a JSON path, bracket-quoting keys that
// dotted notation cannot carry (empty, spaces, dots, quotes, control
// characters, ...). The bracket segment is a full quoted string literal
// (strconv.Quote), so quotes, backslashes, and control characters like
// \n/\t/\r round-trip unambiguously in the diagnostic.
func joinPath(path, key string) string {
	if plainKey.MatchString(key) {
		if path == "" {
			return key
		}
		return path + "." + key
	}
	return path + "[" + strconv.Quote(key) + "]"
}
