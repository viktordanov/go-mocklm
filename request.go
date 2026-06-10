package main

import "encoding/json"

// chatMessage is a provider-agnostic view of one request message whose
// content may be a plain string or an array of typed content blocks
// (Anthropic blocks like text/tool_result/image, or OpenAI multi-part
// content). Real providers accept both shapes; so must the mock.
type chatMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// contentChars approximates the character count of message content,
// accepting both string and block-array shapes. Used for mock prompt-token
// estimates.
//
// Lenient by design: ANY JSON shape is accepted and approximated — this
// sizes mock output, it does NOT validate request schemas. A request the
// real API would 400 (e.g. content as a number, malformed blocks) still
// gets a response here. Schema validation is the planned Phase 2 strict
// mode, not this function's job.
func (m chatMessage) contentChars() int {
	if len(m.Content) == 0 {
		return 0
	}

	var s string
	if err := json.Unmarshal(m.Content, &s); err == nil {
		return len(s)
	}

	var blocks []map[string]any
	if err := json.Unmarshal(m.Content, &blocks); err != nil {
		// Unknown shape — approximate with the raw JSON length.
		return len(m.Content)
	}

	total := 0
	for _, b := range blocks {
		// text blocks, thinking blocks, and string-content tool_results
		for _, key := range []string{"text", "thinking", "content"} {
			if v, ok := b[key].(string); ok {
				total += len(v)
			}
		}
	}
	return total
}

// firstRequestedToolName extracts the name of the first tool offered in the
// request, normalizing across the OpenAI shape
// ({type:"function", function:{name}} / {type:"custom", custom:{name}})
// and the Anthropic shape ({name, input_schema}).
// Returns "" when no tool name is found.
func firstRequestedToolName(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var tools []map[string]any
	if err := json.Unmarshal(raw, &tools); err != nil {
		return ""
	}
	for _, tool := range tools {
		if name, ok := tool["name"].(string); ok && name != "" {
			return name
		}
		for _, wrapper := range []string{"function", "custom"} {
			if inner, ok := tool[wrapper].(map[string]any); ok {
				if name, ok := inner["name"].(string); ok && name != "" {
					return name
				}
			}
		}
	}
	return ""
}
