package main

import (
	"encoding/json"
	"fmt"
)

// The Anthropic request-shape checker (bounded) — `strict = true`.
//
// The default (lenient) mode accepts nearly anything — it sizes mock output
// and never validates schemas. Strict mode adds a bounded, Anthropic-only,
// manual-allowlist shape check on the request side: requests the real API
// would 400 get a 400 here, so a proxy's transform output can be checked
// end-to-end without live API calls. It is NOT a general request-schema
// validator.
//
// Validation depth (documented, deliberately bounded):
//   - top-level: unknown fields rejected (additionalProperties: false),
//     required fields, temperature range;
//   - tools[]: OpenAI wrapper shapes rejected; name required; schema-less
//     entries must be typed server tools;
//   - tool_choice: must be the Anthropic object form with a known type;
//   - messages[]: Anthropic roles only (no OpenAI system/tool roles),
//     content blocks must be typed, and the common block types
//     (text/tool_use/tool_result/thinking) carry their required fields.
//
// It does NOT re-implement the full InputContentBlock union — uncommon typed
// blocks (documents, search results, server tools) are accepted on shape.
//
// Field sets mirror the vendored Anthropic OpenAPI spec (nanollm
// spec/anthropic-openapi.json, 2026-06-11 snapshot). Bump alongside spec
// bumps — see nanollm docs/EXTENSION_SURFACE.md.
var anthropicAllowedFields = map[string]bool{
	"cache_control":  true,
	"container":      true,
	"inference_geo":  true,
	"max_tokens":     true,
	"messages":       true,
	"metadata":       true,
	"model":          true,
	"output_config":  true,
	"service_tier":   true,
	"stop_sequences": true,
	"stream":         true,
	"system":         true,
	"temperature":    true,
	"thinking":       true,
	"tool_choice":    true,
	"tools":          true,
	"top_k":          true,
	"top_p":          true,
}

// validateAnthropicStrict returns a non-empty message when the request
// would be rejected by the real API.
func validateAnthropicStrict(rawBody []byte) string {
	var body map[string]json.RawMessage
	if err := json.Unmarshal(rawBody, &body); err != nil {
		return "Invalid JSON: " + err.Error()
	}

	for field := range body {
		if !anthropicAllowedFields[field] {
			return fmt.Sprintf("Unexpected field: %s", field)
		}
	}

	for _, required := range []string{"model", "messages", "max_tokens"} {
		if _, ok := body[required]; !ok {
			return fmt.Sprintf("Field required: %s", required)
		}
	}

	var maxTokens int
	if err := json.Unmarshal(body["max_tokens"], &maxTokens); err != nil || maxTokens < 1 {
		return "max_tokens: must be an integer >= 1"
	}

	if rawTemp, ok := body["temperature"]; ok {
		var temp float64
		if err := json.Unmarshal(rawTemp, &temp); err != nil || temp < 0 || temp > 1 {
			return "temperature: must be between 0 and 1"
		}
	}

	if msg := validateStrictMessages(body["messages"]); msg != "" {
		return msg
	}
	if rawTools, ok := body["tools"]; ok {
		if msg := validateStrictTools(rawTools); msg != "" {
			return msg
		}
	}
	if rawChoice, ok := body["tool_choice"]; ok {
		if msg := validateStrictToolChoice(rawChoice); msg != "" {
			return msg
		}
	}

	return ""
}

func validateStrictMessages(raw json.RawMessage) string {
	var messages []map[string]json.RawMessage
	if err := json.Unmarshal(raw, &messages); err != nil || len(messages) == 0 {
		return "messages: must be a non-empty array of objects"
	}

	for i, msg := range messages {
		var role string
		if err := json.Unmarshal(msg["role"], &role); err != nil {
			return fmt.Sprintf("messages.%d.role: required", i)
		}
		// OpenAI roles that a proxy must reshape away.
		if role != "user" && role != "assistant" {
			return fmt.Sprintf("messages.%d.role: must be user or assistant, got %q", i, role)
		}

		rawContent, ok := msg["content"]
		if !ok {
			return fmt.Sprintf("messages.%d.content: required", i)
		}
		var asString string
		if json.Unmarshal(rawContent, &asString) == nil {
			continue // plain string content
		}
		var blocks []map[string]json.RawMessage
		if err := json.Unmarshal(rawContent, &blocks); err != nil {
			return fmt.Sprintf("messages.%d.content: must be a string or array of blocks", i)
		}
		for j, block := range blocks {
			if msg := validateStrictBlock(i, j, block); msg != "" {
				return msg
			}
		}
	}
	return ""
}

func validateStrictBlock(i, j int, block map[string]json.RawMessage) string {
	var blockType string
	if err := json.Unmarshal(block["type"], &blockType); err != nil {
		return fmt.Sprintf("messages.%d.content.%d.type: required", i, j)
	}

	at := func(field string) string {
		return fmt.Sprintf("messages.%d.content.%d (%s): %s required", i, j, blockType, field)
	}

	switch blockType {
	case "text":
		if _, ok := block["text"]; !ok {
			return at("text")
		}
	case "tool_use":
		for _, field := range []string{"id", "name", "input"} {
			if _, ok := block[field]; !ok {
				return at(field)
			}
		}
	case "tool_result":
		if _, ok := block["tool_use_id"]; !ok {
			return at("tool_use_id")
		}
	case "thinking":
		for _, field := range []string{"thinking", "signature"} {
			if _, ok := block[field]; !ok {
				return at(field)
			}
		}
	case "image_url":
		// The OpenAI image shape a proxy must reshape to {type: "image"}.
		return fmt.Sprintf("messages.%d.content.%d: image_url is not an Anthropic block type", i, j)
	}
	// Other typed blocks (image, document, search_result, server tools, ...)
	// are accepted on shape — see the depth note above.
	return ""
}

func validateStrictTools(raw json.RawMessage) string {
	var tools []map[string]json.RawMessage
	if err := json.Unmarshal(raw, &tools); err != nil {
		return "tools: must be an array of objects"
	}

	for i, tool := range tools {
		// OpenAI wrapper shapes a proxy must unwrap.
		if _, ok := tool["function"]; ok {
			return fmt.Sprintf("tools.%d: unexpected field function (OpenAI tool shape)", i)
		}
		if _, ok := tool["custom"]; ok {
			return fmt.Sprintf("tools.%d: unexpected field custom (OpenAI tool shape)", i)
		}

		var name string
		if err := json.Unmarshal(tool["name"], &name); err != nil || name == "" {
			return fmt.Sprintf("tools.%d.name: required", i)
		}
		// Plain client tools need an input_schema object; typed entries
		// (bash_20250124, web_search_20250305, ...) carry their own shapes.
		if _, typed := tool["type"]; !typed {
			var schema map[string]json.RawMessage
			if err := json.Unmarshal(tool["input_schema"], &schema); err != nil {
				return fmt.Sprintf("tools.%d.input_schema: required object", i)
			}
		}
	}
	return ""
}

func validateStrictToolChoice(raw json.RawMessage) string {
	var choice map[string]json.RawMessage
	if err := json.Unmarshal(raw, &choice); err != nil {
		// Catches OpenAI string forms ("auto"/"none"/"required") a proxy
		// must reshape to the object form.
		return "tool_choice: must be an object with a type field"
	}

	var choiceType string
	if err := json.Unmarshal(choice["type"], &choiceType); err != nil {
		return "tool_choice.type: required"
	}
	switch choiceType {
	case "auto", "any", "none":
	case "tool":
		var name string
		if err := json.Unmarshal(choice["name"], &name); err != nil || name == "" {
			return "tool_choice.name: required when type is tool"
		}
	default:
		return fmt.Sprintf("tool_choice.type: must be auto, any, none, or tool; got %q", choiceType)
	}
	return ""
}
