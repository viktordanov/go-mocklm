package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// --- Phase 1: Anthropic stream event-ordering contract ---
//
// Schema validation proves each SSE payload is a well-formed
// MessageStreamEvent, but it cannot see the protocol between frames:
// message_start → content_block_start/delta/stop groups → message_delta →
// message_stop. The Anthropic spec never wires MessageStreamEvent to a
// path (it is a components-only schema — the same reason xtask lists it
// as an explicit root), so the ordering contract lives here, hand-written
// and table-driven.

type streamEvent struct {
	name string
	data map[string]any
}

// collectAnthropicEvents parses an Anthropic SSE stream into (event name,
// payload) pairs.
func collectAnthropicEvents(t *testing.T, srvURL, body string) []streamEvent {
	t.Helper()
	resp := postAnthropic(t, srvURL, body)
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	var events []streamEvent
	current := ""
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		switch {
		case strings.HasPrefix(line, "event: "):
			current = strings.TrimPrefix(line, "event: ")
		case strings.HasPrefix(line, "data: "):
			var data map[string]any
			if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &data); err != nil {
				t.Fatalf("event %q carries invalid JSON: %v", current, err)
			}
			events = append(events, streamEvent{name: current, data: data})
		}
	}
	return events
}

// checkAnthropicSequence enforces the stream-ordering contract on a full
// event list:
//
//  1. the first event is message_start, and it appears exactly once
//  2. every payload's "type" matches its SSE event name
//  3. ping events appear only before the first content block
//  4. content blocks are properly bracketed: start opens block N (indexes
//     are sequential from 0), deltas carry the open block's index, stop
//     closes it; no nesting, no deltas outside a block
//  5. message_delta arrives after all blocks are closed, exactly once,
//     and carries a non-null delta.stop_reason
//  6. message_stop is the final event, exactly once, after message_delta
func checkAnthropicSequence(events []streamEvent) error {
	if len(events) == 0 {
		return fmt.Errorf("empty stream")
	}
	if events[0].name != "message_start" {
		return fmt.Errorf("first event is %q, want message_start", events[0].name)
	}

	openBlock := -1 // index of the currently open content block, -1 = none
	nextBlock := 0  // the index the next content_block_start must carry
	sawDelta := false
	sawStop := false

	for i, ev := range events {
		if typ, _ := ev.data["type"].(string); typ != ev.name {
			return fmt.Errorf("event %d: SSE name %q != payload type %q", i, ev.name, typ)
		}
		if sawStop {
			return fmt.Errorf("event %d: %q after message_stop", i, ev.name)
		}

		switch ev.name {
		case "message_start":
			if i != 0 {
				return fmt.Errorf("event %d: duplicate message_start", i)
			}
		case "ping":
			if nextBlock > 0 || openBlock != -1 {
				return fmt.Errorf("event %d: ping after content began", i)
			}
		case "content_block_start":
			if sawDelta {
				return fmt.Errorf("event %d: content_block_start after message_delta", i)
			}
			if openBlock != -1 {
				return fmt.Errorf("event %d: block %d still open", i, openBlock)
			}
			if idx := blockIndex(ev); idx != nextBlock {
				return fmt.Errorf("event %d: block index %d, want %d", i, idx, nextBlock)
			}
			openBlock = nextBlock
		case "content_block_delta":
			if openBlock == -1 {
				return fmt.Errorf("event %d: delta outside a block", i)
			}
			if idx := blockIndex(ev); idx != openBlock {
				return fmt.Errorf("event %d: delta index %d, open block is %d", i, idx, openBlock)
			}
		case "content_block_stop":
			if openBlock == -1 {
				return fmt.Errorf("event %d: stop without an open block", i)
			}
			if idx := blockIndex(ev); idx != openBlock {
				return fmt.Errorf("event %d: stop index %d, open block is %d", i, idx, openBlock)
			}
			openBlock = -1
			nextBlock++
		case "message_delta":
			if sawDelta {
				return fmt.Errorf("event %d: duplicate message_delta", i)
			}
			if openBlock != -1 {
				return fmt.Errorf("event %d: message_delta with block %d open", i, openBlock)
			}
			if nextBlock == 0 {
				return fmt.Errorf("event %d: message_delta before any content block", i)
			}
			delta, _ := ev.data["delta"].(map[string]any)
			if delta == nil || delta["stop_reason"] == nil {
				return fmt.Errorf("event %d: message_delta without a stop_reason", i)
			}
			sawDelta = true
		case "message_stop":
			if !sawDelta {
				return fmt.Errorf("event %d: message_stop before message_delta", i)
			}
			sawStop = true
		default:
			return fmt.Errorf("event %d: unknown event %q", i, ev.name)
		}
	}

	if !sawStop {
		return fmt.Errorf("stream ended without message_stop")
	}
	return nil
}

func blockIndex(ev streamEvent) int {
	f, ok := ev.data["index"].(float64)
	if !ok {
		return -1
	}
	return int(f)
}

// collapseEvents folds runs of consecutive same-named events into
// "name+" so expectations stay stable across token counts (every content
// block in the table emits at least two deltas).
func collapseEvents(events []streamEvent) []string {
	var out []string
	for i := 0; i < len(events); {
		j := i
		for j < len(events) && events[j].name == events[i].name {
			j++
		}
		if j-i > 1 {
			out = append(out, events[i].name+"+")
		} else {
			out = append(out, events[i].name)
		}
		i = j
	}
	return out
}

// blockTypes returns content_block.type for each content_block_start.
func blockTypes(events []streamEvent) []string {
	var out []string
	for _, ev := range events {
		if ev.name != "content_block_start" {
			continue
		}
		block, _ := ev.data["content_block"].(map[string]any)
		typ, _ := block["type"].(string)
		out = append(out, typ)
	}
	return out
}

func TestAnthropicStreamEventSequenceContract(t *testing.T) {
	cases := []struct {
		name           string
		mutate         func(*ProviderConfig)
		wantSequence   []string
		wantBlockTypes []string
	}{
		{
			name:   "default",
			mutate: func(*ProviderConfig) {},
			wantSequence: []string{
				"message_start", "ping",
				"content_block_start", "content_block_delta+", "content_block_stop",
				"message_delta", "message_stop",
			},
			wantBlockTypes: []string{"text"},
		},
		{
			name:   "suppress_ping_events",
			mutate: func(c *ProviderConfig) { c.SuppressPingEvents = true },
			wantSequence: []string{
				"message_start",
				"content_block_start", "content_block_delta+", "content_block_stop",
				"message_delta", "message_stop",
			},
			wantBlockTypes: []string{"text"},
		},
		{
			name:   "reasoning",
			mutate: func(c *ProviderConfig) { c.ReasoningTokens = 15 },
			wantSequence: []string{
				"message_start", "ping",
				"content_block_start", "content_block_delta+", "content_block_stop",
				"content_block_start", "content_block_delta+", "content_block_stop",
				"message_delta", "message_stop",
			},
			wantBlockTypes: []string{"thinking", "text"},
		},
		{
			name:   "tool_use",
			mutate: func(c *ProviderConfig) { c.ToolUseResponse = true },
			wantSequence: []string{
				"message_start", "ping",
				"content_block_start", "content_block_delta+", "content_block_stop",
				"content_block_start", "content_block_delta+", "content_block_stop",
				"message_delta", "message_stop",
			},
			wantBlockTypes: []string{"text", "tool_use"},
		},
		{
			name: "reasoning+tool_use, no ping",
			mutate: func(c *ProviderConfig) {
				c.ReasoningTokens = 15
				c.ToolUseResponse = true
				c.SuppressPingEvents = true
			},
			wantSequence: []string{
				"message_start",
				"content_block_start", "content_block_delta+", "content_block_stop",
				"content_block_start", "content_block_delta+", "content_block_stop",
				"content_block_start", "content_block_delta+", "content_block_stop",
				"message_delta", "message_stop",
			},
			wantBlockTypes: []string{"thinking", "text", "tool_use"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := defaultConfig()
			cfg.Anthropic.ValidateResponses = boolPtr(true)
			tc.mutate(&cfg.Anthropic)
			srv := testServer(cfg)
			defer srv.Close()

			events := collectAnthropicEvents(t, srv.URL, anthropicBody(true))

			if err := checkAnthropicSequence(events); err != nil {
				t.Fatalf("ordering contract violated: %v\nevents: %v", err, collapseEvents(events))
			}
			got := collapseEvents(events)
			if fmt.Sprint(got) != fmt.Sprint(tc.wantSequence) {
				t.Fatalf("sequence mismatch:\n got %v\nwant %v", got, tc.wantSequence)
			}
			if bt := blockTypes(events); fmt.Sprint(bt) != fmt.Sprint(tc.wantBlockTypes) {
				t.Fatalf("block types mismatch:\n got %v\nwant %v", bt, tc.wantBlockTypes)
			}
		})
	}
}

// TestSequenceCheckerHasTeeth feeds the checker deliberately broken
// sequences — a checker that accepts everything would make the contract
// test vacuous.
func TestSequenceCheckerHasTeeth(t *testing.T) {
	ev := func(name string, extra map[string]any) streamEvent {
		data := map[string]any{"type": name}
		for k, v := range extra {
			data[k] = v
		}
		return streamEvent{name: name, data: data}
	}
	start := ev("message_start", nil)
	bStart := ev("content_block_start", map[string]any{"index": float64(0), "content_block": map[string]any{"type": "text"}})
	bDelta := ev("content_block_delta", map[string]any{"index": float64(0)})
	bStop := ev("content_block_stop", map[string]any{"index": float64(0)})
	mDelta := ev("message_delta", map[string]any{"delta": map[string]any{"stop_reason": "end_turn"}})
	mStop := ev("message_stop", nil)

	valid := []streamEvent{start, bStart, bDelta, bStop, mDelta, mStop}
	if err := checkAnthropicSequence(valid); err != nil {
		t.Fatalf("canonical sequence must pass: %v", err)
	}

	cases := []struct {
		name   string
		events []streamEvent
	}{
		{"missing message_start", []streamEvent{bStart, bDelta, bStop, mDelta, mStop}},
		{"delta outside a block", []streamEvent{start, bDelta, mDelta, mStop}},
		{"unclosed block before message_delta", []streamEvent{start, bStart, bDelta, mDelta, mStop}},
		{"message_stop before message_delta", []streamEvent{start, bStart, bDelta, bStop, mStop}},
		{"event after message_stop", []streamEvent{start, bStart, bDelta, bStop, mDelta, mStop, mDelta}},
		{"missing message_stop", []streamEvent{start, bStart, bDelta, bStop, mDelta}},
		{"payload type mismatch", []streamEvent{{name: "message_start", data: map[string]any{"type": "ping"}}, bStart, bDelta, bStop, mDelta, mStop}},
	}
	for _, tc := range cases {
		if err := checkAnthropicSequence(tc.events); err == nil {
			t.Errorf("%s: checker accepted a broken sequence", tc.name)
		}
	}
}
