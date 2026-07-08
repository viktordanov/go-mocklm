# go-mocklm

A configurable mock LLM server that emulates OpenAI and Anthropic APIs. Built for integration-testing LLM proxy/gateway projects (like [nanollm](https://github.com/viktordanov/nanollm)) without real API keys or network calls.

Supports streaming and non-streaming responses, runtime fault injection, named presets for common failure scenarios, and an admin API for live configuration changes.

## Providers Emulated

| Provider | Endpoints | Auth Validation |
|---|---|---|
| **OpenAI** | `POST /v1/chat/completions`, `GET /v1/models`, `POST /v1/responses` | None |
| **Anthropic** | `POST /v1/messages` | Requires `x-api-key` and `anthropic-version` headers (any value accepted) |

Both providers return correctly-structured response JSON and provider-specific error formats.

## Quick Start

```bash
# Build
go build -o mocklm

# Run with default config
./mocklm

# Run with custom config
CONFIG_PATH=my-config.toml ./mocklm
```

The server starts on `0.0.0.0:9999` by default.

### Basic Usage

**OpenAI Chat Completion (non-streaming):**

```bash
curl -s http://localhost:9999/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

**OpenAI Chat Completion (streaming):**

```bash
curl -sN http://localhost:9999/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "stream": true,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

**Anthropic Messages:**

```bash
curl -s http://localhost:9999/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: any-key" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-3-haiku-20240307",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

**OpenAI Responses API:**

```bash
curl -s http://localhost:9999/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "input": [{"role": "user", "content": "Hello"}]
  }'
```

**Health check:**

```bash
curl http://localhost:9999/health
# {"status":"ok"}
```

## Configuration

Config is loaded from `config.toml` (or the path in `CONFIG_PATH` env var).

```toml
[server]
host = "0.0.0.0"   # default: 0.0.0.0
port = 9999         # default: 9999

[openai]
latency_ms = 100
tokens = 20
stream_delay_ms = 50
error_rate = 0.0
error_status = 500
timeout_ms = 0
disconnect_after_chunks = 0
malformed_chunk = false
rate_limit_rpm = 0
reasoning_tokens = 0
thinking_delay_ms = 0

[anthropic]
# Same fields as [openai] — configured independently
latency_ms = 100
tokens = 20
stream_delay_ms = 50
error_rate = 0.0
error_status = 500
timeout_ms = 0
disconnect_after_chunks = 0
malformed_chunk = false
rate_limit_rpm = 0
reasoning_tokens = 0
thinking_delay_ms = 0
```

### Provider Config Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `latency_ms` | int | 0 | Delay (ms) before the first byte of the response |
| `tokens` | int | 20 | Number of words to generate in the response |
| `stream_delay_ms` | int | 0 | Delay (ms) between each streaming chunk |
| `error_rate` | float | 0.0 | Probability (0.0–1.0) of returning an error instead of a response |
| `error_status` | int | 500 | HTTP status code for injected errors |
| `timeout_ms` | int | 0 | Hold the connection for this long, then TCP RST (0 = disabled) |
| `disconnect_after_chunks` | int | 0 | Sever the TCP connection after N content chunks (0 = disabled) |
| `malformed_chunk` | bool | false | Inject `{INVALID JSON CORRUPT` at the stream midpoint |
| `rate_limit_rpm` | int | 0 | Sliding-window rate limit in requests per minute (0 = disabled) |
| `reasoning_tokens` | int | 0 | Generate a thinking/reasoning block with this many tokens (0 = disabled) |
| `thinking_delay_ms` | int | 0 | Extra delay (ms) to simulate thinking time before response |
| `honor_max_tokens` | bool | false | When true, use the request body's `max_tokens` to determine output length |
| `min_tokens` | int | 1 | Minimum output tokens (floor for header/body overrides) |
| `ttft_ms` | int | 0 | Time-to-first-token delay (ms) before the first content chunk in streaming |
| `stream_delay_jitter_ms` | int | 0 | Random jitter (+/-) added to `stream_delay_ms` per chunk |
| `slow_header_ms` | int | 0 | Delay (ms) before writing response headers (simulates slow upstream) |
| `max_concurrent` | int | 0 | Maximum concurrent requests per provider (0 = unlimited). Returns 503 when exceeded |
| `sse_keepalive_interval_ms` | int | 0 | Emit `: ping` SSE comments at this interval during TTFT waits (0 = disabled) |
| `stop_reason` | string | "" | Override the emitted stop reason. Anthropic: `stop_reason` (e.g. `pause_turn`, `refusal`). OpenAI: `finish_reason` (e.g. `content_filter`). Empty = default |
| `strict` | bool | false | Contract-oracle mode (Anthropic): reject requests the real API would 400 — unknown top-level fields, missing `model`/`messages`/`max_tokens`, `temperature` outside [0,1], OpenAI tool wrappers / string `tool_choice` / OpenAI roles / `image_url` blocks, and missing required fields on common content blocks. Bounded depth (see `strict.go`); field sets mirror `CreateMessageParams` in Anthropic's OpenAPI spec |
| `cache_read_tokens` | int | 0 | Prompt-cache reads: Anthropic `usage.cache_read_input_tokens`, OpenAI `usage.prompt_tokens_details.cached_tokens` |
| `cache_creation_tokens` | int | 0 | Anthropic: add `usage.cache_creation_input_tokens` to responses (0 = omitted) |
| `fail_first_n` | int | 0 | Deterministically fail the first N requests (per provider) with `error_status`, then succeed. Counter resets on config update/reset. Deterministic alternative to `error_rate` |
| `disconnect_after_event` | string | "" | Anthropic streaming: TCP RST immediately after the named SSE event type is written (e.g. `message_delta` leaves a stop_reason but no `message_stop`; `content_block_start` cuts before any delta) |
| `emit_nonstandard_fields` | bool | false | Inject genuinely-unknown probe fields (`x_mock_unknown_field` on Anthropic message shapes / `message_delta.delta`, `x_mock_unknown_usage_field` on usage) as an unknown-field-tolerance fault. Spec-required nullable fields (`stop_details`, `usage.inference_geo`, `usage.output_tokens_details`) are always emitted regardless — they are in the pinned spec's `Message.required`/`Usage.required` |
| `suppress_ping_events` | bool | false | Anthropic streaming: omit the typed `ping` event after `message_start`. The real API sends it, but the pinned spec's `MessageStreamEvent` union has no ping arm — the built-in validator handles it via a local ping arm, so this knob is only needed for external strict-validation harnesses |
| `legacy_stream_usage` | bool | false | OpenAI streaming: restore the old mock usage shape (usage rides the finish chunk unconditionally, `stream_options.include_usage` ignored). Default emits the real API shape |
| `validate_responses` | bool? | unset | Self-validation of the surfaces covered by the vendored closure of nanollm's pinned specs: OpenAI chat-completion and Anthropic messages bodies (non-stream + each SSE `data:` payload) and provider error envelopes on every endpoint, checked before writing (see [Spec-Sync & Response Validation](#spec-sync--response-validation) — `/v1/completions`, `/v1/embeddings`, `/v1/responses`, `/v1/models` success bodies are outside the closure and not validated). Tri-state: unset defers to the `MOCKLM_VALIDATE_RESPONSES` env default; explicit `false` (e.g. via `X-MockLM-Fault`) opts a deliberate-fault request out |
| `faults` | []FaultSpec | [] | Generalized two-knob fault list (WHEN × HOW) applied to every request; composes with all knobs above. See [Two-Knob Fault Specs](#two-knob-fault-specs-when--how) |
| `attempt_faults` | [][]FaultSpec | [] | Per-attempt fault arrays: entry `i` applies only to the i-th request (0-based) seen by the provider since the last config update/reset; requests past the end get none. Generalizes `fail_first_n` — `[[{"mode":"error"}], []]` fails attempt 0 and lets attempt 1 succeed, the canonical retry/fallback scenario without header targeting |

## Request Fidelity

- **Message content** may be a plain string or an array of typed content
  blocks on both providers (Anthropic `text`/`tool_use`/`tool_result`/image
  blocks, OpenAI multi-part content). Proxy-generated tool round-trips are
  accepted, not 400'd.
- **Tool echo**: when `tool_use_response = true`, the response calls the
  *first tool offered in the request* (OpenAI `{type:"function",
  function:{name}}`, `{type:"custom", custom:{name}}`, and Anthropic
  `{name}` shapes are all recognized). Falls back to the fixed `get_weather`
  call when no tools are offered.
  - Anthropic streaming emits a real `tool_use` block with
    `input_json_delta` fragments; OpenAI streaming emits a `tool_calls`
    delta chunk. Both finish with the tool stop reason.
- **Thinking**: an Anthropic request with `thinking: {type: "enabled"}` gets
  a thinking block even when `reasoning_tokens` is 0.
- **Recording cap**: the in-memory request recording keeps the most recent
  1000 requests.

## Token Count Resolution

The number of output tokens (words) is resolved in priority order:

1. **`X-MockLM-Tokens` header** — Per-request override. Set this header to an integer to control output length for a single request.
2. **Body `max_tokens`** — Only used when `honor_max_tokens = true` in the provider config. Reads the request body's `max_tokens` (or `max_output_tokens` for the Responses API).
3. **Config `tokens`** — The default from the provider config.

All resolved values are clamped to a minimum of `min_tokens` (default: 1).

```bash
# Override output to 50 tokens for this request only
curl -s http://localhost:9999/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-MockLM-Tokens: 50" \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}]}'
```

## Fault Injection

Faults are checked in order. The first matching fault short-circuits the response.

Injected error envelopes are schema-valid per the pinned specs: Anthropic
bodies carry `type`/`error`/`request_id` with the error type mapped to the
real union member for the status (`529 → overloaded_error`, `5xx → api_error`,
`429 → rate_limit_error`, `4xx → invalid_request_error`, plus
401/403/404/504 mappings); OpenAI bodies carry `type`/`message`/`param`/`code`
(`5xx → server_error`).

### Pre-Response Faults

Checked before any content is generated:

| Priority | Fault | Trigger | Behavior |
|---|---|---|---|
| 1 | Rate limit | `rate_limit_rpm > 0` | 429 + `Retry-After` header + provider-format error JSON |
| 2 | Deterministic error | `fail_first_n > 0` (first N requests per provider) | Returns `error_status` with provider-format error JSON |
| 3 | Random error | `error_rate > 0` (random roll) | Returns `error_status` with provider-format error JSON |
| 4 | Pre-body fault specs | `faults` / `attempt_faults` entries with mode `error`, or `disconnect` without a stream WHEN | Provider-format error (`error_status`/`error_type`/`error_message`) or RST, after an optional cancel-aware `after_ms` hold |
| 5 | Timeout | `timeout_ms > 0` | Sleeps (cancel-aware), then hijacks the TCP socket and sends RST |
| 6 | Latency | `latency_ms > 0` | Sleeps (cancel-aware) before proceeding to generate the response |

All fault-path waits are cancel-aware: a client disconnect frees the handler
immediately instead of leaking it into the sleep.

### Two-Knob Fault Specs (WHEN × HOW)

`faults` (every request) and `attempt_faults` (per request index) carry
uniform fault specs with two orthogonal knobs. **WHEN** (all optional,
AND-ed; absent = first opportunity): `after_ms`, `after_bytes` (body bytes
written), `after_event` (fires right after the named Anthropic SSE event),
`after_n` (SSE frames written). **HOW** (`mode`):

| `mode` | Phase | Behavior |
|---|---|---|
| `error` | pre-body | HTTP error with `error_status`/`error_type`/`error_message` (defaults derive from the provider + config `error_status`). Cannot fire mid-stream — the status is locked once the body starts; specs that try are rejected with 400 |
| `disconnect` | pre-body or stream | TCP hijack + RST; runs mid-stream when any of `after_event`/`after_n`/`after_bytes` is set |
| `malformed_chunk` | stream | Injects `{INVALID JSON CORRUPT`; stream continues |
| `unknown_event` | stream (Anthropic) | **B1**: well-formed but off-vocabulary top-level event (`event_type`, default `message_future`), `repeat` times — same type twice in one stream is the decoder warn-once probe |
| `unknown_block` | stream (Anthropic) | **B2**: complete content block (start + stop) of `block_type` — spec-accurate shapes for the real suppressed types (`redacted_thinking`, `server_tool_use`), a generic probe otherwise |
| `stream_error` | stream (Anthropic) | **B5**: mid-stream `event: error` with `error_type`/`error_message` (default `overloaded_error`); the stream continues — compose `{"mode":"disconnect","after_event":"error"}` to cut after it |
| `stall` | stream | **A7**: stop writing mid-stream and hold the connection open — no bytes, no close — until the client disconnects. Fires after the first frame when no WHEN is set |
| `non_json_body` | pre-body | **C9**: a 200 with a `text/html` error-page body instead of JSON (the classic intermediary-proxy failure). Pre-body only; stream WHENs are rejected with 400 |

`error`-mode specs additionally accept `retry_after`, set verbatim on the
`Retry-After` header — numeric seconds or an HTTP-date (**C2**: the date
form must be ignored by seconds-only clients, falling back to their own
backoff).

Specs fire at most once per request and compose: later specs can match
frames injected by earlier ones. The decoder-fault modes emit well-formed
payloads that are **off the pinned `MessageStreamEvent` union**, so
scenarios driving them must set `validate_responses: false` — with
validation on, the self-validator severs the stream at the injected frame.

### Usage Faults (`usage_fault`, OpenAI chat)

The `usage_fault` provider knob distorts the OpenAI chat usage surface —
the B-OAI-8 shape-coupled-extraction probes. Anthropic usage is
spec-required and not covered.

| Value | Fault |
|---|---|
| `omit` | **D1**: no usage key anywhere — the non-stream response and the entire stream, even when `stream_options.include_usage` was requested. Spec-valid (usage is optional in the pinned response root), so it composes with `validate_responses` |
| `partial` | **D2**: every emitted usage object carries `prompt_tokens` **only** (no `completion_tokens`/`total_tokens`/`*_details`). Off-spec — `CompletionUsage` requires all three — so scenarios must set `validate_responses: false` |
| `trailer` | **D3**: force the real `include_usage` wire shape (`usage: null` on every chunk + a trailing `choices: []` usage chunk) even when the request didn't set `stream_options.include_usage` |

### Deterministic SSE Transport Faults (A2/A3)

Provider knobs that alter how SSE frames hit the wire without changing
their payloads (the re-framing/robustness probes; applied on the chat and
messages streams):

| Knob | Fault |
|---|---|
| `fragment_offset` | **A2**: every frame is flushed in two writes split at this byte offset (frames shorter than the offset go out whole) |
| `fragment_split` | `"rune"` cuts one byte into the frame's first multibyte UTF-8 sequence (pure-ASCII frames fall back to `fragment_offset`); `"event"` cuts right after the first line — between the `event:` and `data:` lines of an Anthropic frame |
| `fragment_delay_ms` | pause between the two fragment writes so the boundary survives kernel buffering and arrives as two reads (default 5 when fragmenting) |
| `crlf_frames` | **A3**: every SSE line ending becomes `\r\n` — valid per the SSE spec, hostile to naive `\n\n` re-framers |
| `coalesce_frames` | buffer N frames into a single write+flush so one TCP chunk carries several frames (composes with `crlf_frames`; mutually exclusive with fragmentation — coalescing wins; the buffered tail flushes at stream end) |
| `content_text` | emit this text verbatim as the response content (words split on whitespace, one per stream delta, no capitalize/period decoration) — lets a scenario stream known bytes, e.g. multibyte runes for `fragment_split: "rune"` |

```bash
# Attempt 0 → 503, attempt 1 → success (retry test, no header needed):
curl -s http://localhost:9999/admin/config -X PUT -d '{"openai":{
  "attempt_faults":[[{"mode":"error","error_status":503}],[]]},"anthropic":{}}'

# Mid-stream overloaded error after 3 frames, then the stream dies:
curl -s http://localhost:9999/admin/config -X PUT -d '{"anthropic":{
  "validate_responses":false,
  "faults":[{"mode":"stream_error","error_type":"overloaded_error","after_n":3},
            {"mode":"disconnect","after_event":"error"}]},"openai":{}}'
```

### Request-Count Introspection

`GET /admin/request-count` returns `{"openai": N, "anthropic": M}` — the
per-provider request counts since the last reset/config update (the same
counter that indexes `fail_first_n` and `attempt_faults`). Every request
that reaches fault processing counts, including ones that end up
rate-limited or failed, so a test can assert exactly how many attempts a
proxy made. `POST /admin/request-count/reset` zeroes it; config updates and
`/admin/reset` do too.

### Per-Request Fault Targeting: `X-MockLM-Fault` header

Any provider-config knob can be overridden for a single request by sending a
JSON fragment in the `X-MockLM-Fault` header (mirrors `X-MockLM-Tokens`).
Keys absent from the JSON keep their configured values; invalid JSON returns
400. This makes faults hermetic — one test can mix healthy and faulty
requests against the same server:

```bash
# This request fails with 503; concurrent requests without the header succeed
curl -s http://localhost:9999/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H 'X-MockLM-Fault: {"error_rate":1.0,"error_status":503}' \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}]}'
```

Note: `rate_limit_rpm` and `max_concurrent` are inherently cross-request and
cannot be targeted per request; the `fail_first_n` counter is shared per
provider between header- and config-driven use.

### Mid-Stream Faults

Checked per-chunk during streaming:

| Fault | Trigger | Behavior |
|---|---|---|
| Disconnect | `disconnect_after_chunks > 0` | TCP hijack + RST after N content chunks. No `[DONE]` sentinel. |
| Event-scheduled disconnect | `disconnect_after_event = "message_delta"` (Anthropic) | TCP hijack + RST immediately after the named event type is written — cuts *between* protocol events rather than by word count |
| Malformed chunk | `malformed_chunk = true` | Injects `{INVALID JSON CORRUPT` at chunk `totalChunks/2`. Stream **continues** after the bad chunk. |
| Fault specs | stream-phase `faults` / `attempt_faults` entries | The generalized engine: disconnects, malformed frames, and the decoder-fault injections (`unknown_event` / `unknown_block` / `stream_error`) scheduled by `after_event` / `after_n` / `after_bytes` / `after_ms` (see [Two-Knob Fault Specs](#two-knob-fault-specs-when--how)) |

### Examples

**100% error rate returning 503:**

```bash
curl -s http://localhost:9999/admin/config \
  -X PUT -H "Content-Type: application/json" \
  -d '{"openai":{"error_rate":1.0,"error_status":503,"tokens":20},"anthropic":{"tokens":20}}'
```

**Disconnect after 3 streaming chunks:**

```bash
curl -s http://localhost:9999/admin/config \
  -X PUT -H "Content-Type: application/json" \
  -d '{"openai":{"disconnect_after_chunks":3,"tokens":20},"anthropic":{"tokens":20}}'
```

**Simulate a 5-second connection timeout:**

```bash
curl -s http://localhost:9999/admin/config \
  -X PUT -H "Content-Type: application/json" \
  -d '{"openai":{"timeout_ms":5000,"tokens":20},"anthropic":{"tokens":20}}'
```

## Presets

18 built-in named configurations for common test scenarios:

| Preset | Description |
|---|---|
| `healthy` | No faults, happy path |
| `openai-disconnect` | OpenAI drops after 3 chunks; Anthropic healthy |
| `anthropic-disconnect` | Anthropic drops after 3 chunks; OpenAI healthy |
| `openai-errors` | OpenAI returns 503 always (error_rate=1.0) |
| `anthropic-errors` | Anthropic returns 529 always |
| `openai-rate-limited` | OpenAI at 1 RPM; Anthropic healthy |
| `both-rate-limited` | Both providers at 2 RPM |
| `openai-slow` | OpenAI 500ms latency + 200ms inter-chunk delay |
| `openai-timeout` | OpenAI holds 5s then TCP RST |
| `malformed-streams` | Both providers inject corrupt JSON mid-stream |
| `flaky-openai` | OpenAI 50% error rate (500) |
| `deterministic-anthropic` | Deterministic Anthropic text response (fixed content, fixed ID) |
| `deterministic-anthropic-stream` | Deterministic Anthropic streaming (fixed content, fixed ID, no delay) |
| `deterministic-anthropic-tool-use` | Deterministic Anthropic response with tool_use content blocks |
| `bench-small` | Benchmark: small responses, zero latency |
| `bench-large` | Benchmark: large responses, zero latency |
| `bench-realistic` | Benchmark: realistic latency profile |
| `connection-pressure` | Connection pressure: limited concurrency with slow headers |

Activate a preset:

```bash
curl -s -X PUT http://localhost:9999/admin/preset/openai-errors
```

## Admin API

All admin endpoints return JSON.

### `GET /admin/config`

Returns the current provider configs and active preset name.

```bash
curl -s http://localhost:9999/admin/config
```

```json
{
  "active_preset": "healthy",
  "openai": {"latency_ms": 0, "tokens": 20, "stream_delay_ms": 0, "error_rate": 0, ...},
  "anthropic": {"latency_ms": 0, "tokens": 20, "stream_delay_ms": 0, "error_rate": 0, ...}
}
```

### `PUT /admin/config`

Replace both provider configs. Marks the active preset as `"custom"`.

```bash
curl -s -X PUT http://localhost:9999/admin/config \
  -H "Content-Type: application/json" \
  -d '{
    "openai": {"tokens": 42, "error_rate": 0.5, "error_status": 503},
    "anthropic": {"tokens": 30}
  }'
```

```json
{"status": "updated", "active_preset": "custom"}
```

### `PUT /admin/preset/{name}`

Activate a named preset.

```bash
curl -s -X PUT http://localhost:9999/admin/preset/openai-disconnect
```

```json
{"status": "activated", "active_preset": "openai-disconnect", "description": "OpenAI drops after 3 chunks; Anthropic healthy (stream resume)"}
```

Returns 404 for unknown presets.

### `POST /admin/reset`

Revert to the config that was loaded at startup.

```bash
curl -s -X POST http://localhost:9999/admin/reset
```

```json
{"status": "reset", "active_preset": ""}
```

### `GET /admin/presets`

List all available presets.

```bash
curl -s http://localhost:9999/admin/presets
```

```json
{
  "presets": [
    {"name": "healthy", "description": "No faults, happy path"},
    {"name": "openai-disconnect", "description": "OpenAI drops after 3 chunks; Anthropic healthy (stream resume)"},
    ...
  ]
}
```

## Streaming Fidelity

### OpenAI Chat Completions (`/v1/chat/completions`)

SSE format: `data: {json}\n\n`, terminated by `data: [DONE]\n\n`.

```
data: {"id":"chatcmpl-mock-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-mock-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Quantum "},"finish_reason":null}]}

data: {"id":"chatcmpl-mock-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"neural."},"finish_reason":null}]}

data: {"id":"chatcmpl-mock-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

Usage follows the real API's `stream_options.include_usage` contract: without
it, no chunk carries a `usage` key; with it, every chunk carries
`"usage": null` and a final trailing chunk with `"choices": []` carries the
usage object (before `[DONE]`). Set `legacy_stream_usage = true` to restore
the old mock shape (usage on the finish chunk, unconditionally).

### Anthropic Messages (`/v1/messages`)

SSE format: `event: {type}\ndata: {json}\n\n`.

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_mock_...","role":"assistant","content":[],"model":"claude-3-haiku-20240307",...}}

event: ping
data: {"type":"ping"}    (real-API behavior; suppress_ping_events omits it — no ping arm in the pinned MessageStreamEvent union)

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Quantum "}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":20}}

event: message_stop
data: {"type":"message_stop"}
```

When `reasoning_tokens > 0`, a thinking block (`content_block_start` with `type: "thinking"` + thinking deltas + `content_block_stop`) is emitted before the text block.

### OpenAI Responses API (`/v1/responses`)

SSE format: `event: {type}\ndata: {json}\n\n`.

```
event: response.created
event: response.output_item.added
event: response.content_part.added
event: response.output_text.delta    (repeated per word)
event: response.output_text.done
event: response.content_part.done
event: response.output_item.done
event: response.completed
```

## Spec-Sync & Response Validation

go-mocklm vendors the **response-side JSON-Schema closure** of the same
sha256-pinned OpenAPI specs nanollm's Rust oracle types are generated from
(`../nanollm/spec/*.json`):

- `spec/openai-responses.schema.json` — 13 schemas reachable from
  `CreateChatCompletionResponse`, `CreateChatCompletionStreamResponse`,
  `ErrorResponse`
- `spec/anthropic-responses.schema.json` — 82 schemas reachable from
  `Message`, `MessageStreamEvent`, `ErrorResponse`
- `spec/pins.json` — sha256 of the source specs at extraction time

Regenerate with `go run ./cmd/specsync` (or `go generate ./...`). The
extractor ports exactly two normalization rules from nanollm's
`xtask/src/main.rs`: the OpenAI `nullable: true` → null-union rewrite
(plus a null `enum` value where an enum co-resides, since `enum` asserts
independently under real validation), and `additionalProperties: false`
injection on the two OpenAI response roots (without it, OpenAI response
validation is unknown-field-blind). `spec_sync_test.go` fails when the
recorded pins diverge from `../nanollm/spec` or when the vendored files
drift from a fresh extraction.

With `validate_responses` on (knob or `MOCKLM_VALIDATE_RESPONSES=1` env
default), every body on the closure's surfaces is validated (full JSON
Schema 2020-12 via `santhosh-tekuri/jsonschema`) before writing:

| Surface | Schema root |
|---|---|
| OpenAI chat non-stream | `CreateChatCompletionResponse` |
| OpenAI chat SSE payloads (except `[DONE]`) | `CreateChatCompletionStreamResponse` |
| Anthropic non-stream | `Message` |
| Anthropic SSE payloads | `MessageStreamEvent` (typed `ping` via a local hand-written arm) |
| Error envelopes, incl. injected faults | provider `ErrorResponse` |

Violations fail loudly: **500** with a spec-shaped `api_error`/`server_error`
envelope before headers, **TCP RST** mid-stream (the violating frame never
reaches the wire), or **panic** when `MOCKLM_VALIDATE_PANIC=1`. Deliberately
off-spec output bypasses validation by design: `malformed_chunk` frames,
`emit_nonstandard_fields` probe bodies, and any request opted out with
`X-MockLM-Fault: {"validate_responses": false}`.

**Not validated** (success bodies outside the extracted closure — the
closure roots mirror nanollm's oracle roots, which never covered these
surfaces): legacy `/v1/completions`, `/v1/embeddings`, `/v1/responses`,
and `/v1/models`. Their error envelopes ARE validated. Extending coverage
would first require fixing the known shape drift on those surfaces
(proposal G5/G6/G7 — Responses API completion is explicitly deferred to
nanollm's R5 work).

The runtime accept/reject surface intentionally differs from nanollm's
generated Rust deserializers in three documented ways: spec `pattern`
assertions are retained (typify strips them, so the validator is stricter
there); the typed `ping` event is accepted via the local arm (the pinned
`MessageStreamEvent` union rejects it); and `nullable: true` enums gain a
`null` enum value (typify instead renders `Option<Enum>`, but `enum`
asserts independently under real JSON-Schema validation). Same pinned
bytes, same roots, deliberate deltas.

Event *ordering* (message_start → content blocks → message_delta →
message_stop) is invisible to schema validation — `contract_test.go`
pins it with a hand-written table-driven state machine.

## Using with nanollm

nanollm's integration test suite (`tests/integration.rs` + `tests/common/mod.rs`) spawns go-mocklm as a subprocess:

1. **Binary location**: Set `MOCKLM_BINARY` env var, or it defaults to `../mocklm/mocklm`.
2. **Per-test isolation**: Each test allocates a free port, writes a temp `config.toml`, and spawns a dedicated mocklm process.
3. **Health polling**: The harness polls `GET /health` every 50ms with a 10s deadline before proceeding.
4. **Runtime config**: Tests call `set_preset()` or `set_config()` via the admin API to change behavior mid-test.
5. **Self-validation**: The harness sets `MOCKLM_VALIDATE_RESPONSES=1`, so every chat-completion body, messages body, and error envelope the mock emits in nanollm's integration tests is checked against the pinned-spec closure — drift on those surfaces fails loudly instead of silently distorting nanollm's coverage. (`/v1/completions`, `/v1/embeddings`, `/v1/responses`, `/v1/models` success bodies are outside the closure and not covered.)
6. **Cleanup**: The `Drop` impl kills the process when the test finishes.

```bash
# Build mocklm, then run nanollm integration tests
cd go-mocklm && go build -o mocklm
cd ../nanollm && MOCKLM_BINARY=../go-mocklm/mocklm cargo test --test integration
```

## Known Limitations

| Gap | Impact |
|---|---|
| **No tool/function call emulation** | Can't test tool_calls proxy path |
| **No `finish_reason` variations** | Always returns `"stop"` / `"end_turn"` — no `"length"`, `"content_filter"`, `"tool_calls"` |
| **No configurable response content** | Content is random words from a fixed vocabulary; can't return specific text for transform testing |
| ~~**No per-request fault injection**~~ | ~~Faults are global config; can't target a single request via header~~ (resolved: `X-MockLM-Fault` header overlays any provider knob per request) |
| ~~**No request recording**~~ | ~~No way to inspect what requests mocklm received~~ (resolved: all providers now record requests via `/admin/requests`) |
| ~~**No mid-stream error events**~~ | ~~Can disconnect or inject malformed JSON, but can't inject a proper error JSON event mid-stream~~ (resolved: `{"mode":"stream_error"}` fault specs inject a well-formed `event: error` at any WHEN) |
| **No Responses API reasoning streaming** | Reasoning items only appear in the final `response.completed` event, not streamed incrementally |
| **No provider-specific rate limit headers** | Only sets `Retry-After`; missing `x-ratelimit-*` (OpenAI) and `anthropic-ratelimit-*` headers |
| ~~**No SSE keep-alive/ping events**~~ | ~~Real providers send `: ping` comment lines~~ (resolved: `sse_keepalive_interval_ms` emits pings during TTFT waits) |
| ~~**No `max_tokens` enforcement**~~ | ~~Always generates exactly `config.tokens` words~~ (resolved: `honor_max_tokens` + `X-MockLM-Tokens` header) |
| **No multi-modal content** | Text only — no image or audio content blocks |
| **No Google/Gemini provider** | Only OpenAI and Anthropic |
