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

## Fault Injection

Faults are checked in order. The first matching fault short-circuits the response.

### Pre-Response Faults

Checked before any content is generated:

| Priority | Fault | Trigger | Behavior |
|---|---|---|---|
| 1 | Rate limit | `rate_limit_rpm > 0` | 429 + `Retry-After` header + provider-format error JSON |
| 2 | Random error | `error_rate > 0` (random roll) | Returns `error_status` with provider-format error JSON |
| 3 | Timeout | `timeout_ms > 0` | Sleeps, then hijacks the TCP socket and sends RST |
| 4 | Latency | `latency_ms > 0` | Sleeps before proceeding to generate the response |

### Mid-Stream Faults

Checked per-chunk during streaming:

| Fault | Trigger | Behavior |
|---|---|---|
| Disconnect | `disconnect_after_chunks > 0` | TCP hijack + RST after N content chunks. No `[DONE]` sentinel. |
| Malformed chunk | `malformed_chunk = true` | Injects `{INVALID JSON CORRUPT` at chunk `totalChunks/2`. Stream **continues** after the bad chunk. |

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

11 built-in named configurations for common test scenarios:

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

data: {"id":"chatcmpl-mock-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{...}}

data: [DONE]
```

### Anthropic Messages (`/v1/messages`)

SSE format: `event: {type}\ndata: {json}\n\n`.

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_mock_...","role":"assistant","content":[],"model":"claude-3-haiku-20240307",...}}

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

## Using with nanollm

nanollm's integration test suite (`tests/integration.rs` + `tests/common/mod.rs`) spawns go-mocklm as a subprocess:

1. **Binary location**: Set `MOCKLM_BINARY` env var, or it defaults to `../mocklm/mocklm`.
2. **Per-test isolation**: Each test allocates a free port, writes a temp `config.toml`, and spawns a dedicated mocklm process.
3. **Health polling**: The harness polls `GET /health` every 50ms with a 10s deadline before proceeding.
4. **Runtime config**: Tests call `set_preset()` or `set_config()` via the admin API to change behavior mid-test.
5. **Cleanup**: The `Drop` impl kills the process when the test finishes.

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
| **No per-request fault injection** | Faults are global config; can't target a single request via header |
| **No request recording** | No way to inspect what requests mocklm received (useful for verifying transformed requests) |
| **No mid-stream error events** | Can disconnect or inject malformed JSON, but can't inject a proper error JSON event mid-stream |
| **No Responses API reasoning streaming** | Reasoning items only appear in the final `response.completed` event, not streamed incrementally |
| **No provider-specific rate limit headers** | Only sets `Retry-After`; missing `x-ratelimit-*` (OpenAI) and `anthropic-ratelimit-*` headers |
| **No SSE keep-alive/ping events** | Real providers send `: ping` comment lines as heartbeats |
| **No `max_tokens` enforcement** | Always generates exactly `config.tokens` words regardless of the request's `max_tokens` |
| **No multi-modal content** | Text only — no image or audio content blocks |
| **No Google/Gemini provider** | Only OpenAI and Anthropic |
