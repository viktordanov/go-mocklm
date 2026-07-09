package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	awseventstream "github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// --- Phase 3c: Bedrock Converse/ConverseStream — eventstream framing,
// SDK-decode to the typed ConverseStreamOutput union (K10), both error
// channels, third-provider counters (R2), scenario exact output as the 4th
// emit path (K15). All hermetic: the aws-sdk-go-v2 client talks to the
// in-process test server, never live AWS. ---

const bedrockTestModel = "anthropic.claude-3-5-sonnet-20240620-v1:0"

// bedrockClient builds a real bedrockruntime client aimed at the mock —
// static dummy credentials (the mock ignores SigV4), retries off so attempt
// counters stay predictable.
func bedrockClient(srvURL string) *bedrockruntime.Client {
	return bedrockruntime.New(bedrockruntime.Options{
		Region:       "us-east-1",
		BaseEndpoint: aws.String(srvURL),
		Credentials: aws.CredentialsProviderFunc(func(context.Context) (aws.Credentials, error) {
			return aws.Credentials{AccessKeyID: "mock", SecretAccessKey: "mock"}, nil
		}),
		RetryMaxAttempts: 1,
	})
}

// bedrockClientWithHeaders is bedrockClient plus fixed headers on every
// request — how SDK traffic carries X-MockLM-* knobs. The injection happens
// after SigV4 signing, which is fine: the mock ignores signatures.
func bedrockClientWithHeaders(srvURL string, headers map[string]string) *bedrockruntime.Client {
	return bedrockruntime.New(bedrockruntime.Options{
		Region:       "us-east-1",
		BaseEndpoint: aws.String(srvURL),
		Credentials: aws.CredentialsProviderFunc(func(context.Context) (aws.Credentials, error) {
			return aws.Credentials{AccessKeyID: "mock", SecretAccessKey: "mock"}, nil
		}),
		RetryMaxAttempts: 1,
		HTTPClient:       headerInjectingClient{headers: headers},
	})
}

type headerInjectingClient struct {
	headers map[string]string
}

func (c headerInjectingClient) Do(req *http.Request) (*http.Response, error) {
	for k, v := range c.headers {
		req.Header.Set(k, v)
	}
	return http.DefaultClient.Do(req)
}

func converseInput(model string) *bedrockruntime.ConverseInput {
	return &bedrockruntime.ConverseInput{
		ModelId: aws.String(model),
		Messages: []types.Message{
			{
				Role: types.ConversationRoleUser,
				Content: []types.ContentBlock{
					&types.ContentBlockMemberText{Value: "Hello mock"},
				},
			},
		},
	}
}

// --- eventstream framing (encoder correctness below the SDK) ---

func TestEventStreamEncoderFraming(t *testing.T) {
	payload := []byte(`{"contentBlockIndex":0,"delta":{"text":"hé🎉"}}`)
	frame := encodeEventStreamMessage([]eventStreamHeader{
		{":event-type", "contentBlockDelta"},
		{":content-type", "application/json"},
		{":message-type", "event"},
	}, payload)

	// The AWS decoder must accept the frame (it verifies both CRCs) and
	// hand back our headers and payload byte-for-byte.
	msg, err := awseventstream.NewDecoder().Decode(bytes.NewReader(frame), nil)
	if err != nil {
		t.Fatalf("AWS eventstream decoder rejected our frame: %v", err)
	}
	if got := msg.Headers.Get(":event-type"); got == nil || got.String() != "contentBlockDelta" {
		t.Fatalf(":event-type header = %v", got)
	}
	if got := msg.Headers.Get(":message-type"); got == nil || got.String() != "event" {
		t.Fatalf(":message-type header = %v", got)
	}
	if got := msg.Headers.Get(":content-type"); got == nil || got.String() != "application/json" {
		t.Fatalf(":content-type header = %v", got)
	}
	if !bytes.Equal(msg.Payload, payload) {
		t.Fatalf("payload = %q, want %q", msg.Payload, payload)
	}

	// A corrupt frame (flipped message CRC) must be rejected — the
	// malformed_chunk dialect.
	bad := make([]byte, len(frame))
	copy(bad, frame)
	bad[len(bad)-1] ^= 0xFF
	if _, err := awseventstream.NewDecoder().Decode(bytes.NewReader(bad), nil); err == nil {
		t.Fatalf("decoder accepted a frame with a broken message CRC")
	}

	// Back-to-back frames decode in sequence (stream framing, not just a
	// single message).
	second := encodeEventStreamMessage([]eventStreamHeader{
		{":event-type", "messageStop"},
		{":content-type", "application/json"},
		{":message-type", "event"},
	}, []byte(`{"stopReason":"end_turn"}`))
	r := bytes.NewReader(append(append([]byte{}, frame...), second...))
	dec := awseventstream.NewDecoder()
	for i, wantEvent := range []string{"contentBlockDelta", "messageStop"} {
		msg, err := dec.Decode(r, nil)
		if err != nil {
			t.Fatalf("frame %d failed to decode: %v", i, err)
		}
		if got := msg.Headers.Get(":event-type").String(); got != wantEvent {
			t.Fatalf("frame %d event = %q, want %q", i, got, wantEvent)
		}
	}
}

// --- SDK decode: non-stream Converse ---

func TestBedrockConverseSDKDecode(t *testing.T) {
	cfg := defaultConfig()
	cfg.Bedrock.Tokens = 5
	srv := testServer(cfg)
	defer srv.Close()

	out, err := bedrockClient(srv.URL).Converse(context.Background(), converseInput(bedrockTestModel))
	if err != nil {
		t.Fatalf("Converse failed: %v", err)
	}

	msgOut, ok := out.Output.(*types.ConverseOutputMemberMessage)
	if !ok {
		t.Fatalf("output union = %T, want ConverseOutputMemberMessage", out.Output)
	}
	if msgOut.Value.Role != types.ConversationRoleAssistant {
		t.Fatalf("role = %q, want assistant", msgOut.Value.Role)
	}
	if len(msgOut.Value.Content) != 1 {
		t.Fatalf("content blocks = %d, want 1", len(msgOut.Value.Content))
	}
	text, ok := msgOut.Value.Content[0].(*types.ContentBlockMemberText)
	if !ok || text.Value == "" {
		t.Fatalf("content[0] = %T (%v), want non-empty text block", msgOut.Value.Content[0], msgOut.Value.Content[0])
	}
	if out.StopReason != types.StopReasonEndTurn {
		t.Fatalf("stopReason = %q, want end_turn", out.StopReason)
	}
	if out.Usage == nil || *out.Usage.OutputTokens < 1 || *out.Usage.TotalTokens != *out.Usage.InputTokens+*out.Usage.OutputTokens {
		t.Fatalf("usage = %+v, want consistent inputTokens/outputTokens/totalTokens", out.Usage)
	}
	if out.Metrics == nil || *out.Metrics.LatencyMs < 1 {
		t.Fatalf("metrics = %+v, want latencyMs >= 1", out.Metrics)
	}
}

// --- SDK decode: ConverseStream to the typed union (K10), exact output as
// the 4th emit path (K15) ---

const bedrockExactProbe = "Exact  bedrock\n\npayload — ünïcode 你好 🎉 with  runs "

func TestBedrockConverseStreamSDKDecodeExactOutput(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	// Register an exact-output scenario keyed by (bedrock, model-from-path).
	reg, _ := json.Marshal(map[string]any{
		"id": "bedrock-exact", "provider": "bedrock", "model": bedrockTestModel,
		"output": map[string]any{
			"text":          bedrockExactProbe,
			"chunking":      map[string]any{"mode": "runes", "size": 4},
			"output_tokens": 21,
		},
	})
	mustRegisterScenario(t, srv.URL, string(reg))

	stream, err := bedrockClient(srv.URL).ConverseStream(context.Background(), &bedrockruntime.ConverseStreamInput{
		ModelId: aws.String(bedrockTestModel),
		Messages: []types.Message{
			{
				Role:    types.ConversationRoleUser,
				Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: "Hello"}},
			},
		},
	})
	if err != nil {
		t.Fatalf("ConverseStream failed: %v", err)
	}
	es := stream.GetStream()
	defer es.Close()

	var (
		order     []string
		text      strings.Builder
		deltaN    int
		stopSeen  types.StopReason
		usageSeen *types.TokenUsage
	)
	for ev := range es.Events() {
		switch v := ev.(type) {
		case *types.ConverseStreamOutputMemberMessageStart:
			order = append(order, "messageStart")
			if v.Value.Role != types.ConversationRoleAssistant {
				t.Fatalf("messageStart role = %q, want assistant", v.Value.Role)
			}
		case *types.ConverseStreamOutputMemberContentBlockDelta:
			if len(order) == 0 || order[len(order)-1] != "contentBlockDelta" {
				order = append(order, "contentBlockDelta")
			}
			if v.Value.ContentBlockIndex == nil || *v.Value.ContentBlockIndex != 0 {
				t.Fatalf("contentBlockIndex = %v, want 0 (K10: exact member casing)", v.Value.ContentBlockIndex)
			}
			d, ok := v.Value.Delta.(*types.ContentBlockDeltaMemberText)
			if !ok {
				t.Fatalf("delta union = %T, want ContentBlockDeltaMemberText", v.Value.Delta)
			}
			text.WriteString(d.Value)
			deltaN++
		case *types.ConverseStreamOutputMemberContentBlockStop:
			order = append(order, "contentBlockStop")
		case *types.ConverseStreamOutputMemberMessageStop:
			order = append(order, "messageStop")
			stopSeen = v.Value.StopReason
		case *types.ConverseStreamOutputMemberMetadata:
			order = append(order, "metadata")
			usageSeen = v.Value.Usage
		default:
			t.Fatalf("unexpected union member %T", ev)
		}
	}
	if err := es.Err(); err != nil {
		t.Fatalf("stream error: %v", err)
	}

	wantOrder := []string{"messageStart", "contentBlockDelta", "contentBlockStop", "messageStop", "metadata"}
	if strings.Join(order, ",") != strings.Join(wantOrder, ",") {
		t.Fatalf("event order = %v, want %v", order, wantOrder)
	}
	if got := text.String(); got != bedrockExactProbe {
		t.Fatalf("reassembled text = %q, want the exact probe %q", got, bedrockExactProbe)
	}
	if wantChunks := len(chunkExact(bedrockExactProbe, Chunking{Mode: "runes", Size: 4})); deltaN != wantChunks {
		t.Fatalf("stream carried %d deltas, want %d (runes/4 slicing)", deltaN, wantChunks)
	}
	if stopSeen != types.StopReasonEndTurn {
		t.Fatalf("stopReason = %q, want end_turn", stopSeen)
	}
	if usageSeen == nil || *usageSeen.OutputTokens != 21 {
		t.Fatalf("usage = %+v, want the pinned outputTokens 21 (R9)", usageSeen)
	}

	// The scenario captured the SDK-signed request body byte-exact.
	var rc struct {
		Count int64 `json:"count"`
	}
	getJSON(t, srv.URL+"/admin/scenarios/bedrock-exact/request-count", &rc)
	if rc.Count != 1 {
		t.Fatalf("scenario request-count = %d, want 1", rc.Count)
	}
}

// --- both error channels ---

func TestBedrockHTTPErrorChannel(t *testing.T) {
	// Pre-body fault "error" 429 → HTTP envelope with X-Amzn-ErrorType:
	// ThrottlingException → the SDK returns the modeled typed error.
	cfg := defaultConfig()
	cfg.Bedrock.Faults = []FaultSpec{{Mode: "error", ErrorStatus: 429}}
	srv := testServer(cfg)
	defer srv.Close()

	_, err := bedrockClient(srv.URL).Converse(context.Background(), converseInput(bedrockTestModel))
	var throttled *types.ThrottlingException
	if !errors.As(err, &throttled) {
		t.Fatalf("Converse error = %v (%T), want types.ThrottlingException", err, err)
	}
}

func TestBedrockBadFaultHeaderTypedError(t *testing.T) {
	// A malformed X-MockLM-Fault header must 400 with the Bedrock-modeled
	// ValidationException in X-Amzn-ErrorType — the AWS restjson dispatch
	// matches no shape for invalid_request_error and would degrade the SDK
	// error to a generic one.
	srv := testServer(defaultConfig())
	defer srv.Close()

	client := bedrockClientWithHeaders(srv.URL, map[string]string{"X-MockLM-Fault": `{not json`})
	_, err := client.Converse(context.Background(), converseInput(bedrockTestModel))
	var validation *types.ValidationException
	if !errors.As(err, &validation) {
		t.Fatalf("Converse error = %v (%T), want types.ValidationException", err, err)
	}
}

func TestBedrockRateLimitTypedError(t *testing.T) {
	// The provider-global limiter's 429 must carry the same Bedrock-modeled
	// ThrottlingException as the fault-injected 429 path.
	cfg := defaultConfig()
	cfg.Bedrock.RateLimitRPM = 1
	srv := testServer(cfg)
	defer srv.Close()

	client := bedrockClient(srv.URL)
	if _, err := client.Converse(context.Background(), converseInput(bedrockTestModel)); err != nil {
		t.Fatalf("first request should pass the limiter: %v", err)
	}
	_, err := client.Converse(context.Background(), converseInput(bedrockTestModel))
	var throttled *types.ThrottlingException
	if !errors.As(err, &throttled) {
		t.Fatalf("limiter-tripped 429 error = %v (%T), want types.ThrottlingException", err, err)
	}
}

func TestBedrockInBandStreamException(t *testing.T) {
	// stream_error after the second frame → an in-band eventstream
	// exception (:message-type=exception, modelStreamErrorException); the
	// SDK yields the leading events, then the typed error, then the stream
	// terminates (AWS semantics — unlike the SSE dialect, which continues).
	cfg := defaultConfig()
	cfg.Bedrock.Tokens = 5
	cfg.Bedrock.Faults = []FaultSpec{{Mode: "stream_error", AfterN: 2, ErrorMessage: "mock mid-stream failure"}}
	srv := testServer(cfg)
	defer srv.Close()

	stream, err := bedrockClient(srv.URL).ConverseStream(context.Background(), &bedrockruntime.ConverseStreamInput{
		ModelId: aws.String(bedrockTestModel),
		Messages: []types.Message{
			{
				Role:    types.ConversationRoleUser,
				Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: "Hello"}},
			},
		},
	})
	if err != nil {
		t.Fatalf("ConverseStream failed: %v", err)
	}
	es := stream.GetStream()
	defer es.Close()

	events := 0
	for range es.Events() {
		events++
	}
	var streamErr *types.ModelStreamErrorException
	if err := es.Err(); !errors.As(err, &streamErr) {
		t.Fatalf("stream error = %v (%T), want types.ModelStreamErrorException", err, err)
	}
	if !strings.Contains(streamErr.ErrorMessage(), "mock mid-stream failure") {
		t.Fatalf("exception message = %q", streamErr.ErrorMessage())
	}
	if events < 2 {
		t.Fatalf("only %d events arrived before the exception, want the 2 leading frames", events)
	}
}

// --- Bedrock-dialect decoder probes (R3) ---

func TestBedrockDecoderProbes(t *testing.T) {
	// unknown_event: an unknown :event-type frame decodes to
	// UnknownUnionMember with our tag. unknown_block: a contentBlockStart
	// whose start union carries an unknown member.
	cfg := defaultConfig()
	cfg.Bedrock.Tokens = 3
	cfg.Bedrock.Faults = []FaultSpec{
		{Mode: "unknown_event", AfterN: 1},
		{Mode: "unknown_block", AfterN: 2},
	}
	srv := testServer(cfg)
	defer srv.Close()

	stream, err := bedrockClient(srv.URL).ConverseStream(context.Background(), &bedrockruntime.ConverseStreamInput{
		ModelId: aws.String(bedrockTestModel),
		Messages: []types.Message{
			{
				Role:    types.ConversationRoleUser,
				Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: "Hello"}},
			},
		},
	})
	if err != nil {
		t.Fatalf("ConverseStream failed: %v", err)
	}
	es := stream.GetStream()
	defer es.Close()

	sawUnknownEvent := false
	sawUnknownBlockStart := false
	for ev := range es.Events() {
		switch v := ev.(type) {
		case *types.UnknownUnionMember:
			if v.Tag == "messageFuture" {
				sawUnknownEvent = true
			}
		case *types.ConverseStreamOutputMemberContentBlockStart:
			if _, unknown := v.Value.Start.(*types.UnknownUnionMember); unknown {
				sawUnknownBlockStart = true
			}
		}
	}
	if err := es.Err(); err != nil {
		t.Fatalf("stream error: %v", err)
	}
	if !sawUnknownEvent {
		t.Fatalf("the unknown_event probe never surfaced as an UnknownUnionMember frame")
	}
	if !sawUnknownBlockStart {
		t.Fatalf("the unknown_block probe never surfaced as a contentBlockStart with an unknown start member")
	}
}

func TestBedrockMalformedChunkBreaksCRC(t *testing.T) {
	cfg := defaultConfig()
	cfg.Bedrock.Tokens = 4
	cfg.Bedrock.Faults = []FaultSpec{{Mode: "malformed_chunk", AfterN: 2}}
	srv := testServer(cfg)
	defer srv.Close()

	stream, err := bedrockClient(srv.URL).ConverseStream(context.Background(), &bedrockruntime.ConverseStreamInput{
		ModelId: aws.String(bedrockTestModel),
		Messages: []types.Message{
			{
				Role:    types.ConversationRoleUser,
				Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: "Hello"}},
			},
		},
	})
	if err != nil {
		t.Fatalf("ConverseStream failed: %v", err)
	}
	es := stream.GetStream()
	defer es.Close()
	for range es.Events() {
	}
	if err := es.Err(); err == nil {
		t.Fatalf("a corrupt-CRC frame slipped through the SDK decoder without error")
	}
}

// --- third-provider plumbing: counters do not alias (R2), fail_first_n,
// provider binding ---

func TestBedrockAttemptCounterIsolation(t *testing.T) {
	cfg := defaultConfig()
	cfg.Bedrock.Tokens = 2
	srv := testServer(cfg)
	defer srv.Close()

	client := bedrockClient(srv.URL)
	for range 2 {
		if _, err := client.Converse(context.Background(), converseInput(bedrockTestModel)); err != nil {
			t.Fatalf("Converse failed: %v", err)
		}
	}

	var counts struct {
		OpenAI    int64 `json:"openai"`
		Anthropic int64 `json:"anthropic"`
		Bedrock   int64 `json:"bedrock"`
	}
	getJSON(t, srv.URL+"/admin/request-count", &counts)
	if counts.Bedrock != 2 {
		t.Fatalf("bedrock attempts = %d, want 2", counts.Bedrock)
	}
	if counts.Anthropic != 0 || counts.OpenAI != 0 {
		t.Fatalf("bedrock traffic leaked into other counters (R2 aliasing): %+v", counts)
	}
}

func TestBedrockFailFirstN(t *testing.T) {
	cfg := defaultConfig()
	cfg.Bedrock.FailFirstN = 1
	cfg.Bedrock.ErrorStatus = 503
	srv := testServer(cfg)
	defer srv.Close()

	client := bedrockClient(srv.URL)
	_, err := client.Converse(context.Background(), converseInput(bedrockTestModel))
	var unavailable *types.ServiceUnavailableException
	if !errors.As(err, &unavailable) {
		t.Fatalf("first attempt error = %v (%T), want ServiceUnavailableException", err, err)
	}
	if _, err := client.Converse(context.Background(), converseInput(bedrockTestModel)); err != nil {
		t.Fatalf("second attempt should succeed: %v", err)
	}
}

func TestBedrockScenarioProviderBinding(t *testing.T) {
	srv := testServer(defaultConfig())
	defer srv.Close()

	reg, _ := json.Marshal(map[string]any{
		"id": "bd", "provider": "bedrock", "output": map[string]any{"text": "x"},
	})
	mustRegisterScenario(t, srv.URL, string(reg))

	// K2: a bedrock scenario targeted at the Anthropic route is a 409.
	resp := postAnthropicScenario(t, srv.URL, anthropicBody(false), "bd")
	defer resp.Body.Close()
	if resp.StatusCode != 409 {
		t.Fatalf("bedrock scenario on /v1/messages: got %d, want 409", resp.StatusCode)
	}
}

func postAnthropicScenario(t *testing.T, url, body, scenario string) *http.Response {
	t.Helper()
	req, _ := http.NewRequest("POST", url+"/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", "k")
	req.Header.Set("anthropic-version", "2023-06-01")
	req.Header.Set("X-MockLM-Scenario", scenario)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	return resp
}

// The raw HTTP error envelope carries the AWS restjson shape: {message}
// body + X-Amzn-ErrorType header (what a non-SDK client sees).
func TestBedrockErrorEnvelopeShape(t *testing.T) {
	cfg := defaultConfig()
	cfg.Bedrock.Faults = []FaultSpec{{Mode: "error", ErrorStatus: 429, ErrorMessage: "slow down"}}
	srv := testServer(cfg)
	defer srv.Close()

	resp, err := postJSON(srv.URL+"/model/"+bedrockTestModel+"/converse", `{"messages":[]}`, nil)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 429 {
		t.Fatalf("status = %d, want 429", resp.StatusCode)
	}
	if got := resp.Header.Get("X-Amzn-ErrorType"); got != "ThrottlingException" {
		t.Fatalf("X-Amzn-ErrorType = %q, want ThrottlingException", got)
	}
	raw, _ := io.ReadAll(resp.Body)
	var envelope struct {
		Message string `json:"message"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil || envelope.Message != "slow down" {
		t.Fatalf("error envelope = %s, want {\"message\":\"slow down\"}", raw)
	}
}
