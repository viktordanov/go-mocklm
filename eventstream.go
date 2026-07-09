package main

import (
	"encoding/binary"
	"hash/crc32"
	"net/http"
)

// AWS application/vnd.amazon.eventstream framing — the binary format
// Bedrock ConverseStream responses ride (NOT SSE). Encoder only, std-lib
// only; the aws-sdk-go-v2 decoder verifies it in tests. Per message:
//
//	[ total_length  : uint32 BE ]
//	[ headers_length: uint32 BE ]
//	[ prelude_crc   : uint32 BE ]  CRC32-IEEE of the previous 8 bytes
//	[ headers ]                    each: 1-byte name length, name,
//	                               1-byte value type (7 = string),
//	                               2-byte value length BE, value
//	[ payload ]
//	[ message_crc   : uint32 BE ]  CRC32-IEEE of everything above

// eventStreamHeader is one string-valued message header.
type eventStreamHeader struct {
	name  string
	value string
}

// encodeEventStreamMessage frames one message.
func encodeEventStreamMessage(headers []eventStreamHeader, payload []byte) []byte {
	var hbuf []byte
	for _, h := range headers {
		hbuf = append(hbuf, byte(len(h.name)))
		hbuf = append(hbuf, h.name...)
		hbuf = append(hbuf, 7) // header value type 7 = string
		var vlen [2]byte
		binary.BigEndian.PutUint16(vlen[:], uint16(len(h.value)))
		hbuf = append(hbuf, vlen[:]...)
		hbuf = append(hbuf, h.value...)
	}

	total := 12 + len(hbuf) + len(payload) + 4
	buf := make([]byte, 0, total)
	var b4 [4]byte
	binary.BigEndian.PutUint32(b4[:], uint32(total))
	buf = append(buf, b4[:]...)
	binary.BigEndian.PutUint32(b4[:], uint32(len(hbuf)))
	buf = append(buf, b4[:]...)
	binary.BigEndian.PutUint32(b4[:], crc32.ChecksumIEEE(buf))
	buf = append(buf, b4[:]...)
	buf = append(buf, hbuf...)
	buf = append(buf, payload...)
	binary.BigEndian.PutUint32(b4[:], crc32.ChecksumIEEE(buf))
	buf = append(buf, b4[:]...)
	return buf
}

// eventStreamWriter emits eventstream messages over an HTTP response —
// the Bedrock sibling of sseWriter. The SSE transport faults
// (crlf_frames / fragment_* / coalesce_frames) do not apply here:
// eventstream has its own binary framing.
type eventStreamWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
	// bytes counts everything written to the stream body so far; it backs
	// the after_bytes fault-trigger knob (streamSink).
	bytes int
}

func newEventStreamWriter(w http.ResponseWriter) *eventStreamWriter {
	flusher, _ := w.(http.Flusher)
	w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
	return &eventStreamWriter{w: w, flusher: flusher}
}

// writeEvent emits one event message: :message-type=event,
// :event-type=name, :content-type=application/json, JSON payload.
func (e *eventStreamWriter) writeEvent(name string, payload []byte) {
	e.writeRaw(encodeEventStreamMessage([]eventStreamHeader{
		{":event-type", name},
		{":content-type", "application/json"},
		{":message-type", "event"},
	}, payload))
}

// writeException emits one in-band exception message
// (:message-type=exception, :exception-type=name) — the streaming error
// channel, distinct from pre-body HTTP errors. Per AWS semantics the
// stream is over after an exception.
func (e *eventStreamWriter) writeException(name string, payload []byte) {
	e.writeRaw(encodeEventStreamMessage([]eventStreamHeader{
		{":exception-type", name},
		{":content-type", "application/json"},
		{":message-type", "exception"},
	}, payload))
}

// bodyBytes implements streamSink.
func (e *eventStreamWriter) bodyBytes() int {
	return e.bytes
}

// writeCorrupt implements streamSink: a well-formed frame whose message
// CRC is flipped — the eventstream dialect of malformed_chunk (there is no
// "corrupt JSON in a valid frame" here; the envelope itself is the
// integrity layer, so the CRC is what a hostile intermediary would break).
func (e *eventStreamWriter) writeCorrupt() {
	frame := encodeEventStreamMessage([]eventStreamHeader{
		{":event-type", "contentBlockDelta"},
		{":content-type", "application/json"},
		{":message-type", "event"},
	}, []byte(`{"contentBlockIndex":0,"delta":{"text":"corrupt"}}`))
	frame[len(frame)-1] ^= 0xFF // break the message CRC
	e.writeRaw(frame)
}

func (e *eventStreamWriter) writeRaw(frame []byte) {
	e.bytes += len(frame)
	e.w.Write(frame)
	if e.flusher != nil {
		e.flusher.Flush()
	}
}
