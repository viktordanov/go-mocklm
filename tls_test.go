package main

import (
	"crypto/ecdsa"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// --- HTTPS + h2/ALPN observation lane ---

func TestResolveListenerSpec(t *testing.T) {
	base := &ServerConfig{Host: "127.0.0.1", Port: 9999}
	clear := func() {
		for _, k := range []string{"MOCKLM_HTTP_ENABLED", "MOCKLM_HTTP_PORT", "MOCKLM_TLS_ENABLED", "MOCKLM_TLS_PORT", "MOCKLM_TLS_CERT", "MOCKLM_TLS_KEY"} {
			os.Unsetenv(k)
		}
	}
	clear()
	t.Cleanup(clear)

	// Defaults: historical single clear-text listener on the config port.
	spec, err := resolveListenerSpec(base)
	if err != nil {
		t.Fatalf("default spec errored: %v", err)
	}
	if !spec.httpEnabled || spec.tlsEnabled || spec.httpAddr != "127.0.0.1:9999" {
		t.Fatalf("default spec wrong: %+v", spec)
	}

	// Full env override.
	t.Setenv("MOCKLM_HTTP_ENABLED", "false")
	t.Setenv("MOCKLM_TLS_ENABLED", "1")
	t.Setenv("MOCKLM_TLS_PORT", "9777")
	spec, err = resolveListenerSpec(base)
	if err != nil {
		t.Fatalf("tls-only spec errored: %v", err)
	}
	if spec.httpEnabled || !spec.tlsEnabled || spec.tlsAddr != "127.0.0.1:9777" {
		t.Fatalf("tls-only spec wrong: %+v", spec)
	}

	// Both lanes disabled is a loud startup error.
	t.Setenv("MOCKLM_TLS_ENABLED", "0")
	if _, err := resolveListenerSpec(base); err == nil {
		t.Fatalf("both-disabled must error")
	}

	// Cert and key are both-or-neither.
	t.Setenv("MOCKLM_TLS_ENABLED", "1")
	t.Setenv("MOCKLM_TLS_CERT", "/tmp/cert.pem")
	if _, err := resolveListenerSpec(base); err == nil {
		t.Fatalf("cert-without-key must error")
	}
	os.Unsetenv("MOCKLM_TLS_CERT")

	// Port collision between the two lanes is rejected.
	t.Setenv("MOCKLM_HTTP_ENABLED", "1")
	t.Setenv("MOCKLM_HTTP_PORT", "9777")
	if _, err := resolveListenerSpec(base); err == nil {
		t.Fatalf("shared port must error")
	}

	// Garbage ports are rejected.
	t.Setenv("MOCKLM_HTTP_PORT", "not-a-port")
	if _, err := resolveListenerSpec(base); err == nil {
		t.Fatalf("invalid port must error")
	}
}

func TestLoadOrGenerateCert(t *testing.T) {
	// No overrides: a usable in-memory self-signed cert for localhost.
	cert, err := loadOrGenerateCert("", "")
	if err != nil {
		t.Fatalf("self-signed generation failed: %v", err)
	}
	if cert.Leaf == nil || cert.Leaf.DNSNames[0] != "localhost" {
		t.Fatalf("self-signed cert should cover localhost, got %+v", cert.Leaf)
	}

	// Override paths: a PEM pair round-trips through LoadX509KeyPair.
	dir := t.TempDir()
	certPath, keyPath := filepath.Join(dir, "cert.pem"), filepath.Join(dir, "key.pem")
	keyDER, err := x509.MarshalECPrivateKey(cert.PrivateKey.(*ecdsa.PrivateKey))
	if err != nil {
		t.Fatalf("marshaling key: %v", err)
	}
	if err := os.WriteFile(certPath, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: cert.Certificate[0]}), 0o600); err != nil {
		t.Fatalf("writing cert: %v", err)
	}
	if err := os.WriteFile(keyPath, pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER}), 0o600); err != nil {
		t.Fatalf("writing key: %v", err)
	}
	if _, err := loadOrGenerateCert(certPath, keyPath); err != nil {
		t.Fatalf("loading PEM pair failed: %v", err)
	}
	if _, err := loadOrGenerateCert(filepath.Join(dir, "missing.pem"), keyPath); err == nil {
		t.Fatalf("missing cert file must error")
	}
}

// startTLSLane runs the real mux behind the real TLS server helper on an
// ephemeral port and returns its base URL, the shared state, and a cert
// pool trusting the generated self-signed certificate — so clients verify
// through the REAL trust path (the cert's 127.0.0.1 IP SAN matches the
// dialed host) instead of disabling verification.
func startTLSLane(t *testing.T) (string, *ServerState, *x509.CertPool) {
	t.Helper()
	state := NewServerState(defaultConfig())
	mux := buildMux(state)
	cert, err := generateSelfSignedCert()
	if err != nil {
		t.Fatalf("cert generation failed: %v", err)
	}
	pool := x509.NewCertPool()
	pool.AddCert(cert.Leaf)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen failed: %v", err)
	}
	srv := newTLSServer("", mux, cert)
	go srv.ServeTLS(ln, "", "")
	t.Cleanup(func() { srv.Close() })
	return "https://" + ln.Addr().String(), state, pool
}

func recordedProtos(state *ServerState) []string {
	var protos []string
	for _, rec := range state.Requests() {
		protos = append(protos, rec.Proto)
	}
	return protos
}

func TestTLSLaneRecordsNegotiatedProtocol(t *testing.T) {
	url, state, pool := startTLSLane(t)

	// Forced-h2 client, verifying against the generated self-signed cert
	// via a trusted pool (no InsecureSkipVerify — the documented pin/trust
	// path must actually work): ALPN negotiates h2, the recorded proto
	// proves it.
	h2Client := &http.Client{Transport: &http.Transport{
		TLSClientConfig:   &tls.Config{RootCAs: pool},
		ForceAttemptHTTP2: true,
	}}
	resp, err := h2Client.Post(url+"/v1/chat/completions", "application/json", strings.NewReader(openaiChatBody(false)))
	if err != nil {
		t.Fatalf("h2 request failed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 || resp.Proto != "HTTP/2.0" {
		t.Fatalf("h2 lane: status %d proto %s, want 200 HTTP/2.0", resp.StatusCode, resp.Proto)
	}

	// h1-pinned client, same real verification: a non-nil empty
	// TLSNextProto disables the client's h2 upgrade, so ALPN settles on
	// http/1.1.
	h1Client := &http.Client{Transport: &http.Transport{
		TLSClientConfig: &tls.Config{RootCAs: pool},
		TLSNextProto:    map[string]func(string, *tls.Conn) http.RoundTripper{},
	}}
	resp, err = h1Client.Post(url+"/v1/chat/completions", "application/json", strings.NewReader(openaiChatBody(false)))
	if err != nil {
		t.Fatalf("h1 request failed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 || resp.Proto != "HTTP/1.1" {
		t.Fatalf("h1-pinned lane: status %d proto %s, want 200 HTTP/1.1", resp.StatusCode, resp.Proto)
	}

	protos := recordedProtos(state)
	if len(protos) != 2 || protos[0] != "HTTP/2.0" || protos[1] != "HTTP/1.1" {
		t.Fatalf("recorded protos = %v, want [HTTP/2.0 HTTP/1.1]", protos)
	}
}

func TestScenarioCaptureProtocolOverTLS(t *testing.T) {
	// CapturedRequest.Protocol stops being inert on the TLS lane: the
	// last-request header reports the negotiated protocol.
	url, _, pool := startTLSLane(t)
	client := &http.Client{Transport: &http.Transport{
		TLSClientConfig:   &tls.Config{RootCAs: pool},
		ForceAttemptHTTP2: true,
	}}

	reg, _ := json.Marshal(map[string]any{
		"id": "proto-cap", "provider": "openai", "model": "gpt-proto",
		"output": map[string]any{"text": "ok"},
	})
	resp, err := client.Post(url+"/admin/scenarios", "application/json", strings.NewReader(string(reg)))
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("register status %d", resp.StatusCode)
	}

	resp, err = client.Post(url+"/v1/chat/completions", "application/json", strings.NewReader(openaiChatBodyModel("gpt-proto", false)))
	if err != nil {
		t.Fatalf("scenario request failed: %v", err)
	}
	resp.Body.Close()

	resp, err = client.Get(url + "/admin/scenarios/proto-cap/last-request")
	if err != nil {
		t.Fatalf("last-request failed: %v", err)
	}
	resp.Body.Close()
	if proto := resp.Header.Get("X-MockLM-Captured-Protocol"); proto != "HTTP/2.0" {
		t.Fatalf("captured protocol = %q, want HTTP/2.0", proto)
	}
}

func TestCleartextLaneRecordsHTTP11(t *testing.T) {
	// The historical clear-text lane records HTTP/1.1 — the negative arm
	// of the ALPN oracle.
	state := NewServerState(defaultConfig())
	srv := httptest.NewServer(buildMux(state))
	defer srv.Close()

	resp, err := postJSON(srv.URL+"/v1/chat/completions", openaiChatBody(false), nil)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	resp.Body.Close()
	if protos := recordedProtos(state); len(protos) != 1 || protos[0] != "HTTP/1.1" {
		t.Fatalf("cleartext recorded protos = %v, want [HTTP/1.1]", protos)
	}
}
