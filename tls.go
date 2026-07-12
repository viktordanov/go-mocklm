package main

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"fmt"
	"math/big"
	"net"
	"net/http"
	"os"
	"strconv"
	"time"
)

// HTTPS + h2/ALPN observation lane.
//
// The mock can run a TLS listener ALONGSIDE the clear-text one so a gateway
// harness can assert transport behavior a plain-HTTP mock cannot observe:
// Go's net/http negotiates HTTP/2 over TLS via ALPN automatically, and
// every recorded request carries r.Proto ("HTTP/2.0" vs "HTTP/1.1") — see
// RecordedRequest.Proto and the scenario capture's Protocol field. Whether
// the client actually spoke h2 is then an /admin/requests (or
// last-request) assertion.
//
// Everything is env-var configurable; defaults preserve the historical
// single clear-text listener:
//
//	MOCKLM_HTTP_ENABLED  clear-text lane on/off (default on; "0"/"false" disables)
//	MOCKLM_HTTP_PORT     clear-text port (default: [server].port from config.toml)
//	MOCKLM_TLS_ENABLED   TLS/h2 lane on/off (default off)
//	MOCKLM_TLS_PORT      TLS port (default 9443)
//	MOCKLM_TLS_CERT      PEM certificate path — both-or-neither with MOCKLM_TLS_KEY;
//	MOCKLM_TLS_KEY       when absent, an in-memory self-signed ECDSA cert is
//	                     generated at startup (localhost / 127.0.0.1 / ::1),
//	                     so clients must skip verification or trust it ad hoc
//
// At least one lane must be enabled.

// defaultTLSPort is the TLS lane's port when MOCKLM_TLS_PORT is unset.
const defaultTLSPort = 9443

// listenerSpec is the resolved dual-listener configuration.
type listenerSpec struct {
	httpEnabled bool
	httpAddr    string
	tlsEnabled  bool
	tlsAddr     string
	certFile    string
	keyFile     string
}

// envEnabled reads an on/off env var with an explicit default for the
// unset case ("" = keep the default; anything else is parsed as truthy).
func envEnabled(name string, unset bool) bool {
	v := os.Getenv(name)
	if v == "" {
		return unset
	}
	return envTruthy(v)
}

// envPort reads a port env var, falling back to def when unset.
func envPort(name string, def int) (int, error) {
	v := os.Getenv(name)
	if v == "" {
		return def, nil
	}
	port, err := strconv.Atoi(v)
	if err != nil || port < 1 || port > 65535 {
		return 0, fmt.Errorf("%s: invalid port %q", name, v)
	}
	return port, nil
}

// resolveListenerSpec builds the dual-listener config from the server
// config and the MOCKLM_HTTP_*/MOCKLM_TLS_* env vars.
func resolveListenerSpec(cfg *ServerConfig) (listenerSpec, error) {
	spec := listenerSpec{
		httpEnabled: envEnabled("MOCKLM_HTTP_ENABLED", true),
		tlsEnabled:  envEnabled("MOCKLM_TLS_ENABLED", false),
		certFile:    os.Getenv("MOCKLM_TLS_CERT"),
		keyFile:     os.Getenv("MOCKLM_TLS_KEY"),
	}
	if !spec.httpEnabled && !spec.tlsEnabled {
		return spec, fmt.Errorf("both listeners disabled (MOCKLM_HTTP_ENABLED and MOCKLM_TLS_ENABLED): nothing to serve")
	}
	if (spec.certFile == "") != (spec.keyFile == "") {
		return spec, fmt.Errorf("MOCKLM_TLS_CERT and MOCKLM_TLS_KEY must be set together (or neither, for an in-memory self-signed cert)")
	}
	httpPort, err := envPort("MOCKLM_HTTP_PORT", cfg.Port)
	if err != nil {
		return spec, err
	}
	tlsPort, err := envPort("MOCKLM_TLS_PORT", defaultTLSPort)
	if err != nil {
		return spec, err
	}
	if spec.httpEnabled && spec.tlsEnabled && httpPort == tlsPort {
		return spec, fmt.Errorf("the clear-text and TLS listeners cannot share port %d", httpPort)
	}
	spec.httpAddr = fmt.Sprintf("%s:%d", cfg.Host, httpPort)
	spec.tlsAddr = fmt.Sprintf("%s:%d", cfg.Host, tlsPort)
	return spec, nil
}

// loadOrGenerateCert loads the PEM pair when override paths are set,
// otherwise generates the in-memory self-signed certificate.
func loadOrGenerateCert(certFile, keyFile string) (tls.Certificate, error) {
	if certFile != "" {
		cert, err := tls.LoadX509KeyPair(certFile, keyFile)
		if err != nil {
			return tls.Certificate{}, fmt.Errorf("loading MOCKLM_TLS_CERT/MOCKLM_TLS_KEY: %w", err)
		}
		return cert, nil
	}
	return generateSelfSignedCert()
}

// generateSelfSignedCert builds a fresh ECDSA P-256 certificate for
// localhost/127.0.0.1/::1, valid for a year — an ephemeral identity for a
// test mock, not a trust decision: clients either skip verification or
// pin the presented certificate.
func generateSelfSignedCert() (tls.Certificate, error) {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return tls.Certificate{}, fmt.Errorf("generating TLS key: %w", err)
	}
	serial, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		return tls.Certificate{}, fmt.Errorf("generating serial: %w", err)
	}
	tmpl := x509.Certificate{
		SerialNumber:          serial,
		Subject:               pkix.Name{CommonName: "mocklm self-signed"},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(365 * 24 * time.Hour),
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IsCA:                  true,
		DNSNames:              []string{"localhost"},
		IPAddresses:           []net.IP{net.IPv4(127, 0, 0, 1), net.IPv6loopback},
	}
	der, err := x509.CreateCertificate(rand.Reader, &tmpl, &tmpl, &key.PublicKey, key)
	if err != nil {
		return tls.Certificate{}, fmt.Errorf("creating self-signed cert: %w", err)
	}
	return tls.Certificate{
		Certificate: [][]byte{der},
		PrivateKey:  key,
		Leaf:        mustParseCert(der),
	}, nil
}

func mustParseCert(der []byte) *x509.Certificate {
	c, err := x509.ParseCertificate(der)
	if err != nil {
		panic("mocklm: self-signed cert failed to re-parse: " + err.Error())
	}
	return c
}

// newTLSServer wraps the handler in an http.Server whose ServeTLS /
// ListenAndServeTLS negotiates h2 via ALPN (net/http does this
// automatically when TLSNextProto is untouched).
func newTLSServer(addr string, handler http.Handler, cert tls.Certificate) *http.Server {
	return &http.Server{
		Addr:      addr,
		Handler:   handler,
		TLSConfig: &tls.Config{Certificates: []tls.Certificate{cert}},
	}
}
