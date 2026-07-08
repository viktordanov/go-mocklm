package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/viktordanov/go-mocklm/internal/specsync"
)

// --- Phase 1: spec-sync closure drift tripwire ---
//
// go-mocklm's validator runs against schemas extracted from nanollm's
// sha256-pinned specs (spec/pins.json records which). These tests fail
// when either side moves without the other: nanollm bumps a spec (pins
// diverge) or someone edits the vendored files by hand (regen drift).

// nanollmSpecDir resolves the sibling nanollm spec directory, honouring
// NANOLLM_SPEC_DIR for non-sibling checkouts.
func nanollmSpecDir(t *testing.T) string {
	t.Helper()
	dir := os.Getenv("NANOLLM_SPEC_DIR")
	if dir == "" {
		dir = "../nanollm/spec"
	}
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("nanollm spec dir not found at %s (set NANOLLM_SPEC_DIR): %v — "+
			"the spec-sync drift tripwire DID NOT RUN", dir, err)
	}
	return dir
}

// recordedPins reads spec/pins.json (the sha256 hashes recorded at
// extraction time).
func recordedPins(t *testing.T) map[string]string {
	t.Helper()
	raw, err := os.ReadFile("spec/pins.json")
	if err != nil {
		t.Fatalf("reading spec/pins.json (run `go run ./cmd/specsync`): %v", err)
	}
	var pins map[string]string
	if err := json.Unmarshal(raw, &pins); err != nil {
		t.Fatalf("parsing spec/pins.json: %v", err)
	}
	return pins
}

// TestSpecPinsMatchNanollm fails when the sha256 hashes recorded in
// spec/pins.json diverge from nanollm's spec/ pins — the same drift
// tripwire nanollm's fidelity matrix uses on the Rust side. A failure
// means nanollm's specs moved: rerun `go run ./cmd/specsync` and re-verify
// the mock's default shapes against the new closure.
func TestSpecPinsMatchNanollm(t *testing.T) {
	specDir := nanollmSpecDir(t)
	pins := recordedPins(t)

	for _, name := range []string{"anthropic-openapi.json", "openai-openapi.json"} {
		recorded, ok := pins[name]
		if !ok {
			t.Fatalf("spec/pins.json has no pin for %s", name)
		}

		// nanollm's own pin file (first whitespace-separated token).
		pinRaw, err := os.ReadFile(filepath.Join(specDir, name+".sha256"))
		if err != nil {
			t.Fatalf("reading nanollm pin for %s: %v", name, err)
		}
		nanollmPin := strings.Fields(string(pinRaw))[0]
		if recorded != nanollmPin {
			t.Errorf("%s: recorded pin %s diverged from nanollm's pin %s — "+
				"rerun `go run ./cmd/specsync`", name, recorded, nanollmPin)
		}

		// The actual file must match too (catches a stale pin file).
		src, err := os.ReadFile(filepath.Join(specDir, name))
		if err != nil {
			t.Fatalf("reading %s: %v", name, err)
		}
		sum := sha256.Sum256(src)
		if actual := hex.EncodeToString(sum[:]); recorded != actual {
			t.Errorf("%s: recorded pin %s does not match the file's sha256 %s — "+
				"rerun `go run ./cmd/specsync`", name, recorded, actual)
		}
	}
}

// TestVendoredSchemasMatchFreshExtraction re-runs the closure extraction
// against nanollm's specs and fails when the checked-in vendored schemas
// differ — the same regen check as xtask's --check mode (B-ANT-9).
func TestVendoredSchemasMatchFreshExtraction(t *testing.T) {
	specDir := nanollmSpecDir(t)

	cases := []struct {
		source   string
		vendored []byte
		extract  func([]byte) (*specsync.Extraction, error)
	}{
		{"openai-openapi.json", openaiResponsesSchemaJSON, specsync.ExtractOpenAI},
		{"anthropic-openapi.json", anthropicResponsesSchemaJSON, specsync.ExtractAnthropic},
	}
	for _, tc := range cases {
		src, err := os.ReadFile(filepath.Join(specDir, tc.source))
		if err != nil {
			t.Fatalf("reading %s: %v", tc.source, err)
		}
		fresh, err := tc.extract(src)
		if err != nil {
			t.Fatalf("extracting from %s: %v", tc.source, err)
		}
		if !bytes.Equal(fresh.Schema, tc.vendored) {
			t.Errorf("vendored schema for %s differs from a fresh extraction — "+
				"run `go run ./cmd/specsync` (or revert hand edits)", tc.source)
		}
	}
}

// TestVendoredSchemasCompile guards the embedded schema files: all six
// roots (plus the local ping arm) must compile under full 2020-12.
func TestVendoredSchemasCompile(t *testing.T) {
	if err := compileSchemas(); err != nil {
		t.Fatalf("vendored schemas failed to compile: %v", err)
	}
}
