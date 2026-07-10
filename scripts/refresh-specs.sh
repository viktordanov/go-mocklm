#!/usr/bin/env bash
#
# refresh-specs.sh — deliberately re-vendor nanollm's pinned OpenAPI specs
# and regenerate the validator closures + pins from them.
#
# go-mocklm vendors nanollm's sha256-pinned specs under testdata/nanollm-spec
# so CI is self-contained (no cross-repo checkout of the private nanollm
# repo). When nanollm's specs move, run this to bump the vendored copy on
# purpose, then review + commit the diff.
#
# Usage (from the repo root):
#   scripts/refresh-specs.sh [PATH_TO_NANOLLM_SPEC_DIR]
#
# PATH_TO_NANOLLM_SPEC_DIR defaults to ../nanollm/spec.
#
# What it does:
#   1. copies openai-openapi.json, anthropic-openapi.json and their .sha256
#      pin files (exact bytes) into testdata/nanollm-spec/
#   2. re-extracts the vendored schema closures + spec/pins.json from the
#      freshly vendored specs via `go run ./cmd/specsync`
#
# After running, `go test ./...` re-verifies the pins and closures match.
set -euo pipefail

SRC="${1:-../nanollm/spec}"
DEST="testdata/nanollm-spec"

if [[ ! -f go.mod ]] || ! grep -q 'go-mocklm' go.mod; then
  echo "error: run this from the go-mocklm repo root" >&2
  exit 1
fi

if [[ ! -d "$SRC" ]]; then
  echo "error: nanollm spec dir not found at '$SRC'" >&2
  echo "       pass the path explicitly: scripts/refresh-specs.sh /path/to/nanollm/spec" >&2
  exit 1
fi

mkdir -p "$DEST"
for f in openai-openapi.json anthropic-openapi.json \
         openai-openapi.json.sha256 anthropic-openapi.json.sha256; do
  if [[ ! -f "$SRC/$f" ]]; then
    echo "error: missing '$SRC/$f'" >&2
    exit 1
  fi
  cp -p "$SRC/$f" "$DEST/$f"
  echo "vendored $f"
done

# Regenerate the vendored closures + spec/pins.json from the vendored specs.
go run ./cmd/specsync -spec-dir "$DEST"

echo
echo "done. Review the diff under $DEST/ and spec/, then commit."
echo "Verify: env -u NANOLLM_SPEC_DIR MOCKLM_REQUIRE_SPEC_SYNC=1 MOCKLM_VALIDATE_RESPONSES=1 go test ./..."
