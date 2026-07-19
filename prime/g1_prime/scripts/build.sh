#!/usr/bin/env bash
# Build the vendored PRIME solver library and the g1cal overlay.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cmake -S "$ROOT" -B "$ROOT/build"
cmake --build "$ROOT/build" -j
echo "build complete: $ROOT/build/overlay/g1_motion_fie"
