#!/usr/bin/env bash
# Contract and real-source regression suite. No generated or mocked source rows.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"

"$PYTHON" -m unittest discover -s "$HERE/tests" -v
