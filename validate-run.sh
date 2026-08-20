#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_DIR=
while (($#)); do
  case "$1" in
    --run-dir) RUN_DIR=${2:?}; shift 2 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -d "$RUN_DIR" ]] || { printf 'run directory not found\n' >&2; exit 1; }

if python3 -c 'import PIL' >/dev/null 2>&1; then
  python3 "$ROOT/scripts/validate_release.py" \
    --release-dir "$RUN_DIR/release" \
    --expected-images 3 \
    --summary "$RUN_DIR/summary.json"
else
  image=$(jq -r '.workload_digest // empty' "$RUN_DIR/release/training-config.json")
  [[ "$image" =~ ^[A-Za-z0-9_./:@-]+$ ]] || {
    printf 'release has no safe workload image reference\n' >&2
    exit 1
  }
  docker run --rm --network none \
    -v "$RUN_DIR:/run" \
    "$image" \
    python3 /opt/ai-build-tools/scripts/validate_release.py \
      --release-dir /run/release --expected-images 3 \
      --summary /run/summary.json
fi

for phase in 00-preflight 10-deploy 20-accept-guest 30-pilot 40-train 50-generate 60-package 70-finalize; do
  [[ -e "$RUN_DIR/state/$phase.complete" ]] || {
    printf 'missing phase marker: %s\n' "$phase" >&2
    exit 1
  }
done

printf 'RESULT=PASS RUN_DIR=%s SUMMARY=%s/summary.json\n' "$RUN_DIR" "$RUN_DIR"
