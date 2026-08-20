#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/scripts/lib.sh"

CONFIG=
PROMPT="a cute little brown bear, friendly expression, soft natural light, detailed photograph"
SEED=26081021
while (($#)); do
  case "$1" in
    --config) CONFIG=${2:?}; shift 2 ;;
    --prompt) PROMPT=${2:?}; shift 2 ;;
    --seed) SEED=${2:?}; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ -n "$CONFIG" ]] || die "--config is required"
load_config "$CONFIG"
require_vars RUNS_ROOT

RUN_DIR=$(readlink -f "$RUNS_ROOT/latest")
[[ -d "$RUN_DIR/release/adapter" ]] || die "no accepted release is available"
[[ -r "$RUN_DIR/state/guest-ip" ]] || die "the retained inference guest is unavailable"
export RUN_DIR

request_id=$(date -u +%Y%m%dT%H%M%SZ)
remote_output="$REMOTE_INPUT_ROOT/requests/$request_id"
remote_train=$(cat "$RUN_DIR/state/remote-train-dir")
printf -v prompt_q '%q' "$PROMPT"
guest_ssh "sudo install -d -m 0775 -o '$GUEST_USER' -g '$GUEST_USER' '$remote_output' && \
  docker run --rm --gpus all --network none \
    -v '$REMOTE_INPUT_ROOT/base:/models/base:ro' \
    -v '$remote_train:/models/adapter:ro' \
    -v '$remote_output:/outputs' \
    '$WORKLOAD_DIGEST' \
    python3 /opt/ai-build-tools/scripts/generate.py \
      --base /models/base --adapter /models/adapter \
      --prompt $prompt_q --seed '$SEED' --resolution '$RESOLUTION' \
      --output /outputs"
guest_ssh "sudo chown -R '$GUEST_USER:$GUEST_USER' '$remote_output'"

mkdir -p "$RUN_DIR/requests/$request_id"
host=$(guest_host); ssh_args
scp "${SSH_ARGS[@]}" -r "$GUEST_USER@$host:$remote_output/." "$RUN_DIR/requests/$request_id/"
printf 'RESULT=PASS IMAGE=%s/cute-bear-request.png\n' "$RUN_DIR/requests/$request_id"
