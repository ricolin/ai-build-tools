#!/usr/bin/env bash
set -euo pipefail
source "$ROOT/scripts/lib.sh"
load_config "$CONFIG"

remote_output="$REMOTE_INPUT_ROOT/runs/${RUN_ID}-train"
[[ "$remote_output" == "$REMOTE_INPUT_ROOT/runs/${RUN_ID}-train" ]] || \
  die "unsafe training output path"
guest_ssh "sudo rm -rf '$remote_output' && \
  sudo install -d -m 0775 -o '$GUEST_USER' -g '$GUEST_USER' '$remote_output'"

guest_ssh "(
  while true; do
    date -u +%FT%TZ
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu,power.draw \
      --format=csv,noheader,nounits
    sleep 30
  done
) >'$remote_output/gpu-telemetry.csv' 2>&1 & echo \$! >'$remote_output/telemetry.pid'"

cleanup_telemetry() {
  guest_ssh "test -r '$remote_output/telemetry.pid' && \
    kill \$(cat '$remote_output/telemetry.pid') 2>/dev/null || true" || true
}
trap cleanup_telemetry EXIT

guest_ssh "timeout --foreground '$TRAIN_TIMEOUT' docker run --rm --gpus all --network none \
  -v '$REMOTE_INPUT_ROOT/base:/models/base:ro' \
  -v '$REMOTE_INPUT_ROOT/dataset:/data:ro' \
  -v '$remote_output:/outputs' \
  '$WORKLOAD_DIGEST' \
  bash /opt/ai-build-tools/scripts/train_lora.sh \
    --base /models/base --dataset /data --output /outputs \
    --max-train-steps '$TRAIN_STEPS' --resolution '$RESOLUTION' \
    --rank '$LORA_RANK' --seed '$TRAIN_SEED' \
  >'$remote_output/train.log' 2>&1"

cleanup_telemetry
trap - EXIT
guest_ssh "sudo chown -R '$GUEST_USER:$GUEST_USER' '$remote_output'"
guest_ssh "test -s '$remote_output/pytorch_lora_weights.safetensors' && \
  ! grep -Eiq '(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' '$remote_output/train.log'"
printf '%s\n' "$remote_output" >"$RUN_DIR/state/remote-train-dir"
