#!/usr/bin/env bash
set -euo pipefail
source "$ROOT/scripts/lib.sh"
load_config "$CONFIG"

remote_output="$REMOTE_INPUT_ROOT/runs/${RUN_ID}-pilot"
[[ "$remote_output" == "$REMOTE_INPUT_ROOT/runs/${RUN_ID}-pilot" ]] || \
  die "unsafe pilot output path"
guest_ssh "sudo rm -rf '$remote_output' && \
  sudo install -d -m 0775 -o '$GUEST_USER' -g '$GUEST_USER' '$remote_output'"

start=$(date +%s)
guest_ssh "timeout --foreground '$TRAIN_TIMEOUT' docker run --rm --gpus all --network none \
  -v '$REMOTE_INPUT_ROOT/base:/models/base:ro' \
  -v '$REMOTE_INPUT_ROOT/dataset:/data:ro' \
  -v '$remote_output:/outputs' \
  '$WORKLOAD_DIGEST' \
  bash /opt/ai-build-tools/scripts/train_lora.sh \
    --base /models/base --dataset /data --output /outputs \
    --max-train-steps '$PILOT_STEPS' --resolution '$RESOLUTION' \
    --rank '$LORA_RANK' --seed '$TRAIN_SEED'"
elapsed=$(($(date +%s) - start))
printf '%s\n' "$elapsed" >"$RUN_DIR/state/pilot-elapsed-seconds"

guest_ssh "sudo chown -R '$GUEST_USER:$GUEST_USER' '$remote_output'"
guest_ssh "test -s '$remote_output/pytorch_lora_weights.safetensors'"
max_full_seconds=3600
estimated=$((elapsed * TRAIN_STEPS / PILOT_STEPS))
printf '%s\n' "$estimated" >"$RUN_DIR/state/estimated-train-seconds"
((estimated <= max_full_seconds)) || die "pilot predicts training will exceed the one-hour phase budget"
