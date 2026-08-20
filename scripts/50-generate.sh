#!/usr/bin/env bash
set -euo pipefail
source "$ROOT/scripts/lib.sh"
load_config "$CONFIG"

remote_train=$(cat "$RUN_DIR/state/remote-train-dir")
remote_output="$REMOTE_INPUT_ROOT/runs/${RUN_ID}-images"
[[ "$remote_output" == "$REMOTE_INPUT_ROOT/runs/${RUN_ID}-images" ]] || \
  die "unsafe generation output path"
guest_ssh "sudo rm -rf '$remote_output' && \
  sudo install -d -m 0775 -o '$GUEST_USER' -g '$GUEST_USER' '$remote_output' && \
  docker run --rm --gpus all --network none \
    -v '$REMOTE_INPUT_ROOT/base:/models/base:ro' \
    -v '$remote_train:/models/adapter:ro' \
    -v '$REMOTE_INPUT_ROOT/prompts:/prompts:ro' \
    -v '$remote_output:/outputs' \
    '$WORKLOAD_DIGEST' \
    python3 /opt/ai-build-tools/scripts/generate.py \
      --base /models/base --adapter /models/adapter \
      --prompts /prompts/bear-v1.json --resolution '$RESOLUTION' \
      --output /outputs"
guest_ssh "sudo chown -R '$GUEST_USER:$GUEST_USER' '$remote_output'"
guest_ssh "test \"\$(find '$remote_output' -maxdepth 1 -name '*.png' | wc -l)\" -eq 3 && \
  test -s '$remote_output/metadata.json'"
printf '%s\n' "$remote_output" >"$RUN_DIR/state/remote-images-dir"
