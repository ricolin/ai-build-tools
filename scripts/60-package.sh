#!/usr/bin/env bash
set -euo pipefail
source "$ROOT/scripts/lib.sh"
load_config "$CONFIG"

remote_train=$(cat "$RUN_DIR/state/remote-train-dir")
remote_images=$(cat "$RUN_DIR/state/remote-images-dir")
mkdir -p "$RUN_DIR/release"/{adapter,images,evidence}
host=$(guest_host); ssh_args
scp "${SSH_ARGS[@]}" \
  "$GUEST_USER@$host:$remote_train/pytorch_lora_weights.safetensors" \
  "$RUN_DIR/release/adapter/"
scp "${SSH_ARGS[@]}" \
  "$GUEST_USER@$host:$remote_train/train.log" \
  "$GUEST_USER@$host:$remote_train/gpu-telemetry.csv" \
  "$RUN_DIR/release/evidence/"
scp "${SSH_ARGS[@]}" -r "$GUEST_USER@$host:$remote_images/." \
  "$RUN_DIR/release/images/"
cp "$RUN_DIR/evidence/preflight/dataset.json" "$RUN_DIR/release/dataset-summary.json"
cp "$RUN_DIR/evidence/preflight/base-model-SHA256SUMS" \
  "$RUN_DIR/release/base-model-SHA256SUMS"
cp "$ROOT/$PROMPTS_FILE" "$RUN_DIR/release/prompts.json"

cat >"$RUN_DIR/release/training-config.json" <<JSON
{
  "base_model_dir": "$(basename "$BASE_MODEL_DIR")",
  "resolution": $RESOLUTION,
  "rank": $LORA_RANK,
  "pilot_steps": $PILOT_STEPS,
  "train_steps": $TRAIN_STEPS,
  "seed": $TRAIN_SEED,
  "workload_digest": "$WORKLOAD_DIGEST",
  "workload_archive_sha256": "$(test -r "$RUN_DIR/evidence/preflight/workload-archive.sha256" && awk '{print $1}' "$RUN_DIR/evidence/preflight/workload-archive.sha256" || true)"
}
JSON

(
  cd "$RUN_DIR/release"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS
)
if python3 -c 'import PIL' >/dev/null 2>&1; then
  python3 "$ROOT/scripts/validate_release.py" \
    --release-dir "$RUN_DIR/release" --expected-images 3 \
    --summary "$RUN_DIR/summary.json"
else
  docker run --rm --network none \
    -v "$RUN_DIR:/run" \
    "$WORKLOAD_DIGEST" \
    python3 /opt/ai-build-tools/scripts/validate_release.py \
      --release-dir /run/release --expected-images 3 \
      --summary /run/summary.json
fi
