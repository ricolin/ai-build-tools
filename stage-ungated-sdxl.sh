#!/usr/bin/env bash
set -euo pipefail

# Downloads only the Diffusers fp16 components needed for SDXL training/inference;
# it never requests or reads an HF token.
readonly REPO=${HF_MODEL_REPO:-stabilityai/stable-diffusion-xl-base-1.0}
readonly REVISION=${HF_MODEL_REVISION:-462165984030d82259a11f4367a4eed129e94a7b}
readonly MODEL_ROOT=${MODEL_ROOT:-/srv/ai-inputs}
readonly MODEL_DIR=$MODEL_ROOT/sdxl-base-1.0
readonly HF_CLI_IMAGE=${HF_CLI_IMAGE:-ai-build-tools:latest}

files=(
  model_index.json
  scheduler/scheduler_config.json
  text_encoder/config.json
  text_encoder/model.fp16.safetensors
  text_encoder_2/config.json
  text_encoder_2/model.fp16.safetensors
  tokenizer/merges.txt
  tokenizer/special_tokens_map.json
  tokenizer/tokenizer_config.json
  tokenizer/vocab.json
  tokenizer_2/merges.txt
  tokenizer_2/special_tokens_map.json
  tokenizer_2/tokenizer_config.json
  tokenizer_2/vocab.json
  unet/config.json
  unet/diffusion_pytorch_model.fp16.safetensors
  vae/config.json
  vae/diffusion_pytorch_model.fp16.safetensors
)

install -d -m 0755 "$MODEL_DIR"
curl -fsSL --max-time 30 \
  "https://huggingface.co/$REPO/resolve/$REVISION/model_index.json" \
  -o /dev/null

docker run --rm --network host \
  -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
  -v "$MODEL_ROOT:/models" \
  --entrypoint hf "$HF_CLI_IMAGE" \
  download "$REPO" "${files[@]}" \
    --revision "$REVISION" \
    --local-dir /models/sdxl-base-1.0 \
    --max-workers 4

printf '%s\n' "$REPO@$REVISION" >"$MODEL_DIR/SOURCE"
find "$MODEL_DIR" -path '*/.cache/*' -prune -o -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  >"$MODEL_ROOT/sdxl-base-1.0.SHA256SUMS"

test -s "$MODEL_DIR/model_index.json"
test -s "$MODEL_DIR/unet/diffusion_pytorch_model.fp16.safetensors"
du -sh "$MODEL_DIR"
printf 'MODEL_READY=%s\n' "$MODEL_DIR"
