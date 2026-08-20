#!/usr/bin/env bash
set -euo pipefail

BASE=
DATASET=
OUTPUT=
STEPS=
RESOLUTION=1024
RANK=32
SEED=26081001

while (($#)); do
  case "$1" in
    --base) BASE=${2:?}; shift 2 ;;
    --dataset) DATASET=${2:?}; shift 2 ;;
    --output) OUTPUT=${2:?}; shift 2 ;;
    --max-train-steps) STEPS=${2:?}; shift 2 ;;
    --resolution) RESOLUTION=${2:?}; shift 2 ;;
    --rank) RANK=${2:?}; shift 2 ;;
    --seed) SEED=${2:?}; shift 2 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

for value in BASE DATASET OUTPUT STEPS; do
  [[ -n "${!value}" ]] || { printf '%s is required\n' "$value" >&2; exit 2; }
done
[[ -d "$BASE" && -d "$DATASET/images" ]] || {
  printf 'base model or dataset images directory is missing\n' >&2
  exit 1
}
mkdir -p "$OUTPUT"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

python3 - <<'PY'
import json, torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print(json.dumps({
    "cuda_devices": torch.cuda.device_count(),
    "device_0": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
}, sort_keys=True))
PY

exec accelerate launch --mixed_precision bf16 \
  /opt/diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path "$BASE" \
  --variant fp16 \
  --instance_data_dir "$DATASET/images" \
  --output_dir "$OUTPUT" \
  --instance_prompt "a detailed photograph of a cbear cute little brown bear" \
  --resolution "$RESOLUTION" \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --max-train-steps "$STEPS" \
  --checkpointing_steps "$STEPS" \
  --checkpoints_total_limit 1 \
  --rank "$RANK" \
  --mixed_precision bf16 \
  --seed "$SEED" \
  --report_to tensorboard
