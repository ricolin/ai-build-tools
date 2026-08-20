ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG SOURCE_DATE_EPOCH=0

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    DIFFUSERS_VERBOSITY=error \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git python3 python3-pip python3-venv tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ai-build-tools
COPY requirements.in /opt/ai-build-tools/requirements.in
RUN python3 -m pip install --break-system-packages --no-cache-dir \
      -r /opt/ai-build-tools/requirements.in

ARG DIFFUSERS_COMMIT
RUN test -n "${DIFFUSERS_COMMIT}" \
    && git clone https://github.com/huggingface/diffusers.git /opt/diffusers \
    && git -C /opt/diffusers checkout --detach "${DIFFUSERS_COMMIT}" \
    && test "$(git -C /opt/diffusers rev-parse HEAD)" = "${DIFFUSERS_COMMIT}" \
    && python3 -m pip install --break-system-packages --no-cache-dir --no-deps \
         /opt/diffusers

# The pinned SDXL DreamBooth script performs a final local LoRA reload even
# without a validation prompt. In HF offline mode the loader requires the
# otherwise conventional filename to be explicit. Assert and patch only that
# exact call so training remains network-isolated.
RUN grep -Fxq '        pipeline.load_lora_weights(args.output_dir)' \
      /opt/diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py \
    && sed -i \
      's/pipeline.load_lora_weights(args.output_dir)$/pipeline.load_lora_weights(args.output_dir, weight_name="pytorch_lora_weights.safetensors")/' \
      /opt/diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py \
    && grep -Fxq \
      '        pipeline.load_lora_weights(args.output_dir, weight_name="pytorch_lora_weights.safetensors")' \
      /opt/diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py

COPY scripts /opt/ai-build-tools/scripts
COPY prompts /opt/ai-build-tools/prompts
COPY policy /opt/ai-build-tools/policy

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python3", "/opt/ai-build-tools/scripts/preflight.py"]
