# AI Build Tools: H200 Workload Training and Validation Suite

This repository provides reproducible automation for deploying, validating,
and fine-tuning AI models on bare-metal GPU infrastructure such as NVIDIA H200
accelerators managed through OpenStack Ironic.

## Overview

The suite executes a guarded end-to-end workflow:
1. **Preflight Verification**: Asserts that all inputs, model artifacts, dataset manifests, container images, OpenStack quotas, and bare-metal nodes match immutable contract constraints.
2. **Infrastructure Allocation**: Deploys a bare-metal server using OpenStack Nova/Ironic with ConfigDrive and cloud-init first-boot bootstrap.
3. **Guest & GPU Runtime Acceptance**: Validates NVIDIA drivers, Fabric Manager, PyTorch CUDA tensor operations, and JAX GPU backend capabilities.
4. **Pilot Run**: Performs a low-step training run to verify dataset loading, gradient computation, and disk write permissions within strict time bounds.
5. **Distributed Fine-Tuning**: Trains a low-rank adapter (LoRA) for Stable Diffusion XL Base 1.0 using all available GPUs in a network-isolated environment.
6. **Inference & Release Validation**: Generates acceptance images, validates checksums, dimensions, and image non-blankness, and packages immutable release artifacts.
7. **Resource Lifecycle**: Retains or cleanly deletes bare-metal server allocations.

## Documentation

- **[AI Architecture & Distributed Training Guide](docs/architecture.md)**: Deep dive into the neural network architecture, SDXL pipeline components, High-Level Architecture Diagram, LoRA PEFT mechanics, distributed training parameters, and inference.

## Repository Structure

```text
├── Dockerfile                   # Pinned CUDA & Diffusers environment
├── requirements.in              # Strict Python package dependencies
├── config.env.example           # Configuration template
├── run-all.sh                   # Main workflow orchestrator
├── validate-run.sh              # Post-run release verification script
├── create-cute-bear.sh          # Inference request runner for retained models
├── stage-ungated-sdxl.sh        # Model staging utility for SDXL fp16
├── cloud-init/
│   ├── gpu-runtime.yaml         # Cloud-init definition for GPU nodes
│   └── gpu-runtime-multipart.txt # Multipart boothook with machine-id setup
├── policy/
│   └── model-card-template.md   # Model governance and metadata card template
├── prompts/
│   └── bear-v1.json             # Pinned prompts for release image generation
├── scripts/
│   ├── lib.sh                   # Common bash functions & validation helpers
│   ├── 00-preflight.sh          # Phase 0: Preflight checks
│   ├── 10-deploy.sh             # Phase 1: Bare-metal deployment
│   ├── 20-accept-guest.sh       # Phase 2: Guest OS & GPU runtime acceptance
│   ├── 30-pilot.sh              # Phase 3: Short pilot run
│   ├── 40-train.sh              # Phase 4: Distributed LoRA training
│   ├── 50-generate.sh           # Phase 5: Image generation
│   ├── 60-package.sh            # Phase 6: Release packaging
│   ├── 70-finalize.sh           # Phase 7: Lifecycle finalization
│   ├── bootstrap_gpu_runtime.sh # Guest runtime & driver installer
│   ├── generate.py              # Inference script
│   ├── jax_gpu_smoke.py         # JAX GPU verification test
│   ├── preflight.py             # PyTorch CUDA verification test
│   ├── prepare_dataset.py       # Dataset manifest validator
│   ├── train_lora.sh            # DreamBooth LoRA training launcher
│   └── validate_release.py      # Image & adapter checksum validator
└── tests/
    └── test_validators.py       # Unit tests for validators
```

## Quick Start

### 1. Build Container Image

```bash
docker build -t ai-build-tools:latest \
  --build-arg DIFFUSERS_COMMIT=462165984030d82259a11f4367a4eed129e94a7b \
  .
```

### 2. Configure Environment

Copy the example configuration and fill in your OpenStack parameters:

```bash
cp config.env.example config.env
# Edit config.env with your OpenStack node and network parameters
```

### 3. Run Preflight Only

```bash
./run-all.sh --config ./config.env --preflight-only
```

### 4. Execute Full Training Workflow

```bash
export CONFIRM_IMAGE_RUN=approved-h200-run
./run-all.sh --config ./config.env --final-action retain
```

### 5. Validate Outputs

```bash
./validate-run.sh --run-dir runs/latest
```

## License

Apache License 2.0
