#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline


DEFAULT_PROMPT = "a cute little brown bear, friendly expression, soft natural light, detailed photograph"
DEFAULT_NEGATIVE = "text, watermark, logo, duplicate animal, malformed paws, extra limbs, blurry"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompts", type=Path)
    group.add_argument("--prompt")
    parser.add_argument("--seed", type=int, default=26081021)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=4.5)
    args = parser.parse_args()

    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prompts:
        requests = json.loads(args.prompts.read_text())
        if not isinstance(requests, list) or not 1 <= len(requests) <= 3:
            raise SystemExit("prompt file must contain 1..3 requests")
    else:
        requests = [{
            "id": "cute-bear-request",
            "prompt": args.prompt or DEFAULT_PROMPT,
            "negative_prompt": DEFAULT_NEGATIVE,
            "seed": args.seed,
        }]

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        str(args.base), torch_dtype=torch.bfloat16, local_files_only=True,
        variant="fp16",
    ).to("cuda")
    pipeline.load_lora_weights(
        str(args.adapter),
        weight_name="pytorch_lora_weights.safetensors",
        local_files_only=True,
    )
    pipeline.set_progress_bar_config(disable=True)

    metadata = []
    for request in requests:
        identifier = str(request["id"])
        if not identifier.replace("-", "").isalnum():
            raise SystemExit(f"unsafe prompt id: {identifier}")
        seed = int(request["seed"])
        generator = torch.Generator(device="cuda").manual_seed(seed)
        image = pipeline(
            prompt=str(request["prompt"]),
            negative_prompt=str(request.get("negative_prompt", DEFAULT_NEGATIVE)),
            width=args.resolution,
            height=args.resolution,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        ).images[0]
        image_path = args.output / f"{identifier}.png"
        image.save(image_path, format="PNG")
        metadata.append({
            "id": identifier,
            "prompt": request["prompt"],
            "negative_prompt": request.get("negative_prompt", DEFAULT_NEGATIVE),
            "seed": seed,
            "width": args.resolution,
            "height": args.resolution,
            "steps": args.steps,
            "guidance": args.guidance,
            "sha256": sha256(image_path),
            "file": image_path.name,
        })

    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"generated": len(metadata), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
