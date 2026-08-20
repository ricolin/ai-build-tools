#!/usr/bin/env python3
import json
import os
import platform
import sys
from pathlib import Path


def main() -> int:
    result = {
        "python": platform.python_version(),
        "hostname": platform.node(),
    }
    try:
        import torch

        result.update(
            torch_version=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            cuda_device_count=torch.cuda.device_count(),
            cuda_devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        )
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("PyTorch did not detect a CUDA GPU")
        torch.manual_seed(26081001)
        x = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
        result["torch_sum"] = float((x @ x.T).sum().cpu())
        if result["torch_sum"] != 3680.0:
            raise RuntimeError("unexpected deterministic PyTorch result")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1

    output = os.environ.get("PREFLIGHT_OUTPUT")
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output:
        Path(output).write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
