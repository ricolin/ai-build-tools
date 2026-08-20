#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageStat


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", required=True, type=Path)
    parser.add_argument("--expected-images", required=True, type=int)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    adapter_candidates = list(args.release_dir.glob("adapter/*.safetensors"))
    if len(adapter_candidates) != 1:
        raise SystemExit("release must contain exactly one adapter safetensors file")
    metadata_path = args.release_dir / "images" / "metadata.json"
    if not metadata_path.is_file():
        raise SystemExit("missing image metadata")
    metadata = json.loads(metadata_path.read_text())
    if len(metadata) != args.expected_images:
        raise SystemExit(f"expected {args.expected_images} images, found {len(metadata)}")

    image_hashes = set()
    images = []
    for record in metadata:
        path = args.release_dir / "images" / record["file"]
        if not path.is_file() or digest(path) != record["sha256"]:
            raise SystemExit(f"missing or invalid image: {path.name}")
        with Image.open(path) as image:
            image.verify()
        with Image.open(path).convert("RGB") as image:
            extrema = ImageStat.Stat(image).extrema
            if all(low == high for low, high in extrema):
                raise SystemExit(f"blank image: {path.name}")
            images.append({"file": path.name, "size": list(image.size), "sha256": record["sha256"]})
        if record["sha256"] in image_hashes:
            raise SystemExit("duplicate generated image hash")
        image_hashes.add(record["sha256"])

    sums = args.release_dir / "SHA256SUMS"
    if not sums.is_file():
        raise SystemExit("missing SHA256SUMS")
    summary = {
        "result": "PASS",
        "release_dir": str(args.release_dir.resolve()),
        "adapter": {
            "file": adapter_candidates[0].name,
            "sha256": digest(adapter_candidates[0]),
        },
        "images": images,
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"release validation failed: {exc}", file=sys.stderr)
        sys.exit(1)
