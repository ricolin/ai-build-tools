#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--max-images", type=int, default=48)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    manifest = args.dataset_dir / "manifest.jsonl"
    if not manifest.is_file():
        raise SystemExit(f"missing dataset manifest: {manifest}")

    records = []
    seen_paths = set()
    seen_hashes = set()
    for number, line in enumerate(manifest.read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        required = {"path", "caption", "source", "license", "sha256"}
        missing = required - record.keys()
        if missing:
            raise SystemExit(f"manifest line {number} missing: {sorted(missing)}")
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit(f"unsafe dataset path on line {number}")
        image_path = args.dataset_dir / relative
        if not image_path.is_file():
            raise SystemExit(f"missing image: {relative}")
        actual = digest(image_path)
        if actual != record["sha256"]:
            raise SystemExit(f"hash mismatch: {relative}")
        if str(relative) in seen_paths or actual in seen_hashes:
            raise SystemExit(f"duplicate dataset entry: {relative}")
        if not str(record["caption"]).strip() or not str(record["license"]).strip():
            raise SystemExit(f"empty caption or license: {relative}")
        with Image.open(image_path) as image:
            image.verify()
        seen_paths.add(str(relative))
        seen_hashes.add(actual)
        records.append(record)

    if not 3 <= len(records) <= args.max_images:
        raise SystemExit(f"dataset contains {len(records)} images; expected 3..{args.max_images}")

    summary = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "image_count": len(records),
        "manifest_sha256": digest(manifest),
        "licenses": sorted({str(item["license"]) for item in records}),
    }
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.summary:
        args.summary.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"dataset validation failed: {exc}", file=sys.stderr)
        sys.exit(1)
