#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ValidatorTests(unittest.TestCase):
    def make_images(self, directory: Path, count: int = 3) -> list[Path]:
        images = []
        for index in range(count):
            path = directory / f"bear-{index}.png"
            image = Image.new("RGB", (32, 32))
            for x in range(32):
                for y in range(32):
                    image.putpixel((x, y), ((x * 7 + index * 31) % 256, y * 7, 120))
            image.save(path)
            images.append(path)
        return images

    def test_dataset_manifest_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            images = self.make_images(dataset)
            records = [
                {
                    "path": image.name,
                    "caption": f"cute bear training image {index}",
                    "source": "synthetic-validator-fixture",
                    "license": "test-only",
                    "sha256": sha256(image),
                }
                for index, image in enumerate(images)
            ]
            (dataset / "manifest.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )
            summary = dataset / "summary.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/prepare_dataset.py"),
                    "--dataset-dir",
                    str(dataset),
                    "--summary",
                    str(summary),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(json.loads(summary.read_text())["image_count"], 3)
            self.assertIn('"image_count": 3', result.stdout)

    def test_release_passes_and_duplicate_image_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            release = Path(temporary)
            (release / "adapter").mkdir()
            (release / "images").mkdir()
            (release / "adapter/pytorch_lora_weights.safetensors").write_bytes(b"adapter")
            images = self.make_images(release / "images")
            metadata = [{"file": image.name, "sha256": sha256(image)} for image in images]
            (release / "images/metadata.json").write_text(json.dumps(metadata))
            (release / "SHA256SUMS").write_text("fixture\n")
            summary = release / "summary.json"
            command = [
                sys.executable,
                str(ROOT / "scripts/validate_release.py"),
                "--release-dir",
                str(release),
                "--expected-images",
                "3",
                "--summary",
                str(summary),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(summary.read_text())["result"], "PASS")

            metadata[2]["file"] = metadata[1]["file"]
            metadata[2]["sha256"] = metadata[1]["sha256"]
            (release / "images/metadata.json").write_text(json.dumps(metadata))
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("duplicate generated image hash", failed.stderr)


if __name__ == "__main__":
    unittest.main()
