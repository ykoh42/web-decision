"""Golden digests from pinned official Strands decoder and Gemma processor."""

import base64
import hashlib
import importlib
import json
import sys
from pathlib import Path

import torch
from transformers import Gemma4ImageProcessorPil

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root.parent / "strands-upstream-v2" / "src"))
image_module = importlib.import_module("strands_decider.images")
image_core = importlib.import_module("strands_decider.core.images")
decode_image = image_module.decode_image
header_image = image_core.decode_image
resize_image = image_core.resize_image
check_decoded_size = image_core.check_decoded_size

fixtures = root / "packages/web-decision/test/fixtures/qwen35-image-input"
processor = Gemma4ImageProcessorPil.from_pretrained(str(root.parent / "strands-e2b-v2"))
names = [
    "single-pixel.png",
    "rgba-alpha.png",
    "la-alpha.png",
    "palette-alpha.png",
    "gray-16bit.png",
    "rgb-16bit.png",
    "rgba-16bit.png",
    "grayscale.jpg",
    "cmyk.jpg",
    "gif-first-transparent.gif",
    "webp-alpha.webp",
    "aspect-rejected.png",
]
for fmt in ("png", "jpeg", "webp"):
    ext = {"png": "png", "jpeg": "jpg", "webp": "webp"}[fmt]
    names.extend(f"{fmt}-orientation-{i}.{ext}" for i in range(1, 9))
rows = []
for name in names:
    raw = (fixtures / name).read_bytes()
    encoded = base64.b64encode(raw).decode()
    header = header_image(encoded)
    image = decode_image(encoded)
    width, height = image.size
    resized = resize_image(image, *check_decoded_size(header, width, height))
    out = processor(images=resized, do_resize=False, return_tensors="pt")
    pixels = out["pixel_values"].to(torch.bfloat16).float().numpy().tobytes()
    positions = out["image_position_ids"].to(torch.int32).numpy().tobytes()
    rows.append(
        {
            "name": name,
            "inputSha256": hashlib.sha256(raw).hexdigest(),
            "width": width,
            "height": height,
            "tokens": header.tokens,
            "pixelsSha256": hashlib.sha256(pixels).hexdigest(),
            "positionsSha256": hashlib.sha256(positions).hexdigest(),
        }
    )
    print(name, flush=True)
(
    root / "packages/web-decision/test/fixtures/strands-gemma4-v2-encoded-image.json"
).write_text(json.dumps(rows, indent=2) + "\n")
