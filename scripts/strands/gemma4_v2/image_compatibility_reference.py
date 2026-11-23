"""Record pinned Pillow and Gemma processor behavior for image compatibility cases."""

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

fixtures = (
    root / "packages/web-decision/test/fixtures/strands-gemma4-v2-image-compatibility"
)
processor = Gemma4ImageProcessorPil.from_pretrained(str(root.parent / "strands-e2b-v2"))
rows = []
for path in sorted(
    path
    for path in fixtures.iterdir()
    if path.suffix in (".png", ".jpg", ".webp", ".gif")
):
    encoded = base64.b64encode(path.read_bytes()).decode()
    header = header_image(encoded)
    image = decode_image(encoded)
    resized = resize_image(image, *check_decoded_size(header, *image.size))
    result = processor(images=resized, do_resize=False, return_tensors="pt")
    pixels = result["pixel_values"].to(torch.bfloat16).float().numpy().tobytes()
    positions = result["image_position_ids"].to(torch.int32).numpy().tobytes()
    rows.append(
        {
            "name": path.name,
            "width": image.width,
            "height": image.height,
            "tokens": header.tokens,
            "rgbSha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "pixelsSha256": hashlib.sha256(pixels).hexdigest(),
            "positionsSha256": hashlib.sha256(positions).hexdigest(),
        }
    )
(fixtures / "reference.json").write_text(json.dumps(rows, indent=2) + "\n")
print(f"Captured {len(rows)} original Pillow image cases")
