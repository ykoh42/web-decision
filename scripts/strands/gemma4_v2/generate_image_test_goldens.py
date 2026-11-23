"""Generate compact goldens from the pinned upstream Gemma 4 image processor."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Gemma4ImageProcessorPil

sys.path.insert(0, str(Path("../strands-upstream-v2/src").resolve()))
from strands_decider.core.images import resize_image, resized_size

processor = Gemma4ImageProcessorPil.from_pretrained("../strands-e2b-v2")
result = []
for width, height in [
    (1, 1),
    (7, 11),
    (121, 73),
    (640, 480),
    (280, 1),
    (1, 280),
    (48, 48),
]:
    i = np.arange(width * height * 3, dtype=np.uint64)
    data = (
        ((i * 29 + (i // 17) * 31 + 7) % 256).astype(np.uint8).reshape(height, width, 3)
    )
    resized = resize_image(Image.fromarray(data, "RGB"), *resized_size(width, height))
    output = processor(images=resized, do_resize=False, return_tensors="pt")
    pixels = output["pixel_values"].to(torch.bfloat16).float().numpy().tobytes()
    positions = output["image_position_ids"].to(torch.int32).numpy().tobytes()
    w, h = resized.size
    result.append(
        {
            "width": width,
            "height": height,
            "resizedWidth": w,
            "resizedHeight": h,
            "tokens": w * h // (48 * 48),
            "pixelsSha256": hashlib.sha256(pixels).hexdigest(),
            "positionsSha256": hashlib.sha256(positions).hexdigest(),
        }
    )
Path("packages/web-decision/test/fixtures/strands-gemma4-v2-image.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print("captured", len(result), "image test goldens")
