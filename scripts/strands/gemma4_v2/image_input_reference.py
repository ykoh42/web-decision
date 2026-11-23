"""Capture official normalized patches after the upstream's two-pass resize."""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Gemma4ImageProcessorPil

sys.path.insert(0, str(Path("../strands-upstream-v2/src").resolve()))
from strands_decider.core.images import resize_image, resized_size

root = Path("../e2b-v2-resize-reference")
processor = Gemma4ImageProcessorPil.from_pretrained("../strands-e2b-v2")
for width, height in [
    (1, 1),
    (7, 11),
    (31, 47),
    (121, 73),
    (640, 480),
    (280, 1),
    (1, 280),
    (48, 48),
]:
    source = (
        np.frombuffer((root / f"{width}x{height}.input").read_bytes(), dtype=np.uint8)
        .reshape(height, width, 3)
        .copy()
    )
    resized = resize_image(Image.fromarray(source, "RGB"), *resized_size(width, height))
    outputs = processor(images=resized, do_resize=False, return_tensors="pt")
    outputs["pixel_values"].to(torch.bfloat16).float().numpy().tofile(
        root / f"{width}x{height}.pixels"
    )
    outputs["image_position_ids"].to(torch.int32).numpy().tofile(
        root / f"{width}x{height}.positions"
    )
    print("prepared", width, height, flush=True)
