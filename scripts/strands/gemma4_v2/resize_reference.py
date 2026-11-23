"""Capture exact official width-then-height Pillow image resizing."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path("../strands-upstream-v2/src").resolve()))
from strands_decider.core.images import resize_image, resized_size

root = Path("../e2b-v2-resize-reference")
root.mkdir(exist_ok=True)
rng = np.random.default_rng(117)
sizes = [(1, 1), (7, 11), (31, 47), (121, 73), (640, 480), (280, 1), (1, 280), (48, 48)]
for width, height in sizes:
    data = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    outw, outh = resized_size(width, height)
    result = np.asarray(
        resize_image(Image.fromarray(data, "RGB"), outw, outh), dtype=np.uint8
    )
    (root / f"{width}x{height}.input").write_bytes(data.tobytes())
    (root / f"{width}x{height}.output").write_bytes(result.tobytes())
    print(width, height, outw, outh, flush=True)
