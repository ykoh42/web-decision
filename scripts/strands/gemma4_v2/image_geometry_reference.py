import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("../strands-upstream-v2/src").resolve()))
from strands_decider.core.images import resized_size, soft_tokens

sizes = [
    (1, 1),
    (1, 279),
    (1, 280),
    (1, 281),
    (279, 1),
    (280, 1),
    (281, 1),
    (1, 4096),
    (4096, 1),
    (2, 10),
    (10, 2),
    (64, 64),
    (512, 512),
    (1920, 1080),
    (1080, 1920),
    (4096, 4096),
]
sizes += [
    (a, b)
    for a in (3, 47, 48, 49, 255, 256, 257, 1023, 1024, 1025)
    for b in (1, 3, 47, 48, 49, 255, 256, 257, 1023, 1024, 1025)
]
rows = []
for width, height in sizes:
    w, h = resized_size(width, height)
    rows.append(
        {
            "input": [width, height],
            "output": {"width": w, "height": h, "tokens": soft_tokens(width, height)},
        }
    )
Path("../e2b-v2-image-geometry-reference.json").write_text(json.dumps(rows))
print("captured", len(rows), "official image dimensions")
