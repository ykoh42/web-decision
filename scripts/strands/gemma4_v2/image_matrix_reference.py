"""Generate CPU-only image input and insertion references from pinned upstream."""

import base64
import hashlib
import importlib
import io
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Gemma4ImageProcessorPil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT.parent / "strands-upstream-v2" / "src"))
image_module = importlib.import_module("strands_decider.images")
image_core = importlib.import_module("strands_decider.core.images")
decode_image = image_module.decode_image
header_image = image_core.decode_image
resize_image = image_core.resize_image
check_decoded_size = image_core.check_decoded_size

FIXTURES = ROOT / "packages/web-decision/test/fixtures/strands-gemma4-v2-matrix"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def image_png(width, height):
    if width * height > 2_000_000:
        image = Image.new("RGB", (width, height), (17, 93, 201))
    else:
        y, x = np.indices((height, width), dtype=np.uint32)
        pixels = np.stack(
            ((x * 13 + y * 3) % 256, (x * 7 + y * 11) % 256, (x * 5 + y * 19) % 256),
            axis=-1,
        ).astype(np.uint8)
        image = Image.fromarray(pixels, mode="RGB")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def image_reference():
    processor = Gemma4ImageProcessorPil.from_pretrained(
        str(ROOT.parent / "strands-e2b-v2")
    )
    cases = [
        (1, 1),
        (1, 279),
        (1, 280),
        (1, 281),
        (279, 1),
        (280, 1),
        (281, 1),
        (47, 49),
        (64, 64),
        (512, 256),
        (256, 512),
        (1024, 768),
        (4096, 1),
        (1, 4096),
        (4096, 4096),
    ]
    rows = []
    for width, height in cases:
        name = f"{width}x{height}.png"
        raw = image_png(width, height)
        (FIXTURES / name).write_bytes(raw)
        encoded = base64.b64encode(raw).decode()
        header = header_image(encoded)
        image = decode_image(encoded)
        resized = resize_image(image, *check_decoded_size(header, *image.size))
        out = processor(images=resized, do_resize=False, return_tensors="pt")
        pixels = out["pixel_values"].to(torch.bfloat16).float().numpy().tobytes()
        positions = out["image_position_ids"].to(torch.int32).numpy().tobytes()
        rows.append(
            {
                "name": name,
                "width": width,
                "height": height,
                "tokens": header.tokens,
                "inputSha256": sha(raw),
                "pixelsSha256": sha(pixels),
                "positionsSha256": sha(positions),
            }
        )
        print("official image", name, "tokens", header.tokens, flush=True)
    original = bytearray((FIXTURES / "1x1.png").read_bytes())
    original[16:24] = struct.pack(">II", 4097, 4096)
    original[29:33] = struct.pack(">I", zlib.crc32(original[12:29]))
    (FIXTURES / "over-pixel-limit.png").write_bytes(original)
    invalid = [
        "",
        "%%%",
        base64.b64encode(bytes(original)).decode(),
        base64.b64encode(b"\x89PNG\r\n\x1a\n").decode(),
    ]
    for entry in invalid:
        try:
            header_image(entry)
        except ValueError:
            continue
        raise ValueError("Invalid fixture was accepted by the official implementation")
    return {
        "images": rows,
        "invalid": invalid,
        "multi": [["1x1.png", "1x281.png", "281x1.png"], ["64x64.png", "256x512.png"]],
    }


def insertion_reference():
    cases = [
        {"name": "single", "n": 1, "spans": [[0, 1]]},
        {"name": "left", "n": 7, "spans": [[0, 1]]},
        {"name": "right", "n": 7, "spans": [[6, 1]]},
        {"name": "multiple", "n": 12, "spans": [[1, 2], [8, 3]]},
        {"name": "all-280", "n": 280, "spans": [[0, 280]]},
    ]
    for item in cases:
        n = item["n"]
        hidden = (
            (((np.arange(n * 1536, dtype=np.float64) % 113) - 56) / 37)
            .astype(np.float32)
            .reshape(1, n, 1536)
        )
        for index, (start, count) in enumerate(item["spans"]):
            image = (
                (
                    (((np.arange(count * 1536, dtype=np.float64) % 89) - 44) / 17)
                    + index * 0.5
                )
                .astype(np.float32)
                .reshape(count, 1536)
            )
            hidden[0, start : start + count] = (
                torch.from_numpy(image).to(torch.bfloat16).float().numpy()
            )
        item["sha256"] = sha(hidden.tobytes())
    return cases


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    result = image_reference()
    result["insertions"] = insertion_reference()
    (FIXTURES / "reference.json").write_text(json.dumps(result, indent=2) + "\n")
    print("CPU-only official references generated", flush=True)


if __name__ == "__main__":
    main()
