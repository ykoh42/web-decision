"""Hash the pinned original checkpoint in ranges suitable for browser loading."""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

MODEL_SHA256 = "593a28a67754653c7ffffff80a5186f18966cc6262c2f4f097414c187e5fd48c"
REPOSITORY = "StrandsAgents/strands-decider-E2B-gemma4-v2-2611"
REVISION = "a5f32ea1bffe2e91c25c0c641d444e8115e28212"
CHUNK = 4 * 1024 * 1024
EMBEDDINGS = (
    "model.language_model.embed_tokens.weight",
    "model.language_model.embed_tokens_per_layer.weight",
)


def read_header(file):
    header_size = struct.unpack("<Q", file.read(8))[0]
    if header_size > 16 * 1024 * 1024:
        raise ValueError("Unexpected safetensors header size")
    header = json.loads(file.read(header_size))
    base = 8 + header_size
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if info["dtype"] != "BF16":
            raise ValueError("Unexpected tensor dtype")
        start, end = info["data_offsets"]
        tensors[name] = {
            "offset": base + start,
            "length": end - start,
            "shape": info["shape"],
            "dtype": info["dtype"],
        }
    return base, tensors


def make_segments(tensors, total):
    tables = []
    for name in EMBEDDINGS:
        tensor = tensors[name]
        shape = tensor["shape"]
        if len(shape) != 2 or tensor["length"] != math.prod(shape) * 2:
            raise ValueError("Unexpected embedding shape")
        tables.append(
            (tensor["offset"], tensor["offset"] + tensor["length"], shape[1] * 2 * 64)
        )
    tables.sort()
    segments = []
    cursor = 0
    for start, end, block in tables:
        if not cursor <= start < end <= total or block > CHUNK:
            raise ValueError("Unexpected embedding offsets")
        segments.extend(((cursor, start, CHUNK), (start, end, block)))
        cursor = end
    segments.append((cursor, total, CHUNK))
    return segments


def hash_segments(file, segments):
    file.seek(0)
    complete = hashlib.sha256()
    chunks = []
    read = 0
    for start, end, block in segments:
        if file.tell() != start:
            raise ValueError("Unexpected file position")
        while file.tell() < end:
            offset = file.tell()
            data = file.read(min(block, end - offset))
            if not data:
                raise ValueError("Truncated checkpoint")
            complete.update(data)
            chunks.append([offset, len(data), hashlib.sha256(data).hexdigest()])
            read += len(data)
            if len(chunks) % 2048 == 0:
                print("hashed MiB:", read // (1024 * 1024), flush=True)
    return complete.hexdigest(), chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    model = args.checkpoint / "model.safetensors"
    total = model.stat().st_size
    with model.open("rb") as file:
        base, tensors = read_header(file)
        digest, chunks = hash_segments(file, make_segments(tensors, total))
    if digest != MODEL_SHA256:
        raise ValueError("Checkpoint SHA-256 differs from pinned source")
    result = {
        "format": "web-decision-strands-gemma4-v2-source-v1",
        "repository": REPOSITORY,
        "revision": REVISION,
        "file": "model.safetensors",
        "bytes": total,
        "sha256": MODEL_SHA256,
        "headerBytes": base,
        "chunks": chunks,
        "tensors": tensors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(
        "Verified checkpoint and wrote",
        len(chunks),
        "range hashes for",
        len(tensors),
        "tensors",
        flush=True,
    )


if __name__ == "__main__":
    main()
