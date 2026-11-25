"""Verify the original Gemma v1 checkpoint files and index their remote ranges."""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

from causal_layer_ir import ADAPTER_FILES, BASE_FILES

BASE_REPOSITORY = "google/gemma-4-E2B-it"
BASE_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
ADAPTER_REPOSITORY = "StrandsAgents/strands-decider-E2B-gemma4-v1-2610"
ADAPTER_REVISION = "d383ecf8534b13d49b3b567a1b66f88fca6d50ba"
SOURCE_INVENTORY_SHA256 = (
    "eb953da8adbbddce431abf40bbc3f2fe6f217a83255ee4c41001d760b9a98c87"
)
CHUNK_BYTES = 4 * 1024 * 1024
EMBEDDINGS = (
    "model.language_model.embed_tokens.weight",
    "model.language_model.embed_tokens_per_layer.weight",
)


def read_header(file, total, dtype):
    length = struct.unpack("<Q", file.read(8))[0]
    if length > 16 * 1024 * 1024 or length + 8 > total:
        raise ValueError("Invalid safetensors header size")
    raw = file.read(length)
    if len(raw) != length:
        raise ValueError("Truncated safetensors header")
    base = length + 8
    tensors = {}
    for name, info in json.loads(raw).items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        shape = info["shape"]
        width = {"BF16": 2, "F32": 4}[dtype]
        if (
            info["dtype"] != dtype
            or not 0 <= start < end <= total - base
            or end - start != math.prod(shape) * width
        ):
            raise ValueError(f"Invalid pinned tensor {name}")
        tensors[name] = {
            "offset": base + start,
            "length": end - start,
            "shape": shape,
            "dtype": dtype,
        }
    cursor = base
    for info in sorted(tensors.values(), key=lambda tensor: tensor["offset"]):
        if info["offset"] != cursor:
            raise ValueError("Invalid safetensors tensor coverage")
        cursor += info["length"]
    if cursor != total:
        raise ValueError("Safetensors tensors do not cover the complete file")
    return base, tensors


def segments(tensors, total, embedding_names):
    cursor = 0
    for tensor in sorted(
        (tensors[name] for name in embedding_names), key=lambda tensor: tensor["offset"]
    ):
        offset = tensor["offset"]
        _, columns = tensor["shape"]
        yield cursor, offset, CHUNK_BYTES
        cursor = offset + tensor["length"]
        yield offset, cursor, columns * 2 * 64
    yield cursor, total, CHUNK_BYTES


def inspect_source(
    local_file, name, repository, revision, expected, dtype, embedding_names=()
):
    file_sha = hashlib.sha256()
    chunks = []
    with local_file.open("rb") as file:
        total = local_file.stat().st_size
        header, tensors = read_header(file, total, dtype)
        file.seek(0)
        for start, end, step in segments(tensors, total, embedding_names):
            for offset in range(start, end, step):
                length = min(step, end - offset)
                data = file.read(length)
                if len(data) != length:
                    raise ValueError("Truncated original file")
                file_sha.update(data)
                chunks.append([offset, length, hashlib.sha256(data).hexdigest()])
        if file.read(1) or file_sha.hexdigest() != expected:
            raise ValueError(f"Original SHA-256 mismatch for {name}")
    return {
        "repository": repository,
        "revision": revision,
        "file": name,
        "bytes": total,
        "sha256": expected,
        "headerBytes": header,
        "chunks": chunks,
        "tensors": tensors,
    }


def build_inventory(base, checkpoint):
    return {
        "format": "web-decision-gemma4-e2b-v1-original-ranges-v1",
        "base": inspect_source(
            base / "model.safetensors",
            "model.safetensors",
            BASE_REPOSITORY,
            BASE_REVISION,
            BASE_FILES["model.safetensors"],
            "BF16",
            EMBEDDINGS,
        ),
        "adapter": inspect_source(
            checkpoint / "lora/adapter_model.safetensors",
            "lora/adapter_model.safetensors",
            ADAPTER_REPOSITORY,
            ADAPTER_REVISION,
            ADAPTER_FILES["adapter_model.safetensors"],
            "F32",
        ),
    }


def read_pinned_inventory(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_INVENTORY_SHA256:
        raise ValueError(
            "The Gemma v1 source inventory does not match the pinned ranges"
        )
    return raw, json.loads(raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = build_inventory(args.base, args.checkpoint)
    with args.output.open("x") as stream:
        json.dump(result, stream, separators=(",", ":"))
        stream.write("\n")
