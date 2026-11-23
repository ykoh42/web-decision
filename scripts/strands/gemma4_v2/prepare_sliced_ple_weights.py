"""Prepare per-layer BF16 projection weights for the sliced PLE experiment."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

checkpoint = Path(sys.argv[1])
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
with safe_open(checkpoint / "model.safetensors", framework="pt") as f:
    weights = (
        f.get_tensor("model.language_model.per_layer_model_projection.weight")
        .float()
        .numpy()
    )
    norm = (
        f.get_tensor("model.language_model.per_layer_projection_norm.weight")
        .float()
        .numpy()
    )


def pack(value):
    bits = value.copy().view(np.uint32).reshape(-1)
    if np.any(bits & np.uint32(65535)):
        raise ValueError("Weights must be exact BF16")
    return (bits >> np.uint32(16)).astype(np.uint16).view(np.uint32)


manifest = {}
for layer in range(35):
    data = pack(weights[layer * 256 : (layer + 1) * 256]).tobytes()
    (output / f"{layer}.bin").write_bytes(data)
    manifest[str(layer)] = hashlib.sha256(data).hexdigest()
data = pack(norm).tobytes()
(output / "norm.bin").write_bytes(data)
manifest["norm"] = hashlib.sha256(data).hexdigest()
(output / "manifest.json").write_text(json.dumps(manifest) + "\n")
print("prepared 35 layer PLE weight slices")
