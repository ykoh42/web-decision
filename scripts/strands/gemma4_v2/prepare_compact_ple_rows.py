"""Prepare only observed token rows and a single layer at a time for the test."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

checkpoint = Path(sys.argv[1])
capture = Path(sys.argv[2])
output = Path(sys.argv[3])
output.mkdir(parents=True, exist_ok=True)
sample = json.loads((capture / "tokens.json").read_text())
ids = sample["state"] + sample["questions"][0]
ids = [0 if i == 258880 else i for i in ids]
unique = list(dict.fromkeys(ids))
mapping = {value: i for i, value in enumerate(unique)}
indices = np.asarray([mapping[i] for i in ids], dtype=np.int32)
data = indices.tobytes()
(output / "indices.bin").write_bytes(data)
manifest = {
    "unique": len(unique),
    "length": len(ids),
    "indices": hashlib.sha256(data).hexdigest(),
    "layers": {},
}
with safe_open(checkpoint / "model.safetensors", framework="pt") as f:
    embedding = f.get_slice("model.language_model.embed_tokens_per_layer.weight")
    for layer in range(35):
        rows = [
            embedding[i : i + 1, layer * 256 : (layer + 1) * 256]
            .view(torch.uint16)
            .numpy()
            .reshape(256)
            for i in unique
        ]
        data = np.stack(rows).copy().view(np.uint32).tobytes()
        (output / f"{layer}.bin").write_bytes(data)
        manifest["layers"][str(layer)] = hashlib.sha256(data).hexdigest()
(output / "manifest.json").write_text(json.dumps(manifest) + "\n")
print("prepared", len(ids), "tokens,", len(unique), "unique rows in 35 layers")
