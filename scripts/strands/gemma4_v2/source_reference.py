"""Independent source-row oracle from the original checkpoint."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ids = [0, 1, 63, 64, 131071, 131072, 262143]
root = Path(sys.argv[1])
result = {"ids": ids}
with safe_open(root / "model.safetensors", framework="pt") as f:
    word = f.get_slice("model.language_model.embed_tokens.weight")
    ple = f.get_slice("model.language_model.embed_tokens_per_layer.weight")
    for layer in (None, 0, 17, 34):
        rows = [
            (
                word[i : i + 1, :]
                if layer is None
                else ple[i : i + 1, layer * 256 : (layer + 1) * 256]
            )
            .view(torch.uint16)
            .numpy()
            .reshape(-1)
            for i in ids
        ]
        result["word" if layer is None else str(layer)] = hashlib.sha256(
            np.stack(rows).tobytes()
        ).hexdigest()
    norm = (
        f.get_tensor("model.language_model.per_layer_projection_norm.weight")
        .view(torch.uint16)
        .numpy()
        .tobytes()
    )
    result["norm"] = hashlib.sha256(norm).hexdigest()
Path("../e2b-v2-source-reference.json").write_text(json.dumps(result) + "\n")
print("Source row oracle prepared")
