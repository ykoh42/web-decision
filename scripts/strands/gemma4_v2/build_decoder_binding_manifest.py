"""Create the decoder binding contract from source-verified graph arguments."""

import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
checkout = Path(sys.argv[2])
if (
    "All 35 layer parameter bindings matched the pinned original"
    not in (root / "e2b-v2-layer-source-bindings.log").read_text()
):
    raise ValueError("Source binding verification is missing")
source = json.loads((root / "e2b-v2-source-inventory-hybrid.json").read_text())
bindings = json.loads((root / "e2b-v2-layer-source-bindings.json").read_text())
if len(bindings) != 35:
    raise ValueError("Expected all 35 decoder layers")
origins = {}
for family in (0, 4, 15, 19):
    binary = (root / f"e2b-v2-chunked-valid-family{family}/layer.wasm").read_bytes()
    origins[str(family)] = {
        "bytes": len(binary),
        "sha256": hashlib.sha256(binary).hexdigest(),
    }
layers = []
for row in bindings:
    layer = row["layer"]
    parameters = []
    for name in row["names"]:
        tensor = source["tensors"][name]
        if tensor["dtype"] != "BF16" or tensor["length"] % 4:
            raise ValueError("Unexpected packed tensor")
        parameters.append(
            {
                "name": name,
                "shape": tensor["shape"],
                "dtype": tensor["dtype"],
                "bytes": tensor["length"],
                "packedWords": tensor["length"] // 4,
            }
        )
    scalar = f"model.language_model.layers.{layer}.layer_scalar"
    if source["tensors"][scalar]["shape"] != [1]:
        raise ValueError("Unexpected layer scalar")
    layers.append(
        {
            "layer": layer,
            "family": row["family"],
            "scalar": scalar,
            "parameters": parameters,
        }
    )
manifest = {
    "format": "web-decision-gemma4-v2-decoder-bindings-v1",
    "repository": source["repository"],
    "revision": source["revision"],
    "sourceInventorySha256": hashlib.sha256(
        (root / "e2b-v2-source-inventory-hybrid.json").read_bytes()
    ).hexdigest(),
    "sourceGraphArtifacts": origins,
    "layers": layers,
}
output = (
    checkout / "packages/web-decision/assets/strands-gemma4-v2/decoder-bindings.json"
)
output.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
print(
    "Wrote decoder binding manifest",
    len(layers),
    "layers; sha256",
    hashlib.sha256(output.read_bytes()).hexdigest(),
)
