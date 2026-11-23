"""Verify that reusable graph arguments match the pinned source for every layer."""

import ast
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
inventory = json.loads((root / "e2b-v2-source-inventory-hybrid.json").read_text())


def verify_records(source, names, records, inventory):
    for name, record in zip(names, records, strict=True):
        tensor = inventory["tensors"][name]
        source.seek(tensor["offset"])
        digest = hashlib.sha256()
        remaining = tensor["length"]
        while remaining:
            chunk = source.read(min(4 * 1024 * 1024, remaining))
            if not chunk:
                raise ValueError("Truncated source")
            digest.update(chunk)
            remaining -= len(chunk)
        if digest.hexdigest() != record["sha256"]:
            raise ValueError(f"Parameter digest mismatch for {name}")


def read_families(root):
    families = {}
    for family in (0, 4, 15, 19):
        lines = (
            (root / f"e2b-v2-chunked-valid-family{family}-build.log")
            .read_text()
            .splitlines()
        )
        family_line = next(line for line in lines if line.startswith("Parameters: "))
        families[family] = [
            name.removeprefix("block.")
            for name in ast.literal_eval(family_line.removeprefix("Parameters: "))
        ]
    return families


def verify(root, checkpoint, inventory):
    families = read_families(root)
    output = []
    with (checkpoint / "model.safetensors").open("rb") as source:
        for layer in range(35):
            global_layer = layer % 5 == 4
            family = (
                (4 if global_layer else 0)
                if layer < 15
                else (19 if global_layer else 15)
            )
            names = [
                f"model.language_model.layers.{layer}.{name}"
                for name in families[family]
            ]
            records = json.loads(
                (root / f"e2b-v2-layer{layer}-packed-webgpu/inputs.json").read_text()
            )[4 if layer < 15 else 6 :]
            if len(names) != len(records):
                raise ValueError(f"Parameter count mismatch for layer {layer}")
            verify_records(source, names, records, inventory)
            output.append({"layer": layer, "family": family, "names": names})
            if layer % 5 == 4:
                print("verified through layer", layer, flush=True)
    return output


output = verify(root, checkpoint, inventory)
(root / "e2b-v2-layer-source-bindings.json").write_text(json.dumps(output) + "\n")
print("All 35 layer parameter bindings matched the pinned original")
