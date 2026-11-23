"""Verify vision graph argument order against the pinned original checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def verify_parameters(index, row, names, reference, weights, binding):
    parameters = []
    for position, (name, expected) in enumerate(
        zip(names, row["weights"], strict=True)
    ):
        if expected["name"] != f"weight-{position}.bin":
            raise ValueError("Unexpected layer input order")
        if file_digest(weights / str(index) / expected["name"]) != expected["sha256"]:
            raise ValueError("Vision weight file digest differs")
        if index == 0 and (
            expected["shape"] != reference[position]["shape"]
            or expected["sha256"] != reference[position]["sha256"]
        ):
            raise ValueError("Graph argument order differs from layer zero")
        parameters.append(
            binding(f"model.vision_tower.encoder.layers.{index}.{name}", expected)
        )
    return parameters


def verify_layers(records, names, reference, weights, binding):
    layers = []
    for index, row in enumerate(records):
        if row["layer"] != index or len(row["weights"]) != 41:
            raise ValueError(f"Invalid vision layer {index}")
        parameters = verify_parameters(index, row, names, reference, weights, binding)
        layers.append({"layer": index, "parameters": parameters})

    return layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("graph", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("patch", type=Path)
    parser.add_argument("projection", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    inventory_file = Path(
        "packages/web-decision/assets/strands-gemma4-v2/source-inventory.json"
    )
    inventory_bytes = inventory_file.read_bytes()
    inventory = json.loads(inventory_bytes)
    checkpoint = args.checkpoint / "model.safetensors"
    if file_digest(checkpoint) != inventory["sha256"]:
        raise ValueError("Original checkpoint digest differs from the pinned inventory")
    names = json.loads((args.graph / "parameter-names.json").read_text())
    reference = json.loads((args.graph / "inputs.json").read_text())[5:]
    records = json.loads((args.weights / "manifest.json").read_text())
    if (
        len(names) != 41
        or len(set(names)) != 41
        or len(reference) != 41
        or len(records) != 16
    ):
        raise ValueError("Unexpected graph or weight inventory")

    with safe_open(checkpoint, framework="pt") as source:

        def binding(name, expected):
            metadata = inventory["tensors"][name]
            tensor = source.get_tensor(name)
            if metadata["dtype"] != "BF16" or list(tensor.shape) != metadata["shape"]:
                raise ValueError(f"Original tensor metadata differs: {name}")
            raw = tensor.float().numpy().tobytes()
            if (
                digest(raw) != expected["sha256"]
                or list(tensor.shape) != expected["shape"]
            ):
                raise ValueError(
                    f"Graph input differs from the original tensor: {name}"
                )
            return {
                "name": name,
                "shape": metadata["shape"],
                "dtype": "BF16",
                "float32Sha256": digest(raw),
            }

        patch_names = [
            "model.vision_tower.patch_embedder.position_embedding_table",
            "model.vision_tower.patch_embedder.input_proj.weight",
        ]
        patch_reference = json.loads((args.patch / "inputs.json").read_text())[2:]
        patch = [
            binding(name, expected)
            for name, expected in zip(patch_names, patch_reference, strict=True)
        ]
        projection_reference = json.loads(
            (args.projection / "inputs.json").read_text()
        )[1:]
        projection = [
            binding(
                "model.embed_vision.embedding_projection.weight",
                projection_reference[0],
            )
        ]
        if len(projection_reference) != 1:
            raise ValueError("Unexpected projection parameters")
        layers = verify_layers(records, names, reference, args.weights, binding)

    result = {
        "format": "web-decision-gemma4-v2-vision-bindings-v1",
        "repository": inventory["repository"],
        "revision": inventory["revision"],
        "sourceInventorySha256": digest(inventory_bytes),
        "sourceGraphs": {
            key: {
                "sha256": file_digest(path / "layer.wasm"),
                "bytes": (path / "layer.wasm").stat().st_size,
            }
            for key, path in (
                ("patch", args.patch),
                ("layer", args.graph),
                ("projection", args.projection),
            )
        },
        "patch": patch,
        "layers": layers,
        "projection": projection,
    }
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(
        f"Verified 16 layers and patch/projection against original source; manifest sha256 {file_digest(args.output)}"
    )


if __name__ == "__main__":
    main()
