"""Extract real E2B vision-layer weights in the verified graph input order."""

import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("graph", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    path = args.checkpoint / "model.safetensors"
    with path.open("rb") as file:
        if (
            hashlib.file_digest(file, "sha256").hexdigest()
            != "593a28a67754653c7ffffff80a5186f18966cc6262c2f4f097414c187e5fd48c"
        ):
            raise ValueError("Unexpected source checkpoint")
    names = json.loads((args.graph / "parameter-names.json").read_text())
    reference = json.loads((args.graph / "inputs.json").read_text())[5:]
    if len(names) != len(reference) or len(set(names)) != len(names):
        raise ValueError("Invalid graph parameter list")
    manifest = []
    args.output.mkdir(parents=True, exist_ok=True)
    with safe_open(path, framework="pt") as file:
        for layer in range(16):
            directory = args.output / str(layer)
            directory.mkdir(exist_ok=True)
            records = []
            for index, (name, item) in enumerate(zip(names, reference)):
                source = f"model.vision_tower.encoder.layers.{layer}.{name}"
                array = file.get_tensor(source).float().numpy()
                if list(array.shape) != item["shape"]:
                    raise ValueError(f"Unexpected shape: {source}")
                raw = array.tobytes()
                digest = hashlib.sha256(raw).hexdigest()
                if layer == 0 and digest != item["sha256"]:
                    raise ValueError(f"Graph parameter order mismatch: {source}")
                filename = f"weight-{index}.bin"
                (directory / filename).write_bytes(raw)
                records.append(
                    {"name": filename, "shape": list(array.shape), "sha256": digest}
                )
            manifest.append({"layer": layer, "weights": records})
    (args.output / "manifest.json").write_text(json.dumps(manifest) + "\n")
    print(
        f"Prepared {len(manifest)} checkpoint layers with {len(names)} parameters each",
        flush=True,
    )


if __name__ == "__main__":
    main()
