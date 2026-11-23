"""Prepare variable-length cases for the reusable prefix-attention kernel."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("kernel", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    base = {
        name: torch.from_numpy(
            np.fromfile(args.fixture / f"{name}.bin", np.float32)
            .reshape(1, heads, 80, 256)
            .copy()
        )
        for name, heads in (("query", 8), ("key", 1), ("value", 1))
    }
    cases = ((31, 13), (32, 16), (33, 16), (63, 31), (64, 32), (65, 32), (81, 33))
    args.output.mkdir(parents=True, exist_ok=True)
    for length, split in cases:
        tensors = {
            name: tensor.repeat(1, 1, (length + 79) // 80, 1)[
                :, :, :length
            ].contiguous()
            for name, tensor in base.items()
        }
        valid = torch.ones(1, length, dtype=torch.int32)
        if length > 2:
            valid[:, 1::37] = 0
        query, key, value = (
            tensors[name].to("mps") for name in ("query", "key", "value")
        )
        with torch.inference_mode():
            scores = query @ key.transpose(-1, -2)
            scores = scores.masked_fill(valid.to("mps")[:, None, None, :] == 0, -1e30)
            if split:
                scores[:, :, :split, split:] = -1e30
            expected = (
                (torch.softmax(scores, dim=-1) @ value).transpose(1, 2).cpu().numpy()
            )
        arrays = [
            tensors["query"].transpose(1, 2).contiguous().numpy(),
            tensors["key"].transpose(1, 2).contiguous().numpy(),
            tensors["value"].transpose(1, 2).contiguous().numpy(),
            valid.numpy(),
            np.array([split], dtype=np.int32),
        ]
        directory = args.output / f"{length}-{split}"
        directory.mkdir(exist_ok=True)
        (directory / "layer.wasm").symlink_to(args.kernel.resolve())
        records = []
        for index, array in enumerate(arrays):
            raw = array.tobytes()
            name = f"input-{index}.bin"
            (directory / name).write_bytes(raw)
            records.append(
                {
                    "name": name,
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        (directory / "inputs.json").write_text(json.dumps(records) + "\n")
        (directory / "output-shape.json").write_text(
            json.dumps(list(expected.shape)) + "\n"
        )
        expected.tofile(directory / "expected.bin")
        print(f"Prepared length={length} split={split}", flush=True)
        del query, key, value, scores, expected
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
