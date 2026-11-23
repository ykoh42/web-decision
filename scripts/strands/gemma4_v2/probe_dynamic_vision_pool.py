"""Export model-local dynamic image pooling and compare varied image shapes."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
import tvm
from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionPooler
from tvm import relax
from tvm.script import ir as I
from tvm.script import tirx as T


@I.ir_module(s_tir=True)
class Pool:
    @T.prim_func(s_tir=True)
    def pool(
        hidden: T.Buffer((1, 2520, 768), "float32"),
        indices_h: T.handle,
        output_h: T.handle,
    ):
        n = T.int32()
        indices = T.match_buffer(indices_h, (n, 9), "int32")
        output = T.match_buffer(output_h, (n, 768), "float32")
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        for block in T.thread_binding(n * 3, thread="blockIdx.x"):
            for thread in T.thread_binding(256, thread="threadIdx.x"):
                index = block * 256 + thread
                row, column = index // 768, index % 768
                total = T.alloc_buffer((1,), "float32", scope="local")
                total[0] = T.float32(0)
                for item in range(9):
                    total[0] = total[0] + hidden[
                        0, indices[row, item], column
                    ] * T.float32(1 / 9)
                output[row, column] = total[0] * T.float32(768**0.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--tvm-source", type=Path, required=True)
    p.add_argument("--emsdk", type=Path, required=True)
    a = p.parse_args()
    n = tvm.tirx.Var("n", "int64")
    builder = relax.BlockBuilder(Pool)
    hidden = relax.Var("hidden", relax.TensorType((1, 2520, 768), "float32"))
    indices = relax.Var("indices", relax.TensorType((n, 9), "int32"))
    with builder.function(
        "main",
        [hidden, indices],
        attrs={"tir_var_lower_bound": {"n": 1}, "tir_var_upper_bound": {"n": 280}},
    ):
        with builder.dataflow():
            result = builder.emit(
                relax.call_tir(
                    builder.get().get_global_var("pool"),
                    (hidden, indices),
                    relax.TensorType((n, 768), "float32"),
                )
            )
            result = builder.emit_output(result)
        builder.emit_func_output(result)
    target = tvm.target.Target(
        "webgpu", host={"kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"}
    )
    with tvm.transform.PassContext(disabled_pass=["StaticPlanBlockMemory"]):
        executable = relax.build(builder.get(), target=target)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen35"))
    from qwen35_tvm_build import (
        export_tvmjs,
        verify_emsdk_checkout,
        verify_tvm_checkout,
    )

    verify_tvm_checkout(a.tvm_source)
    verify_emsdk_checkout(a.emsdk)
    a.output.mkdir(parents=True, exist_ok=True)
    export_tvmjs(executable, a.output / "layer.wasm", a.emsdk, a.tvm_source)
    values = np.random.default_rng(41).normal(size=(1, 2520, 768)).astype(np.float32)
    pooler = Gemma4VisionPooler(
        transformers.AutoConfig.from_pretrained(str(a.checkpoint)).vision_config
    ).to("mps")
    for width, height in ((1, 1), (1, 2), (4, 5), (16, 16), (1, 280), (14, 20)):
        coordinates = [(x, y) for y in range(height * 3) for x in range(width * 3)]
        coordinates += [(-1, -1)] * (2520 - len(coordinates))
        pos = torch.tensor([coordinates], dtype=torch.int32, device="mps")
        with torch.inference_mode():
            out, valid = pooler(
                torch.from_numpy(values).to("mps"),
                pos,
                (pos == -1).all(dim=-1),
                output_length=280,
            )
            expected = out[valid].cpu().numpy()
        groups = [[] for _ in range(width * height)]
        for i, (x, y) in enumerate(coordinates):
            if x >= 0:
                groups[x // 3 + width * (y // 3)].append(i)
        assert expected.shape == (width * height, 768) and all(
            len(x) == 9 for x in groups
        )
        directory = a.output / f"{width}x{height}"
        directory.mkdir(exist_ok=True)
        (directory / "layer.wasm").symlink_to("../layer.wasm")
        records = []
        for i, array in enumerate((values, np.asarray(groups, dtype=np.int32))):
            raw = array.tobytes()
            name = f"input-{i}.bin"
            (directory / name).write_bytes(raw)
            records.append(
                {
                    "name": name,
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        (directory / "inputs.json").write_text(json.dumps(records))
        (directory / "output-shape.json").write_text(json.dumps(list(expected.shape)))
        expected.tofile(directory / "expected.bin")
        print("prepared", width, height, flush=True)


if __name__ == "__main__":
    main()
