"""Build a dynamic, compact word embedding lookup from the pinned checkpoint."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import tvm
from safetensors import safe_open
from tvm import relax
from tvm.script import ir as I
from tvm.script import tirx as T


@I.ir_module(s_tir=True)
class Lookup:
    @T.prim_func(s_tir=True)
    def word(rows_h: T.handle, ids_h: T.handle, out_h: T.handle):
        n = T.int32()
        u = T.int32()
        rows = T.match_buffer(rows_h, (u, 768), "uint32")
        ids = T.match_buffer(ids_h, (n,), "int32")
        out = T.match_buffer(out_h, (1, n, 1536), "float32")
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        for chunk in T.thread_binding(T.ceildiv(n, 32768), thread="blockIdx.y"):
            for pos in T.thread_binding(T.min(n, 32768), thread="blockIdx.x"):
                for lane in T.thread_binding(256, thread="threadIdx.x"):
                    row = chunk * 32768 + pos
                    if row < n:
                        for group in range(6):
                            column = group * 256 + lane
                            word = rows[ids[row], column // 2]
                            half = T.if_then_else(
                                column % 2 == 0,
                                word & T.uint32(65535),
                                word >> T.uint32(16),
                            )
                            value = T.reinterpret(
                                "float32", half << T.uint32(16)
                            ) * T.float32(39.25)
                            bits = T.reinterpret("uint32", value)
                            rounded = (
                                bits
                                + T.uint32(32767)
                                + ((bits >> T.uint32(16)) & T.uint32(1))
                            ) & T.uint32(4294901760)
                            out[0, row, column] = T.reinterpret("float32", rounded)


def build():
    n = tvm.tirx.Var("length", "int64")
    u = tvm.tirx.Var("unique", "int64")
    rows = relax.Var("rows", relax.TensorType((u, 768), "uint32"))
    ids = relax.Var("ids", relax.TensorType((n,), "int32"))
    builder = relax.BlockBuilder(Lookup)
    with builder.function(
        "packed_main",
        [rows, ids],
        attrs={
            "tir_var_lower_bound": {"length": 1, "unique": 1},
            "tir_var_upper_bound": {"length": 131072, "unique": 262144},
        },
    ):
        with builder.dataflow():
            output = builder.emit(
                relax.call_tir(
                    builder.get().get_global_var("word"),
                    (rows, ids),
                    relax.TensorType((1, n, 1536), "float32"),
                )
            )
            output = builder.emit_output(output)
        builder.emit_func_output(output)
    return builder.get()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("capture", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--tvm-source", type=Path, required=True)
    p.add_argument("--emsdk", type=Path, required=True)
    a = p.parse_args()
    tokens = json.loads((a.capture / "tokens.json").read_text())
    ids = tokens["state"] + tokens["questions"][0]
    unique = list(dict.fromkeys(ids))
    lookup = {value: i for i, value in enumerate(unique)}
    indices = np.asarray([lookup[x] for x in ids], np.int32)
    with safe_open(a.checkpoint / "model.safetensors", framework="pt") as f:
        table = f.get_slice("model.language_model.embed_tokens.weight")
        rows = (
            np.stack(
                [
                    table[i : i + 1, :].view(torch.uint16).numpy().reshape(1536)
                    for i in unique
                ]
            )
            .copy()
            .view(np.uint32)
        )
    module = build()
    target = tvm.target.Target(
        "webgpu", host={"kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"}
    )
    with tvm.transform.PassContext(disabled_pass=["StaticPlanBlockMemory"]):
        executable = relax.build(module, target=target)
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
    for length in (2, 31, len(ids)):
        d = a.output / str(length)
        d.mkdir(exist_ok=True)
        (d / "layer.wasm").symlink_to("../layer.wasm")
        records = []
        for i, array in enumerate((rows, indices[:length])):
            raw = array.tobytes()
            name = f"input-{i}.bin"
            (d / name).write_bytes(raw)
            records.append(
                {
                    "name": name,
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        (d / "inputs.json").write_text(json.dumps(records))
        (d / "output-shape.json").write_text(json.dumps([1, length, 1536]))
        expected = torch.load(a.capture / "layer0.pt", weights_only=True)[
            "hidden"
        ].numpy()[:, :length, :]
        expected.copy().tofile(d / "expected.bin")
    print("Dynamic word lookup exported", flush=True)


if __name__ == "__main__":
    main()
