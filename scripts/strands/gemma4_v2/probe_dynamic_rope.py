"""Dynamic RoPE with BF16 rounding, built using model-local TVM kernels."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import tvm
from tvm import relax
from tvm.script import tirx as T


@T.macro
def rotary_row(ids, freq, out, row, lane, dim, is_sine):
    for group in range(dim // 256):
        col = group * 256 + lane
        angle = T.cast(ids[row], "float32") * freq[col % (dim // 2)]
        if is_sine:
            value = T.sin(angle)
        else:
            value = T.cos(angle)
        bits = T.reinterpret("uint32", value)
        rounded = (
            bits + T.uint32(32767) + ((bits >> T.uint32(16)) & T.uint32(1))
        ) & T.uint32(4294901760)
        out[0, row, col] = T.reinterpret("float32", rounded)


def kernel(dim, is_sine):
    @T.prim_func(s_tir=True)
    def rotary(
        ids_h: T.handle, freq: T.Buffer((dim // 2,), "float32"), out_h: T.handle
    ):
        n = T.int32()
        ids = T.match_buffer(ids_h, (n,), "int32")
        out = T.match_buffer(out_h, (1, n, dim), "float32")
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        for chunk in T.thread_binding(T.ceildiv(n, 32768), thread="blockIdx.y"):
            for pos in T.thread_binding(T.min(n, 32768), thread="blockIdx.x"):
                for lane in T.thread_binding(256, thread="threadIdx.x"):
                    row = chunk * 32768 + pos
                    if row < n:
                        rotary_row(ids, freq, out, row, lane, dim, is_sine)

    return rotary


def build():
    n = tvm.tirx.Var("length", "int64")
    bb = relax.BlockBuilder()
    for name, dim, sine in [
        ("local_cos", 256, False),
        ("local_sin", 256, True),
        ("global_cos", 512, False),
        ("global_sin", 512, True),
    ]:
        func = bb.add_func(
            kernel(dim, sine).with_attr("global_symbol", name + "_kernel"),
            name + "_kernel",
        )
        ids = relax.Var("ids", relax.TensorType((n,), "int32"))
        freq = relax.Var("freq", relax.TensorType((dim // 2,), "float32"))
        with bb.function(
            name,
            [ids, freq],
            attrs={
                "tir_var_lower_bound": {"length": 1},
                "tir_var_upper_bound": {"length": 131072},
            },
        ):
            with bb.dataflow():
                out = bb.emit(
                    relax.call_tir(
                        func, (ids, freq), relax.TensorType((1, n, dim), "float32")
                    )
                )
                out = bb.emit_output(out)
            bb.emit_func_output(out)
    return bb.get()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("capture", type=Path)
    p.add_argument("freq", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--tvm-source", type=Path, required=True)
    p.add_argument("--emsdk", type=Path, required=True)
    a = p.parse_args()
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
    export_tvmjs(executable, a.output / "positions.wasm", a.emsdk, a.tvm_source)
    (a.output / "positions.bin").write_bytes(np.arange(80, dtype=np.int32).tobytes())
    for kind in ("local", "global"):
        (a.output / (kind + ".bin")).write_bytes(
            (a.freq / (kind + ".bin")).read_bytes()
        )
    data = []
    for layer in (0, 4):
        data.extend(
            x.numpy().reshape(-1)
            for x in torch.load(a.capture / f"layer{layer}.pt", weights_only=True)[
                "position_embeddings"
            ]
        )
    np.concatenate(data).tofile(a.output / "expected.bin")
    print("Dynamic RoPE exported", flush=True)


if __name__ == "__main__":
    main()
