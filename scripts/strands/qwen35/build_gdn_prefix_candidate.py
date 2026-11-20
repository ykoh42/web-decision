"""Build the prefix-aware GDN decoder candidate from the pinned R19 TVMScript."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import textwrap
from pathlib import Path
from typing import TypeVar

from qwen35_tvm_build import (
    ROOT,
    SOURCE_GRAPH_DIGESTS,
    export_tvmjs,
    load_compiler_input,
    verify_emsdk_checkout,
    verify_tvm_checkout,
)

CHUNKS = "(s22 + ((64 - s_prefix % 64) % 64) + 63) // 64"
MAX_CHUNKS = 65
PACKERS = ("qwen35_gdn_chunkify_beta", "qwen35_gdn_chunkify_qkv_decay")
UNPACK = "qwen35_gdn_unpack"


def replace_checked(source: str, old: str, new: str, count: int = 1) -> str:
    if source.count(old) != count:
        raise ValueError(f"Unexpected R19 GDN occurrence count for {old!r}")
    return source.replace(old, new)


def prefix_kernel(source: str, name: str) -> str:
    entry = source.index(f"    def {name}(")
    end = source.index("    @T.prim_func", entry)
    body = source[entry:end]
    if name == UNPACK:
        body = replace_checked(body, f"def {name}(", f"def {name}(prefix: T.int32, ")
        body = replace_checked(
            body,
            "chunk_index: T.int32 = position // 64",
            "padded = T.if_then_else(position < prefix, position, position + ((64 - prefix % 64) % 64))\n                        chunk_index: T.int32 = padded // 64",
        )
        body = replace_checked(
            body, "row: T.int32 = position % 64", "row: T.int32 = padded % 64"
        )
    else:
        dim = "'(sequence_dim + T.int64(64) - T.int64(1)) // T.int64(64)'"
        body = replace_checked(
            body, f"def {name}(", f"def {name}(prefix: T.int32, chunk_count: T.int32, "
        )
        body = replace_checked(body, dim, "chunk_count", 1 if name == PACKERS[0] else 4)
        body = replace_checked(
            body,
            "T.thread_binding((sequence_dim + T.int64(63)) // T.int64(64),",
            "T.thread_binding(chunk_count,",
        )
        body = replace_checked(
            body,
            'position: T.int32 = T.Cast("int32", chunk_index * T.int64(64) + T.Cast("int64", row))',
            "padded = chunk_index * 64 + row\n                        gap = (64 - prefix % 64) % 64\n                        position = T.if_then_else(padded < prefix, padded, padded - gap)",
        )
        body = replace_checked(
            body,
            "if position <",
            "if (padded < prefix or padded >= prefix + gap) and position <",
        )
    return source[:entry] + body + source[end:]


def candidate_script(source: str, cumsum: str) -> str:
    """Retain the pinned packing arithmetic and change only the logical GDN rows."""
    source = replace_checked(
        source,
        's86 = TypeVar("s86")',
        's86 = TypeVar("s86")\ns_prefix = TypeVar("s_prefix")',
    )
    slot = 'n_slots: R.Tensor((s0,), dtype="int32"),'
    source = replace_checked(
        source, slot, slot + " source_prefix: R.Shape([s_prefix]),"
    )
    source = replace_checked(source, '"num_input": 6', '"num_input": 7')
    source = replace_checked(source, '"s86": 2}', '"s86": 2, "s_prefix": 0}')
    source = replace_checked(source, '"s86": 255}', '"s86": 255, "s_prefix": 4096}')
    source = replace_checked(source, "(s22 + 64 - 1) // 64", CHUNKS, 1008)
    for name in (*PACKERS, UNPACK):
        source = prefix_kernel(source, name)
        call = f"R.call_tir(cls.{name}, ("
        args = 'T.Cast("int32", s_prefix), '
        if name in PACKERS:
            args += f'T.Cast("int32", {CHUNKS}), '
        source = replace_checked(source, call, call + args, 18)
    # Sixty-four signed log decays produce one FP32 prefix sum for each row.
    shape = f'R.Tensor((s0, 16, {CHUNKS}, 64), dtype="float32")'
    pattern = rf"(?P<lhs>lv\w+): {re.escape(shape)} = R\.cumsum\((?P<input>lv\w+), axis=3, dtype=None, exclusive=False\)"
    call = rf'\g<lhs> = R.call_tir(cls.qwen35_gdn_cumsum64, (T.Cast("int32", s0), T.Cast("int32", {CHUNKS}), \g<input>), out_ty={shape})'
    source, count = re.subn(pattern, call, source)
    if count != 36:
        raise ValueError(
            "Expected two cumulative-decay uses in each of the 18 GDN layers"
        )
    source = replace_checked(
        source,
        "    @R.function\n",
        textwrap.indent(cumsum, "    ") + "\n\n    @R.function\n",
    )
    # Checking the original shape against its clamped size uses WASM VM match_shape.
    return replace_checked(
        source,
        "        with R.dataflow():",
        "        checked_prefix = R.match_cast(source_prefix, R.Shape([T.max(T.min(s_prefix, s22), T.int64(0))]))\n        with R.dataflow():",
    )


def fp32_activations(source: str) -> str:
    """Use each FP32 activation before R19 rounds it to BF16, keeping packed weights."""
    for value in ("conv", "silu", "normalized", "weighted", "product"):
        rounded = f'T.reinterpret("float32", {value}_bits + T.uint32(32767) + {value}_lsb & T.uint32(4294901760))'
        source = replace_checked(source, rounded, value, 2)
    for target, rounded, value in (
        (
            "output[row, output_feature]",
            'T.reinterpret("float32", rounded_bits)',
            "result",
        ),
        (
            "output[batch_index, position, head_index, value_index]",
            'T.reinterpret("float32", rounded)',
            "value",
        ),
        ("output[batch, query_index, head, dim]", "rounded[0]", "normalized"),
        ("y[i]", 'T.reinterpret("float32", rounded)', "x[i]"),
    ):
        source = replace_checked(source, f"{target} = {rounded}", f"{target} = {value}")
    call = "R.call_tir(cls.qwen35_text_bf16_rne_fp32,"
    pattern = (
        rf"^(            \w+ = ){re.escape(call)} \((\w+),\), out_ty=R\.Tensor\(.*\)\)$"
    )
    source, count = re.subn(pattern, r"\1\2", source, flags=re.MULTILINE)
    if count != 628 or call in source:
        raise ValueError(f"Expected 628 R19 BF16 activation calls, replaced {count}")
    return source


def bind_chunk_count(source: str) -> str:
    """Give TE one bound dimension instead of two unordered shape parameters."""
    source = source.replace(CHUNKS, "s_chunks")
    source = replace_checked(source, '"s_prefix": 0}', '"s_prefix": 0, "s_chunks": 1}')
    source = replace_checked(
        source, '"s_prefix": 4096}', f'"s_prefix": 4096, "s_chunks": {MAX_CHUNKS}}}'
    )
    return replace_checked(
        source,
        "        with R.dataflow():",
        "        s_chunks = T.int64()\n"
        f"        chunks_value = R.shape([{CHUNKS}])\n"
        "        chunks_shape = R.match_cast(chunks_value, R.Shape([s_chunks]))\n"
        "        with R.dataflow():",
    )


def candidate_module(precision: str = "bf16"):
    import tvm
    from qwen35_gdn_cumulative_decay import qwen35_gdn_cumsum64
    from tvm import relax
    from tvm.script import ir as I
    from tvm.script import relax as R
    from tvm.script import tirx as T
    from tvm.tirx.layout import Axis

    if precision not in ("bf16", "fp32"):
        raise ValueError(f"Unsupported Qwen decoder activation precision {precision!r}")
    source, metadata = load_compiler_input("text", tvm, relax)
    program = candidate_script(source, qwen35_gdn_cumsum64.script())
    if precision == "fp32":
        program = bind_chunk_count(fp32_activations(program))
    mod = tvm.script.from_source(
        program,
        extra_vars={
            "TypeVar": TypeVar,
            "I": I,
            "R": R,
            "T": T,
            "Axis": Axis,
            "tvm": tvm,
            "relax": relax,
            "metadata": metadata,
        },
        check_well_formed=True,
    )
    return mod, hashlib.sha256(program.encode()).hexdigest()


def build(output: Path, precision: str = "bf16") -> None:
    import tvm
    from qwen35_vm_shape_order import order_vm_shape_stores
    from tvm import relax

    tvm_home, emsdk = ROOT / ".toolchains/apache-tvm-src", ROOT / ".toolchains/emsdk"
    verify_tvm_checkout(tvm_home)
    verify_emsdk_checkout(emsdk)
    if tvm.__version__ != "0.27.0.post1":
        raise ValueError("The pinned TVM version is required")
    mod, program_digest = candidate_module(precision)
    target = tvm.target.Target(
        "webgpu", host={"kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"}
    )
    request_arguments = [param.name for param in mod["main"].params]
    if precision == "fp32":
        mod = relax.get_default_pipeline(target)(mod)
        order_vm_shape_stores(mod)
        executable = relax.build(mod, target=target, relax_pipeline=None)
    else:
        executable = relax.build(mod, target=target)
    output.mkdir(parents=True, exist_ok=True)
    name = (
        "qwen35-gdn-prefix-candidate"
        if precision == "bf16"
        else "qwen35-gdn-prefix-fp32-candidate"
    )
    wasm = output / f"{name}.wasm"
    export_tvmjs(executable, wasm, emsdk, tvm_home)
    manifest = {
        "sourceGraphSha256": SOURCE_GRAPH_DIGESTS["text"],
        "candidateScriptSha256": program_digest,
        "tvm": tvm.__version__,
        "wasmSha256": hashlib.sha256(wasm.read_bytes()).hexdigest(),
        "mainArguments": len(request_arguments),
        "requestArguments": request_arguments[:7],
        "maxSequence": 4096,
        "maxChunks": MAX_CHUNKS,
    }
    if precision == "fp32":
        manifest["activationPrecision"] = precision
    wasm.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()
    build(args.output_dir, args.precision)
