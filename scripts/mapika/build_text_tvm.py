"""Build Mapika 2B v12's BF16 decoder and letter readout for TVM WebGPU."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import TypeVar

ROOT = Path(__file__).resolve().parents[2]
QWEN_BUILD_DIR = ROOT / "scripts/strands/qwen35"
ASSETS = ROOT / "packages/web-decision/assets"
MODEL_REVISION = "7614a6c7ea5a047aa7d7bcfb1a0015469381c3e3"
QWEN_BINDINGS_SHA256 = (
    "03604675d3bdbf6f8a7cef7dcc372c0626a0f4d0b9964ec06b200dc5918d5526"
)
MAPIKA_INVENTORY_SHA256 = (
    "151295ce4b89e7dc049e1fb7420fda52670772cf391c27fc741e90182f617a60"
)
MAX_POSITION_EMBEDDINGS = 262144
LETTER_COUNT = 255
PACKED_LETTER_ROWS = 256


def assignment_name(statement: ast.stmt) -> str:
    if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
        return statement.target.id
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target = statement.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    raise ValueError("Expected one named binding in the pinned Qwen decoder")


def remove_adapter_addition(value: ast.expr, removed: set[str]) -> str | None:
    if not isinstance(value, ast.Call) or ast.unparse(value.func) != "R.add":
        return None
    base, update = value.args
    if not isinstance(base, ast.Name) or not isinstance(update, ast.Name):
        raise ValueError("Expected a named base and LoRA update in the Qwen decoder")
    if base.id in removed or update.id not in removed:
        raise ValueError(
            "Expected the LoRA update on the right side of its base addition"
        )
    return base.id


def base_bindings(
    flow: ast.With, lines: list[str], removed: set[str], prefix_cast: str
) -> list[str]:
    body = []
    additions = 0
    for statement in flow.body:
        name = assignment_name(statement)
        replacement, rewrote_addition = backbone_binding(
            statement, name, removed, lines, prefix_cast
        )
        body.extend(replacement)
        additions += rewrote_addition
        if name == "lv5575":
            if additions != 186:
                raise ValueError(
                    f"Expected 186 Strands LoRA additions, found {additions}"
                )
            return body
    raise ValueError(
        "The pinned Qwen decoder no longer exposes its final normalization"
    )


def backbone_binding(statement, name, removed, lines, prefix_cast):
    value = statement.value
    inputs = {node.id for node in ast.walk(value) if isinstance(node, ast.Name)}
    if inputs.isdisjoint(removed):
        binding = lines[statement.lineno - 1 : statement.end_lineno]
        return [line.replace(prefix_cast, "T.int32(0)") for line in binding], False
    base = remove_adapter_addition(value, removed)
    if base is not None:
        return [f"            {name} = {base}"], True
    removed.add(name)
    return [], False


def readout_script() -> list[str]:
    return [
        '            hidden: R.Tensor((s_slots, 2048), dtype="float32") = R.index_tensor(lv5575, (slot_batch, slot_indices))',
        f'            padded = R.call_tir(cls.qwen35_full_text_packed_linear, (T.Cast("int32", s_slots), T.int32(2048), T.int32({PACKED_LETTER_ROWS}), hidden, letter_weights), out_ty=R.Tensor((s_slots, {PACKED_LETTER_ROWS}), dtype="float32"))',
        f'            logits: R.Tensor((s_slots, {LETTER_COUNT}), dtype="float32") = R.strided_slice(padded, (1,), (0,), ({LETTER_COUNT},), (1,), assume_inbound=True)',
        "            R.output(logits)",
        "        return logits",
    ]


def find_main(source):
    graph = next(
        node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)
    )
    return next(
        node
        for node in graph.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )


def mapika_text_script(source: str) -> tuple[str, list[str]]:
    """Retain the checked Qwen backbone and replace the Strands-specific graph tail."""
    lines = source.splitlines()
    entry = find_main(source)
    flow = next(node for node in entry.body if isinstance(node, ast.With))
    inputs = {
        param.arg: lines[param.lineno - 1][param.col_offset : param.end_col_offset]
        for param in entry.args.args
    }
    base = [
        name for name in inputs if name.startswith("p_torso_") and "_lora_" not in name
    ]
    if len(base) != 319:
        raise ValueError("The pinned Qwen backbone must have 319 external parameters")
    params = [
        inputs[name] for name in ("inputs_embeds", "attention_mask", "position_ids")
    ]
    params.extend(
        [
            'slot_batch: R.Tensor((s_slots,), dtype="int32")',
            'slot_indices: R.Tensor((s_slots,), dtype="int32")',
            f'letter_weights: R.Tensor(({PACKED_LETTER_ROWS}, 1024), dtype="int32")',
        ]
    )
    params.extend(inputs[name] for name in base)
    removed = {name for name in inputs if "_lora_" in name}
    definition = [
        f'    def main({", ".join(params)}) -> R.Tensor((s_slots, {LETTER_COUNT}), dtype="float32"):',
        # StaticPlanBlockMemory allocates for the upper bound even on short rows.
        '        R.func_attr({"num_input": 5, "params": [], "tir_var_lower_bound": {"s0": 1, "s22": 1, "s_slots": 1, "s_chunks": 1}})',
        "        cls = Module",
        "        s_chunks = T.int64()",
        "        chunks_shape = R.match_cast(R.shape([(s22 + 63) // 64]), R.Shape([s_chunks]))",
        "        with R.dataflow():",
    ]
    decoder = base_bindings(flow, lines, removed, 'T.Cast("int32", s_prefix)')
    header = lines[: entry.lineno - 1]
    header.insert(0, 's_slots = TypeVar("s_slots")')
    return "\n".join([*header, *definition, *decoder, *readout_script()]) + "\n", base


def graph_bindings(parameters, qwen, inventory):
    from math import prod

    base = [entry for entry in qwen["bindings"] if entry["asset"] == "base"]
    if [entry["relaxArgumentName"] for entry in base] != parameters:
        raise ValueError("Mapika decoder parameter order differs from the graph")
    native = {
        tensor["name"]: tensor for tensor in inventory["weights"]["text"]["tensors"]
    }
    bindings = []
    for entry in base:
        name, shape = entry["sourceName"], entry["relaxShape"]
        tensor = native[name]
        if (
            entry["relaxDtype"] != "int32"
            or tensor["dtype"] != "BF16"
            or tensor["shape"] != entry["shape"]
            or tensor["length"] != prod(shape) * 4
        ):
            raise ValueError(
                f"Mapika graph weight differs from the native tensor {name}"
            )
        bindings.append({"name": name, "shape": shape})
    return bindings


def build(output: Path) -> None:
    sys.path.insert(0, str(QWEN_BUILD_DIR))
    import tvm
    from build_gdn_prefix_candidate import bind_chunk_count, candidate_script
    from qwen35_gdn_cumulative_decay import qwen35_gdn_cumsum64
    from qwen35_tvm_build import (
        export_tvmjs,
        load_compiler_input,
        pinned,
        verify_emsdk_checkout,
        verify_tvm_checkout,
    )
    from qwen35_vm_shape_order import order_vm_shape_stores
    from tvm import relax
    from tvm.script import ir as I
    from tvm.script import relax as R
    from tvm.script import tirx as T
    from tvm.tirx.layout import Axis

    tvm_root = ROOT / ".toolchains/apache-tvm-src"
    emsdk = ROOT / ".toolchains/emsdk"
    tvm_revision = verify_tvm_checkout(tvm_root)
    emsdk_revision = verify_emsdk_checkout(emsdk)
    if tvm.__version__ != "0.27.0.post1":
        raise ValueError("Use the pinned TVM Python compiler")
    source, metadata = load_compiler_input("text", tvm, relax)
    reusable = bind_chunk_count(candidate_script(source, qwen35_gdn_cumsum64.script()))
    program, parameters = mapika_text_script(reusable)
    bindings = graph_bindings(
        parameters,
        json.loads(
            pinned(ASSETS / "qwen35/r19/weight-bindings.json", QWEN_BINDINGS_SHA256)
        ),
        json.loads(
            pinned(ASSETS / "mapika2b/source-inventory.json", MAPIKA_INVENTORY_SHA256)
        ),
    )
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
    target = tvm.target.Target(
        "webgpu", host={"kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"}
    )
    lowered = relax.get_default_pipeline(target)(mod)
    order_vm_shape_stores(lowered)
    executable = relax.build(lowered, target=target, relax_pipeline=None)
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "text.wasm"
    export_tvmjs(executable, artifact, emsdk, tvm_root)
    report = {
        "modelRevision": MODEL_REVISION,
        "tvmRevision": tvm_revision,
        "emsdkRevision": emsdk_revision,
        "textScriptSha256": hashlib.sha256(program.encode()).hexdigest(),
        "wasmBytes": artifact.stat().st_size,
        "wasmSha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "inputs": [param.name for param in mod["main"].params[:5]],
        "externalParameters": ["letter_weights", *parameters],
        "activationPrecision": "bfloat16",
        "letterCount": LETTER_COUNT,
        "maxPositionEmbeddings": MAX_POSITION_EMBEDDINGS,
    }
    (output / "text-build.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = {"wasmSha256": report["wasmSha256"], "bindings": bindings}
    (output / "text-bindings.json").write_text(
        json.dumps(manifest, separators=(",", ":")) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    build(parser.parse_args().output_dir)
