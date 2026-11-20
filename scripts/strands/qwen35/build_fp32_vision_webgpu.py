"""Build the CPU VisionEngine FP32 precision variant of the original Qwen vision graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import TypeVar

from qwen35_tvm_build import (
    ROOT,
    SOURCE_GRAPH_DIGESTS,
    export_tvmjs,
    inspect_webgpu_source,
    load_compiler_input,
    pinned,
    verify_emsdk_checkout,
    verify_tvm_checkout,
)

ROUND_KERNEL = "round_bf16_webgpu"
ROUND_TIR_SHA256 = "de5a69ef4d8bb3fb615e92fd50c6dbcd3ddeee01101aa8cfd8971a35d38ccc1c"
VM_SHA256 = "be2f5f6209bc1c79b8fd2f1eba540fe0ab21c1afd60460de893e277f54d86a5f"
SHADER_INVENTORY_SHA256 = (
    "c1c8827d4e633d292898210f2667b2b780fd97774d97bed3323d650119fd0537"
)
WASM_SHA256 = "4fa14d07c05c881ef3134fbdf16925e856401fb9608fcee2054d2b6d7aa10ec0"


def shader_inventory_sha256(wgsl: str) -> str:
    # TVM may print independent kernels in a different order on a fresh process.
    sections = re.split(r"(?m)^// Function: (\w+)\n(?=//-+\n// Function: )", wgsl)
    entries = [
        f"{name} {hashlib.sha256(shader.rstrip().encode()).hexdigest()}"
        for name, shader in zip(sections[1::2], sections[2::2], strict=True)
    ]
    return hashlib.sha256("\n".join(sorted(entries)).encode()).hexdigest()


def fp32_boundary(function):
    """Keep the scheduled grid, bounds guard and ABI, replacing only BF16 rounding."""
    from tvm.script import from_source
    from tvm.script import tirx as T

    script = function.script()
    if hashlib.sha256(script.encode()).hexdigest() != ROUND_TIR_SHA256:
        raise ValueError("The scheduled Qwen vision BF16 round differs from its pin.")
    prefix, _, _ = script.partition(
        'bits: T.uint32 = T.reinterpret("uint32", input_buffer[value_index])'
    )
    return from_source(
        prefix + "output_buffer[value_index] = input_buffer[value_index]",
        extra_vars={"T": T},
        check_well_formed=True,
    )


def scheduled_vision(target):
    import tvm
    from tvm import relax
    from tvm.script import ir as I
    from tvm.script import relax as R
    from tvm.script import tirx as T
    from tvm.tirx.layout import Axis

    source, metadata = load_compiler_input("vision", tvm, relax)
    module = tvm.script.from_source(
        source,
        extra_vars={
            "T": T,
            "I": I,
            "R": R,
            "TypeVar": TypeVar,
            "Axis": Axis,
            "metadata": metadata,
        },
        check_well_formed=True,
    )
    with target:
        return relax.get_default_pipeline(target)(module)


def build_report(
    staged: Path,
    destination: Path,
    version: str,
    commit: str,
    wgsl_sha: str,
    shape_slots: list[int],
) -> dict:
    wasm_pin = {"bytes": len(pinned(staged, WASM_SHA256)), "sha256": WASM_SHA256}
    return {
        "format": "qwen35-fp32-vision-webgpu-build-v1",
        "activationPrecision": "float32",
        "sourceRelaxSha256": SOURCE_GRAPH_DIGESTS["vision"],
        "toolchain": {"apache-tvm": version},
        "tvmCommit": commit,
        "execution": {"onnxUsed": False},
        "tvmWebgpuBuild": {"status": "passed", "deviceRuntimeExecuted": False},
        "tvmJsArtifact": {"status": "passed", "wasm": wasm_pin},
        "replacedKernel": ROUND_KERNEL,
        "hostShapeStoreOrder": shape_slots,
        "vmSha256": VM_SHA256,
        "shaderInventorySha256": SHADER_INVENTORY_SHA256,
        "wgslSha256": wgsl_sha,
        "wasm": str(destination.resolve()),
        "wasmBytes": wasm_pin["bytes"],
        "wasmSha256": wasm_pin["sha256"],
    }


def build(output: Path) -> None:
    import tvm
    from qwen35_vm_shape_order import order_vm_shape_stores
    from tvm import relax

    tvm_home = ROOT / ".toolchains/apache-tvm-src"
    emsdk = ROOT / ".toolchains/emsdk"
    tvm_commit = verify_tvm_checkout(tvm_home)
    verify_emsdk_checkout(emsdk)
    if tvm.__version__ != "0.27.0.post1":
        raise ValueError("The pinned TVM version is required.")
    target = tvm.target.Target(
        "webgpu", host={"kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"}
    )
    scheduled = scheduled_vision(target)
    key = scheduled.get_global_var(ROUND_KERNEL)
    scheduled.update_func(key, fp32_boundary(scheduled[key]))
    # VMShapeLower emits independent host heap writes in an unspecified order.
    order_vm_shape_stores(scheduled)
    executable = relax.build(scheduled, target=target, relax_pipeline=None)
    vm_sha = hashlib.sha256(str(executable.as_text()).encode()).hexdigest()
    if vm_sha != VM_SHA256:
        raise ValueError("The FP32 precision variant changed the original vision VM.")
    wgsl = inspect_webgpu_source(executable.mod)
    if shader_inventory_sha256(wgsl) != SHADER_INVENTORY_SHA256:
        raise ValueError(
            "The compiled vision shaders differ from the verified FP32 graph."
        )
    output.mkdir(parents=True, exist_ok=True)
    wasm = output / "qwen35-dynamic-vision.webgpu.wasm"
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        staged = Path(temporary) / wasm.name
        export_tvmjs(executable, staged, emsdk, tvm_home)
        slots = [int(store.indices[0]) for store in scheduled["shape_func"].body.seq]
        report = build_report(
            staged,
            wasm,
            tvm.__version__,
            tvm_commit,
            hashlib.sha256(wgsl.encode()).hexdigest(),
            slots,
        )
        metadata = Path(temporary) / "build-report.json"
        metadata.write_text(json.dumps(report, indent=2) + "\n")
        staged.replace(wasm)
        metadata.replace(output / metadata.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    build(parser.parse_args().output_dir)
