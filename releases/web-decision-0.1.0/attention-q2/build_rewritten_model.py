from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import subprocess
import time
import traceback
from pathlib import Path

import onnx
import numpy as np
import tvm
from tvm import relax, tirx
from tvm.contrib import tvmjs
from tvm.relax.frontend.onnx import from_onnx
from tvm.ir import prim


EXPERIMENT = Path(__file__).resolve().parent


def find_workspace_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pnpm-workspace.yaml").is_file() and (
            parent / "scripts/tvm_selective_params.py"
        ).is_file():
            return parent
    raise RuntimeError("Run this builder from a web-decision repository checkout.")


ROOT = find_workspace_root()
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(EXPERIMENT))

import tvm_selective_params  # noqa: E402,F401
import tvm_narrow_int64  # noqa: E402,F401
import tvm_q4_operators  # noqa: E402,F401
import tvm_dynamic_onnx  # noqa: E402,F401
from build_inputs import verify_pinned_file  # noqa: E402
from output_safety import assert_fresh_output_path  # noqa: E402
from relax_rewrite import rewrite_attention  # noqa: E402
from build_report import raise_for_failed_build, record_wasm_export_failure  # noqa: E402
from online_kernel import KEY_TILE, QUERY_TILE, REDUCE_LANES  # noqa: E402
from online_tir import attention_primfunc, online_cpu_reference  # noqa: E402
from prototype import full_attention_reference, kev_additive_bias  # noqa: E402


GRAPH = ROOT / "models/kev-0.6b-onnx/onnx/model_q4f16_direct_dynamic_u32.onnx"
GRAPH_EXTERNAL_DATA = GRAPH.with_name("model_q4f16.onnx_data")
EXPECTED_GRAPH_SHA256 = "6be92dfb51391d2cf025879f7cd1c686fd5cf0b22471a456d7333c11feca03fd"
EXPECTED_GRAPH_BYTES = 3152008
EXPECTED_EXTERNAL_DATA_SHA256 = "777335dbc052e2e02ec7591f6eefcfb29e2c28caf52075e492d226fc27e0824b"
EXPECTED_EXTERNAL_DATA_BYTES = 335357952
EXPECTED_WASM_SHA256 = "5e995762514782786cf0230f7f88bf3b0f1db3499f9c44cea9a691539adee4b0"
EXPECTED_WASM_BYTES = 6680260
EXPECTED_TVM_VERSION = "0.27.0.post1"
EXPECTED_EMSCRIPTEN_VERSION = "4.0.23"


def assert_reproducible_environment() -> str:
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("The q2-k128 build requires Python 3.12.")
    if os.environ.get("PYTHONHASHSEED") != "0" or os.environ.get("SOURCE_DATE_EPOCH") != "0":
        raise SystemExit("Set PYTHONHASHSEED=0 and SOURCE_DATE_EPOCH=0 before starting Python.")
    expected_tvm_home = ROOT / ".toolchains/apache-tvm-src"
    if Path(os.environ.get("TVM_HOME", "")).resolve() != expected_tvm_home.resolve():
        raise SystemExit(f"Set TVM_HOME to the pinned source checkout: {expected_tvm_home}")
    emcc = shutil.which("emcc")
    if emcc is None:
        raise SystemExit("Activate the pinned Emscripten 4.0.23 environment before building.")
    result = subprocess.run(
        [emcc, "--version"], check=True, capture_output=True, text=True
    )
    version = result.stdout.splitlines()[0]
    if EXPECTED_EMSCRIPTEN_VERSION not in version:
        raise SystemExit(
            f"Expected Emscripten {EXPECTED_EMSCRIPTEN_VERSION}, found {version}."
        )
    return version


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_shape_func(mod):
    """Order only the independent shape writes present after attention fusion."""
    global_var = mod.get_global_var("shape_func")
    function = mod[global_var]
    if not isinstance(function.body, tirx.SeqStmt):
        raise ValueError("Unexpected generated shape function body")
    stores = list(function.body.seq)
    if not stores or not all(
        isinstance(store, tirx.BufferStore)
        and len(store.indices) == 1
        and isinstance(store.indices[0], prim.IntImm)
        for store in stores
    ):
        raise ValueError("Unexpected generated shape function stores")
    buffer = stores[0].buffer
    indices = [int(store.indices[0]) for store in stores]
    if (
        len(function.params) != 1
        or not buffer.same_as(function.params[0])
        or sorted(indices) != list(range(1, len(stores) + 1))
        or any(not store.buffer.same_as(buffer) for store in stores)
    ):
        raise ValueError("Unexpected generated shape function buffer or indexes")

    def is_independent(expression):
        if isinstance(expression, prim.IntImm):
            return True
        if isinstance(expression, prim.Mul):
            return is_independent(expression.a) and is_independent(expression.b)
        if isinstance(expression, tvm.ir.expr.TensorLoad):
            return (
                expression.source.same_as(buffer)
                and len(expression.indices) == 1
                and isinstance(expression.indices[0], prim.IntImm)
                and int(expression.indices[0]) == 0
            )
        return False

    if not all(is_independent(store.value) for store in stores):
        raise ValueError("Generated shape writes are not independent")
    ordered = sorted(stores, key=lambda store: int(store.indices[0]))
    mod.update_func(global_var, function.with_body(tirx.SeqStmt(ordered)))
    return len(stores)


def run_boundary_cpu_differential() -> dict:
    sequence_length = KEY_TILE + 7
    input_ids = np.zeros(sequence_length, dtype="int32")
    input_ids[0] = 151659
    delimiter_positions = sorted({1, 8, 31, 32, KEY_TILE, KEY_TILE + 1})
    input_ids[delimiter_positions] = 151660
    attention_mask = np.ones(sequence_length, dtype="int32")
    attention_mask[-2] = 0
    bias = kev_additive_bias(input_ids, attention_mask)

    random = np.random.default_rng(8192 + sequence_length)
    query = random.normal(size=(1, 16, sequence_length, 128)).astype("float16")
    key = random.normal(size=(1, 16, 128, sequence_length)).astype("float16")
    value = random.normal(size=(1, 16, sequence_length, 128)).astype("float16")
    expected = full_attention_reference(query, key, bias, value)
    actual = online_cpu_reference(query, key, value, bias)
    difference = np.abs(actual.astype("float32") - expected.astype("float32"))
    return {
        "kind": "synthetic CPU differential; not model activations or GPU",
        "sequenceLength": sequence_length,
        "queryTile": QUERY_TILE,
        "keyTile": KEY_TILE,
        "delimiterPositions": delimiter_positions,
        "invalidKeyPosition": sequence_length - 2,
        "queryTailRows": sequence_length % 2,
        "keyTailRows": sequence_length % KEY_TILE,
        "maxAbsoluteError": float(difference.max()),
        "meanAbsoluteError": float(difference.mean()),
        "exactFloat16Elements": int(np.count_nonzero(difference == 0)),
        "totalElements": int(difference.size),
        "tolerance": 0.002,
        "withinTolerance": bool(difference.max() <= 0.002),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models/kev-attention-q2-k128-build",
        help="A new isolated output directory; existing output is never overwritten.",
    )
    args = parser.parse_args()
    try:
        output = assert_fresh_output_path(args.output)
    except (FileExistsError, NotADirectoryError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if KEY_TILE != 128:
        raise SystemExit("This release builder is pinned to KEY_TILE=128; set KEY_TILE=128.")
    if tvm.__version__ != EXPECTED_TVM_VERSION:
        raise SystemExit(
            f"Expected TVM {EXPECTED_TVM_VERSION}, found {tvm.__version__}."
        )
    emscripten_version = assert_reproducible_environment()
    try:
        graph_sha256 = verify_pinned_file(
            GRAPH, EXPECTED_GRAPH_BYTES, EXPECTED_GRAPH_SHA256, "ONNX input graph"
        )
        external_data_sha256 = verify_pinned_file(
            GRAPH_EXTERNAL_DATA,
            EXPECTED_EXTERNAL_DATA_BYTES,
            EXPECTED_EXTERNAL_DATA_SHA256,
            "ONNX external tensor data",
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    output.mkdir(parents=True, exist_ok=False)

    started = time.monotonic()
    report = {
        "kind": "kev-attention-query-tile-relax-prototype",
        "status": "running",
        "sourceGraph": str(GRAPH.relative_to(ROOT)),
        "sourceGraphSha256": graph_sha256,
        "sourceExternalData": str(GRAPH_EXTERNAL_DATA.relative_to(ROOT)),
        "sourceExternalDataSha256": external_data_sha256,
        "sourceExternalDataBytes": EXPECTED_EXTERNAL_DATA_BYTES,
        "tvmVersion": tvm.__version__,
        "emscriptenVersion": emscripten_version,
        "rewrittenAttentionCalls": 0,
        "stages": [],
        "gpuExecuted": False,
        "notes": [
            "Isolated compiler prototype only; it is not a model package or release asset.",
            "No browser or GPU inference is run by this script.",
            f"Each WebGPU workgroup processes two query rows for one head, with key tiles of {KEY_TILE}.",
            "The trailing barrier was removed because the next tile's first barrier synchronizes score reuse.",
            "The online recurrence changes fp32 softmax and value accumulation order versus full-matrix execution.",
            "The standalone CPU differential is synthetic and is not model-activation or GPU validation.",
        ],
    }

    def stage(name: str) -> None:
        report["stages"].append(
            {"name": name, "elapsedSeconds": time.monotonic() - started}
        )
        print(name, flush=True)

    try:
        stage("Importing the pinned dynamic ONNX graph")
        model = onnx.load(GRAPH, load_external_data=True)
        onnx.checker.check_model = lambda _: None
        mod = from_onnx(
            model,
            shape_dict=None,
            dtype_dict={"input_ids": "int32", "attention_mask": "int32"},
            keep_params_in_input=True,
        )
        mod, params = relax.frontend.detach_params(mod)
        report["parameterCount"] = len(params["main"])

        stage("Rewriting all exact QK-scale-bias-softmax-V chains before LegalizeOps")
        mod, count = rewrite_attention(mod)
        report["rewrittenAttentionCalls"] = count
        report["preLegalizeFunctionCount"] = len(mod.functions)

        stage("Applying the pinned WebGPU Relax lowering pipeline")
        target = tvm.target.Target(
            "webgpu", host={"kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"}
        )
        from tvm.relax.backend.gpu_generic.pipeline import (
            dataflow_lower_passes,
            finalize_passes,
            legalize_passes,
            library_dispatch_passes,
        )

        passes = (
            library_dispatch_passes(target)
            + legalize_passes(target)
            + dataflow_lower_passes(target)
            + finalize_passes(target)
        )
        with target:
            for pipeline_pass in passes:
                mod = pipeline_pass(mod)
        report["normalizedShapeWriteCount"] = normalize_shape_func(mod)
        report["loweredFunctionCount"] = len(mod.functions)
        report["loweringPipelinePasses"] = [
            pipeline_pass.info.name for pipeline_pass in passes
        ]
        lowered_ir = output / "lowered-ir.py"
        lowered_ir.write_text(mod.script())
        report["loweredIrSha256"] = sha256(lowered_ir)
        report["queryKernelPresentAfterLowering"] = (
            mod.get_global_var("kev_attention_query_tile") is not None
        )
        report["cpuBoundaryDifferential"] = run_boundary_cpu_differential()
        if not report["cpuBoundaryDifferential"]["withinTolerance"]:
            raise ValueError("Synthetic CPU boundary differential exceeded tolerance")

        stage("Compiling the rewritten Relax module for WebGPU and wasm32")
        executable = relax.build(mod, target=target, relax_pipeline=None)
        report["compileStatus"] = "success"
        report["status"] = "compiled"
        del params, model

        stage("Saving generated WGSL for the integrated attention PrimFunc")
        kernel_module = tvm.compile(
            tvm.IRModule({"kev_attention_query_tile": attention_primfunc()}),
            target="webgpu",
        )
        kernel_sources = []
        pending_modules = [kernel_module.mod]
        while pending_modules:
            runtime_module = pending_modules.pop()
            pending_modules.extend(runtime_module.imports)
            try:
                source = runtime_module.inspect_source("wgsl")
            except Exception:
                continue
            if "kev_attention_query_tile_kernel" in source:
                kernel_sources.append(source)
        if len(kernel_sources) != 1:
            raise RuntimeError(
                f"Expected one attention WGSL source, found {len(kernel_sources)}"
            )
        wgsl = output / "kev_attention_query_tile.wgsl"
        wgsl.write_text(kernel_sources[0])
        report["attentionWgslSha256"] = sha256(wgsl)
        report["attentionWgslBytes"] = wgsl.stat().st_size
        expected_partial = (
            f"var<workgroup> partial : array<f32, "
            f"{QUERY_TILE * KEY_TILE * REDUCE_LANES}>;"
        )
        expected_scores = (
            f"var<workgroup> scores : array<f32, {QUERY_TILE * KEY_TILE}>;"
        )
        if (
            expected_partial not in kernel_sources[0]
            or expected_scores not in kernel_sources[0]
        ):
            raise RuntimeError("Generated WGSL does not match the selected attention scratch dimensions")
        barrier_count = kernel_sources[0].count("workgroupBarrier();")
        if barrier_count != 2:
            raise RuntimeError(f"Expected two WGSL barriers per key tile, found {barrier_count}")
        report["attentionQueryTile"] = QUERY_TILE
        report["attentionKeyTile"] = KEY_TILE
        report["attentionReductionLanes"] = REDUCE_LANES
        report["attentionWorkgroupSize"] = [128, QUERY_TILE, 1]
        report["attentionWorkgroupScratchBytes"] = (
            QUERY_TILE * (KEY_TILE * REDUCE_LANES + KEY_TILE) * 4
        )
        report["attentionWorkgroupsAt8192"] = 65536
        report["attentionKeyTilesPerQueryAt8192"] = 8192 // KEY_TILE
        report["attentionWorkgroupBarriersPerKeyTile"] = barrier_count
        report["estimatedBarrierInstancesAt8192Across28Layers"] = (
            report["attentionWorkgroupsAt8192"]
            * report["attentionKeyTilesPerQueryAt8192"]
            * barrier_count
            * 28
        )
        report["performanceValidation"] = "not run; compile evidence only"

        try:
            wasm = output / "model.wasm"
            executable.export_library(str(wasm), fcompile=tvmjs.create_tvmjs_wasm)
            report["wasmSha256"] = sha256(wasm)
            report["wasmBytes"] = wasm.stat().st_size
            report["wasmExportStatus"] = "success"
            report["status"] = "success"
        except Exception as error:
            record_wasm_export_failure(report, error)
        if report.get("wasmExportStatus") == "success" and (
            report["wasmSha256"] != EXPECTED_WASM_SHA256
            or report["wasmBytes"] != EXPECTED_WASM_BYTES
        ):
            report.update(
                status="failed",
                errorType="ValueError",
                error="The q2-k128 WASM does not match the release hash and size pins.",
            )
    except Exception as error:
        report.update(
            status="failed",
            errorType=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )

    report["elapsedSeconds"] = time.monotonic() - started
    (output / "build-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    raise_for_failed_build(report)


if __name__ == "__main__":
    main()
