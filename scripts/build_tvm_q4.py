#!/usr/bin/env python3
"""Compile Kev's actual packed q4 weights with TVM WebGPU custom operators."""

import argparse
import hashlib
import importlib
import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import onnx
import tvm
from safe_output import (
    atomic_write,
    create_temporary_directory,
    open_directory,
    promote_directory,
    promote_file,
    remove_tree,
    require_absent,
)
from tvm import relax
from tvm.contrib import tvmjs
from tvm.relax.frontend.onnx import from_onnx

CUSTOM_OPERATOR_MODULES = (
    "tvm_selective_params",
    "tvm_narrow_int64",
    "tvm_q4_operators",
)
for module_name in CUSTOM_OPERATOR_MODULES:
    importlib.import_module(module_name)

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BuildWorkspace:
    root: Path
    root_fd: int
    models_fd: int
    public_fd: int
    reports_fd: int
    stage_name: str
    stage_fd: int
    staged_wasm: Path
    staged_cache: Path
    wasm_output: Path
    cache_output: Path
    wasm_name: str
    cache_name: str


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=69)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument(
        "--diagnostic-dir",
        type=Path,
        help="New directory for intermediate IR snapshots",
    )
    args = parser.parse_args(argv)
    if args.diagnostic_dir:
        args.diagnostic_dir.mkdir(parents=True, exist_ok=False)
    if args.dynamic:
        importlib.import_module("tvm_dynamic_onnx")
    if not 1 <= args.seq_len <= 1024:
        raise SystemExit("seq-len must be in 1..1024 for this prototype")
    return args


def build_suffix(args):
    return "dynamic" if args.dynamic else f"seq{args.seq_len}"


def model_path(root, suffix):
    return root / "models/kev-0.6b-onnx/onnx" / f"model_q4f16_direct_{suffix}_u32.onnx"


def open_workspace(root, suffix):
    wasm_name = f"tvm-kev-q4-{suffix}.wasm"
    cache_name = f"tvm-kev-q4-{suffix}"
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    models_fd = None
    public_fd = None
    reports_fd = None
    stage_fd = None
    try:
        models_fd = open_directory(root, "models")
        public_fd = open_directory(root, "public")
        reports_fd = open_directory(root, "reports")
        require_absent(public_fd, wasm_name)
        require_absent(models_fd, cache_name)
        stage_name, stage_fd = create_temporary_directory(root_fd, ".tvm-q4-stage")
    except BaseException:
        for fd in (stage_fd, reports_fd, public_fd, models_fd, root_fd):
            if fd is not None:
                os.close(fd)
        raise

    # TVM needs path strings; final publication stays anchored to open directory FDs.
    stage_root = root / stage_name
    return BuildWorkspace(
        root=root,
        root_fd=root_fd,
        models_fd=models_fd,
        public_fd=public_fd,
        reports_fd=reports_fd,
        stage_name=stage_name,
        stage_fd=stage_fd,
        staged_wasm=stage_root / wasm_name,
        staged_cache=stage_root / cache_name,
        wasm_output=root / "public" / wasm_name,
        cache_output=root / "models" / cache_name,
        wasm_name=wasm_name,
        cache_name=cache_name,
    )


def close_workspace(workspace):
    os.close(workspace.stage_fd)
    try:
        remove_tree(workspace.root_fd, workspace.stage_name)
    finally:
        os.close(workspace.root_fd)
        os.close(workspace.models_fd)
        os.close(workspace.public_fd)
        os.close(workspace.reports_fd)


def mark_stage(report, started, name):
    report["stages"].append(
        {"name": name, "elapsedSeconds": time.perf_counter() - started}
    )
    print(name, flush=True)


def build_report(args, path):
    report = {
        "kind": "tvm-kev-q4-webgpu-build",
        "tvmVersion": tvm.__version__,
        "model": str(path.relative_to(ROOT)),
        "inputShape": [1, "sequence_length" if args.dynamic else args.seq_len],
        "weightPrecision": "original q4 packed bytes, f16 scales",
        "accumulation": "float32",
        "target": "webgpu / wasm32",
        "stages": [],
        "notes": [
            "Only a uint32 view of the original bytes is used; no global FP16 weight expansion.",
            "Shape and int32/boolean handling are restricted to this prototype.",
        ],
    }
    if args.dynamic:
        report.update(
            evidenceSchema="web-decision-tvm-build-v1",
            modelId="kev-0.6b-q4f16",
            artifact="model.wasm",
        )
    return report


def load_and_import_model(args, path, workspace, report, started):
    mark_stage(report, started, "Loading packed ONNX graph")
    model = onnx.load(path, load_external_data=True)
    onnx.checker.check_model = lambda _: None
    shape_dict = (
        None
        if args.dynamic
        else {"input_ids": [1, args.seq_len], "attention_mask": [1, args.seq_len]}
    )
    mark_stage(report, started, "Importing q4 operators into Relax")
    mod = from_onnx(
        model,
        shape_dict=shape_dict,
        dtype_dict={"input_ids": "int32", "attention_mask": "int32"},
        keep_params_in_input=True,
    )
    del model
    mod, params = relax.frontend.detach_params(mod)
    weights = params["main"]
    report["numberOfParameters"] = len(weights)
    report["parameterBytes"] = sum(int(weight.numpy().nbytes) for weight in weights)
    if args.diagnostic_dir:
        (args.diagnostic_dir / "import-ir.py").write_text(mod.script())
    if args.dynamic:
        text = mod.script()
        atomic_write(
            workspace.reports_fd,
            "tvm-q4-dynamic-import-ir.py",
            lambda output: output.write(text.encode()),
        )
    return mod, weights


def create_target():
    return tvm.target.Target(
        "webgpu", host={"kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"}
    )


def lower_model(args, mod, target, workspace, report, started):
    mark_stage(report, started, "Scheduling packed WebGPU kernels")
    if args.dynamic:
        from tvm.relax.backend.gpu_generic.pipeline import (
            dataflow_lower_passes,
            finalize_passes,
            legalize_passes,
            library_dispatch_passes,
        )

        lowered = mod
        with target:
            passes = (
                library_dispatch_passes(target)
                + legalize_passes(target)
                + dataflow_lower_passes(target)
                + finalize_passes(target)
            )
            for pipeline_pass in passes:
                print("Dynamic pipeline:", pipeline_pass.info.name, flush=True)
                lowered = pipeline_pass(lowered)
    else:
        with target:
            lowered = relax.get_default_pipeline(target)(mod)

    if args.dynamic:
        from tvm_kev_shape_order import normalize

        lowered = normalize(lowered)
        report["shapeStoreOrdering"] = (
            "canonical, with model-specific independence checks"
        )
    if args.diagnostic_dir:
        (args.diagnostic_dir / "lowered-ir.py").write_text(lowered.script())
    return lowered


def collect_q4_kernel_checks(lowered):
    checks = []
    for global_var, function in lowered.functions.items():
        if "packed_q4_matmul" in global_var.name_hint:
            script = function.script()
            checks.append(
                {
                    "name": global_var.name_hint,
                    "usesSharedMemory": 'scope="shared"' in script,
                    "hasPackedUint32": '"uint32"' in script,
                    "hasGlobalDecodedBuffer": "q4_decoded" in script
                    and 'scope="global"' in script,
                }
            )
    return checks


def validate_q4_kernels(lowered, report):
    checks = collect_q4_kernel_checks(lowered)
    report["q4KernelChecks"] = checks
    if len(checks) < 5 or any(
        not check["hasPackedUint32"] or check["hasGlobalDecodedBuffer"]
        for check in checks
    ):
        raise ValueError("Packed kernel scheduling checks failed")


def compile_model(lowered, target, report, started):
    mark_stage(report, started, "Compiling TVM WebGPU executable")
    return relax.build(lowered, target=target, relax_pipeline=None)


def export_wasm(executable, workspace, report, started):
    mark_stage(report, started, "Linking WebAssembly runtime")
    executable.export_library(
        str(workspace.staged_wasm), fcompile=tvmjs.create_tvmjs_wasm
    )


def write_tensor_cache(args, weights, workspace, report, started):
    mark_stage(report, started, "Writing packed tensor cache")
    tvmjs.dump_tensor_cache(
        ((f"param_{index}", value) for index, value in enumerate(weights)),
        str(workspace.staged_cache),
        encode_format="raw",
        show_progress=False,
        meta_data={
            "numParams": len(weights),
            "source": "original q4f16 checkpoint",
            "shape": [1, "sequence_length" if args.dynamic else args.seq_len],
        },
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def promote_outputs(workspace, report):
    report.update(
        status="success",
        wasm=str(workspace.wasm_output.relative_to(workspace.root)),
        wasmBytes=workspace.staged_wasm.stat().st_size,
        wasmSha256=sha256_file(workspace.staged_wasm),
        embeddedBuildRoot=str(workspace.root).encode()
        in workspace.staged_wasm.read_bytes(),
        tensorCache=str(workspace.cache_output.relative_to(workspace.root)),
        tensorCacheManifestSha256=sha256_file(
            workspace.staged_cache / "tensor-cache.json"
        ),
    )
    promote_directory(
        workspace.stage_fd,
        workspace.cache_name,
        workspace.models_fd,
        workspace.cache_name,
    )
    try:
        promote_file(
            workspace.stage_fd,
            workspace.wasm_name,
            workspace.public_fd,
            workspace.wasm_name,
        )
    except BaseException:
        remove_tree(workspace.models_fd, workspace.cache_name)
        raise


def run_build(args, path, workspace, report, started):
    mod, weights = load_and_import_model(args, path, workspace, report, started)
    target = create_target()
    lowered = lower_model(args, mod, target, workspace, report, started)
    validate_q4_kernels(lowered, report)
    executable = compile_model(lowered, target, report, started)
    export_wasm(executable, workspace, report, started)
    write_tensor_cache(args, weights, workspace, report, started)
    promote_outputs(workspace, report)


def report_filename(args):
    if args.dynamic:
        return "tvm-q4-build-dynamic.json"
    if args.seq_len == 69:
        return "tvm-q4-build.json"
    return f"tvm-q4-build-seq{args.seq_len}.json"


def main(argv=None):
    args = parse_arguments(argv)
    suffix = build_suffix(args)
    path = model_path(ROOT, suffix)
    workspace = open_workspace(ROOT, suffix)
    try:
        report = build_report(args, path)
        started = time.perf_counter()
        try:
            run_build(args, path, workspace, report, started)
        except BaseException as error:
            report.update(
                status="failed",
                errorType=type(error).__name__,
                error=str(error),
                traceback=traceback.format_exc(),
            )
        report["elapsedSeconds"] = time.perf_counter() - started
        report_text = json.dumps(report, indent=2) + "\n"
        atomic_write(
            workspace.reports_fd,
            report_filename(args),
            lambda output: output.write(report_text.encode()),
        )
    finally:
        close_workspace(workspace)

    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
