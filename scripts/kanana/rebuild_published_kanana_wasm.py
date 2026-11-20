"""Rebuild the published Kanana WASM using the original model conversion and a stable host shape heap."""

import hashlib
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

import tvm
from kanana_shape_heap import canonicalize_shape_heap

ROOT = Path(__file__).resolve().parents[2]
RECIPE = (
    ROOT / "packages/web-decision/src/catalog/provenance/kanana-conversion-recipe.json"
)


def published_recipe(graph):
    return json.loads(RECIPE.read_text(encoding="utf-8"))[graph]


def assert_published_compiler(graph, original):
    source = Path(original.__file__).resolve()
    recipe = published_recipe(graph)
    if source.relative_to(ROOT).as_posix() != recipe["script"]:
        raise ValueError(f"Unexpected Kanana {graph} compiler path.")
    if hashlib.sha256(source.read_bytes()).hexdigest() != recipe["scriptSha256"]:
        raise ValueError(f"The historical Kanana {graph} compiler bytes changed.")


def verify_published_bytes(graph, path):
    expected = published_recipe(graph)["output"]
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if (len(payload), digest) != (expected["bytes"], expected["sha256"]):
        raise RuntimeError(
            f"Rebuilt Kanana {graph} differs from the published WASM: {len(payload)} bytes, SHA-256 {digest}."
        )


def compiled(module, target, graph, base_pipeline):
    normalize_shapes = canonicalize_shape_heap(graph)

    @tvm.transform.module_pass(opt_level=0)
    def pipeline(mod, _ctx):
        return normalize_shapes(base_pipeline(mod))

    return tvm.relax.build(module, target=target, relax_pipeline=pipeline)


def rebuild_text():
    original = importlib.import_module("prototype_text_tvm_relax")
    assert_published_compiler("text", original)
    run = original._prepare_run(original._parse_arguments())
    if not (
        run.args.target == "webgpu"
        and run.args.checkpoint_full
        and run.args.online_attention
    ):
        raise ValueError("Use the pinned full Kanana text WebGPU recipe.")
    original._save_report(run)
    try:
        original._load_model_stage(run)
        original._build_example_inputs(run)
        original._run_pytorch_stage(run)
        original._run_torch_export_stage(run)
        original._run_relax_import_stage(run)
        original._run_llvm_build_stage(run)
        original._run_cpu_vm_stage(run)

        target = original._build_webgpu_target(run)
        run.report["webgpuAttentionLowering"] = original.install_online_attention_tir(
            run.tvm, run.relax_module, target
        )
        (run.output_dir / "relax-module.py").write_text(
            run.relax_module.script(), encoding="utf-8"
        )
        run.current_stage = "webgpu-build"
        print("stage webgpu-build start", flush=True)
        started = time.perf_counter()
        run.executable = compiled(
            run.relax_module, target, "text", tvm.relax.get_default_pipeline(target)
        )
        run.report["relaxWebgpuBuild"] = "passed"
        original._finish_stage(run, "webgpuBuild", started)
        original._export_webgpu_wasm(run)
        verify_published_bytes("text", Path(run.report["webgpuWasmPath"]))
        original._run_safetensors_manifest_stage(run)
    except Exception as error:
        original._record_pipeline_error(run, error)
    original._finalize_run(run)


def rebuild_vision():
    original = importlib.import_module("export_dynamic_vision_tvm")
    assert_published_compiler("vision", original)
    from tvm.contrib import tvmjs
    from tvm.relax.backend.gpu_generic import pipeline as gpu_pipeline

    args = original.parse_args()
    (
        model_dir,
        output_dir,
        checkpoint,
        checkpoint_hash,
        custom_hashes,
        max_patch_count,
    ) = original.validate_pinned_inputs(args)
    config, vision = original.load_vision_checkpoint(
        model_dir, checkpoint, original.torch
    )
    previous_flash, wrapped, grid, pixel_values, export_inputs, expected = (
        original.create_export_inputs(config, vision, max_patch_count, original.torch)
    )
    try:
        exported = original.export_torch_program(
            wrapped, export_inputs, max_patch_count, output_dir, original.torch
        )
        mod = original.import_relax_program(exported, output_dir)
        original.validate_parameter_inputs(exported, mod, original.torch)
        target = tvm.target.Target(
            "webgpu", host={"kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"}
        )
        base_pipeline = original.build_dynamic_allocation_pipeline(
            target, gpu_pipeline, tvm
        )
        executable = compiled(mod, target, "vision", base_pipeline)
        manifest, constants = original.create_weight_manifest(exported, original.torch)
        original.write_export_references(
            output_dir, manifest, constants, expected, pixel_values, grid
        )
        original.write_cases(output_dir, wrapped, config, original.torch, original.np)
        wasm_path = original.export_wasm(executable, output_dir, tvmjs)
        verify_published_bytes("vision", wasm_path)
        subprocess.run(
            [
                "node",
                "--import",
                "tsx",
                str(ROOT / "scripts/kanana/hash_safetensors_ranges.mts"),
                str(checkpoint),
                str(output_dir / "range-manifest.json"),
                "--verify-fp16-rn",
            ],
            cwd=ROOT,
            check=True,
        )
        original.write_build_report(
            output_dir,
            checkpoint_hash,
            custom_hashes,
            original.torch,
            tvm,
            manifest,
            max_patch_count,
            wasm_path,
        )
    finally:
        handlers = original.ALL_ATTENTION_FUNCTIONS
        if previous_flash is None:
            handlers.pop("flash_attention_2", None)
        else:
            handlers.register("flash_attention_2", previous_flash)


def main():
    commands = {"text": rebuild_text, "vision": rebuild_vision}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        raise SystemExit(
            "Usage: rebuild_published_kanana_wasm.py {text|vision} [original recipe arguments]"
        )
    graph = sys.argv.pop(1)
    commands[graph]()


if __name__ == "__main__":
    main()
