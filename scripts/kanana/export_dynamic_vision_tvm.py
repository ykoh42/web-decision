#!/usr/bin/env python3
"""Export the pinned Kanana vision tower as a dynamic-shape TVM WebGPU graph."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import tvm
from safetensors import safe_open
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
)
from transformers.vision_utils import (
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "kakaocorp/kanana-2-v-decide-3b-2611"
MODEL_REVISION = "12820d654ed9789a0f7f48daeee1703c3483c464"
CHECKPOINT_BYTES = 7_330_945_212
CHECKPOINT_SHA256 = "528930fa261b31ed1f47eecbe7d08efd2e226207508124ae186e1a65dbbbda3b"
PINNED_CUSTOM_FILES = {
    "config.json": "64f4a103accb8946bb7ded7dc5d200f9d6041df3eaa891c04124bb41fa06a69b",
    "configuration_kanana_v_decide.py": "7dfa4a0480db133aec3a2da8411db2232aea58097deb1e626978b554c421ec93",
    "modeling_kanana_v_decide.py": "7e7abf64026f9c30e21fb663a93b5206b9b97795fea7dabc8769c967d5812be1",
}
PROCESSOR_MAX_PIXELS = 16_777_216
VISION_PATCH_AREA = 16 * 16
DEFAULT_MODEL_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--kakaocorp--kanana-2-v-decide-3b-2611"
    / "snapshots"
    / MODEL_REVISION
)
try:
    from build_evidence import add_tvm_build_evidence
    from probe_torch_export_vision_tvm import import_pinned_vision_classes
except ModuleNotFoundError:
    from scripts.kanana.build_evidence import add_tvm_build_evidence
    from scripts.kanana.probe_torch_export_vision_tvm import (
        import_pinned_vision_classes,
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_cases(output_dir, vision, config, torch, np):
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for name, height, width in (
        ("minimum-256", 16, 16),
        ("middle-384", 16, 24),
        ("large-512", 16, 32),
    ):
        patch_count = height * width
        grid = torch.tensor([[1, height, width]], dtype=torch.int64)
        positions = get_vision_position_ids(grid, config.spatial_merge_size).to(
            dtype=torch.int32
        )
        indices, interpolation_weights = get_vision_interpolation_indices_and_weights(
            grid,
            num_grid_per_side=int(math.sqrt(config.num_position_embeddings)),
            mode="bilinear",
            align_corners=True,
            spatial_merge_size=config.spatial_merge_size,
        )
        pixels = torch.randn(
            (patch_count, 1536),
            dtype=torch.float32,
            generator=torch.Generator().manual_seed(patch_count),
        )
        merged_tokens = torch.zeros((patch_count // 4,), dtype=torch.int32)
        with torch.inference_mode():
            expected = vision(
                pixels,
                positions,
                indices.to(torch.int32),
                interpolation_weights,
                merged_tokens,
            )
        arrays = {
            "pixels": pixels.numpy(),
            "positions": positions.numpy(),
            "indices": indices.to(torch.int32).numpy(),
            "weights": interpolation_weights.numpy(),
            "merged_tokens": merged_tokens.numpy(),
            "expected": expected.numpy(),
        }
        files = {}
        for key, array in arrays.items():
            filename = f"{name}-{key}.bin"
            (cases_dir / filename).write_bytes(np.ascontiguousarray(array).tobytes())
            files[key] = filename
        cases.append(
            {
                "name": name,
                "grid": [1, height, width],
                "pixelShape": list(pixels.shape),
                "positionShape": list(positions.shape),
                "indexShape": list(indices.shape),
                "weightShape": list(interpolation_weights.shape),
                "mergedShape": list(merged_tokens.shape),
                "expectedShape": list(expected.shape),
                "files": files,
            }
        )
        print(
            json.dumps(
                {"stage": "reference-case", "case": name, "shape": list(expected.shape)}
            ),
            flush=True,
        )
    (output_dir / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate_model_inputs(args):
    if not tvm.__version__.startswith("0.27."):
        raise SystemExit(f"expected TVM 0.27, found {tvm.__version__}")
    model_dir = args.model_dir.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if model_dir.name != MODEL_REVISION:
        raise SystemExit(
            f"expected pinned model revision {MODEL_REVISION}: {model_dir}"
        )
    checkpoint = model_dir / "model.safetensors"
    if not checkpoint.is_file() or checkpoint.stat().st_size != CHECKPOINT_BYTES:
        raise SystemExit(
            "the local checkpoint is missing or does not match the pinned byte length"
        )
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise SystemExit("the local checkpoint does not match the pinned SHA-256")
    custom_hashes = {
        name: sha256_file(model_dir / name) for name in PINNED_CUSTOM_FILES
    }
    if custom_hashes != PINNED_CUSTOM_FILES:
        raise SystemExit(
            "the local model config or custom model code differs from the pinned revision"
        )
    max_patch_count = PROCESSOR_MAX_PIXELS // VISION_PATCH_AREA
    return (
        model_dir,
        output_dir,
        checkpoint,
        checkpoint_hash,
        custom_hashes,
        max_patch_count,
    )


def load_vision_tower(model_dir, checkpoint, torch):
    config_data = json.loads((model_dir / "config.json").read_text())["vision_config"]
    config_data.pop("model_type", None)
    VisionConfig, VisionModel = import_pinned_vision_classes(model_dir)
    config = VisionConfig(**config_data)
    config._attn_implementation = "eager"
    vision = VisionModel(config).eval().to(dtype=torch.float32)
    expected_keys = set(vision.state_dict())
    loaded = {}
    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        for name in source.keys():
            if name.startswith("model.visual."):
                loaded[name.removeprefix("model.visual.")] = source.get_tensor(name).to(
                    dtype=torch.float32
                )
    missing = expected_keys.difference(loaded)
    unexpected = set(loaded).difference(expected_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"vision weight names mismatch: missing={sorted(missing)[:10]}, unexpected={sorted(unexpected)[:10]}"
        )
    load_result = vision.load_state_dict(loaded, strict=True)
    del loaded
    print(
        json.dumps(
            {
                "stage": "checkpoint",
                "weights": 333,
                "missing": load_result.missing_keys,
                "unexpected": load_result.unexpected_keys,
            }
        ),
        flush=True,
    )
    return config, vision


def make_vision_inputs(config, torch, patch_count):
    torch.manual_seed(20261110)
    grid = torch.tensor([[1, 16, 16]], dtype=torch.int64)
    pixel_values = torch.randn((patch_count, 1536), dtype=torch.float32)
    position_ids = get_vision_position_ids(grid, config.spatial_merge_size)
    interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
        grid,
        num_grid_per_side=int(math.sqrt(config.num_position_embeddings)),
        mode="bilinear",
        align_corners=True,
        spatial_merge_size=config.spatial_merge_size,
    )
    return grid, pixel_values, position_ids, interp_indices, interp_weights


def register_export_attention(config):
    previous_flash = ALL_ATTENTION_FUNCTIONS.get("flash_attention_2")
    ALL_ATTENTION_FUNCTIONS.register("flash_attention_2", eager_attention_forward)
    config._attn_implementation = "flash_attention_2"

    return previous_flash


def make_dynamic_vision(vision, patch_count, torch):
    class DynamicVision(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = vision
            self.register_buffer(
                "cu_seqlens",
                torch.tensor([0, patch_count], dtype=torch.int32),
                persistent=False,
            )

        def forward(self, pixels, positions, indices, weights, merged_tokens):
            hidden = self.vision.patch_embed(pixels)
            position_embeddings = (
                self.vision.pos_embed(indices) * weights[:, :, None]
            ).sum(1)
            hidden = hidden + position_embeddings.to(hidden.dtype)
            rotary = self.vision.rotary_pos_emb(hidden, positions)
            for block in self.vision.blocks:
                hidden = block(
                    hidden,
                    cu_seqlens=self.cu_seqlens,
                    max_seqlen=pixels.shape[0],
                    position_embeddings=rotary,
                )
            hidden = self.vision.merger.norm(hidden)
            hidden = hidden.reshape(
                merged_tokens.shape[0], self.vision.merger.hidden_size
            )
            hidden = self.vision.merger.linear_fc2(
                self.vision.merger.act_fn(self.vision.merger.linear_fc1(hidden))
            )
            return hidden

    return DynamicVision().eval()


def export_vision_graph(wrapped, inputs, grid, max_patch_count, output_dir, torch):
    pixel_values, position_ids, interp_indices, interp_weights, merged_tokens = inputs
    export_inputs = (
        pixel_values,
        position_ids.to(dtype=torch.int32),
        interp_indices.to(dtype=torch.int32),
        interp_weights,
        merged_tokens,
    )
    with torch.inference_mode():
        started = time.perf_counter()
        expected = wrapped(*export_inputs)
        print(
            json.dumps(
                {
                    "stage": "pytorch-reference",
                    "seconds": round(time.perf_counter() - started, 3),
                    "shape": list(expected.shape),
                    "dtype": str(expected.dtype),
                    "norm": float(expected.norm()),
                }
            ),
            flush=True,
        )
    merged = torch.export.Dim("merged_tokens", min=64, max=max_patch_count // 4)
    patches = 4 * merged
    dynamic_shapes = (
        {0: patches},
        {0: patches},
        {0: patches},
        {0: patches},
        {0: merged},
    )
    started = time.perf_counter()
    exported = torch.export.export(
        wrapped, export_inputs, dynamic_shapes=dynamic_shapes, strict=False
    )
    print(
        json.dumps(
            {
                "stage": "torch-export",
                "seconds": round(time.perf_counter() - started, 3),
                "nodes": len(list(exported.graph_module.graph.nodes)),
            }
        ),
        flush=True,
    )
    (output_dir / "exported-program.py").write_text(exported.graph_module.code)
    return expected, exported, pixel_values, grid


def narrow_take_indices(mod, tvm, relax, expr_functor):
    @expr_functor.mutator
    class NarrowTakeIndices(relax.PyExprMutator):
        def __init__(self, current_module):
            super().__init__(current_module)
            self.rewritten = 0

        def visit_call_(self, call):
            call = super().visit_call_(call)
            if call.op.name != "relax.take":
                return call
            value, index = call.args
            if (
                isinstance(index, tvm.ir.expr.GenericConst)
                and str(index.value.dtype) == "int64"
            ):
                self.rewritten += 1
                return relax.Call(
                    call.op,
                    [value, relax.const(int(index.value.numpy()), "int32")],
                    call.attrs,
                    call.ty_args,
                    call.span,
                )
            return call

    rewriter = NarrowTakeIndices(mod)
    for global_var, function in mod.functions_items():
        if isinstance(function, relax.Function):
            rewriter.builder_.update_func(global_var, rewriter.visit_expr(function))
    mod = rewriter.builder_.get()
    print(
        json.dumps({"stage": "rewrite-take-indices", "count": rewriter.rewritten}),
        flush=True,
    )
    return mod


def import_relax_graph(exported, output_dir, torch, tvm):
    from tvm import relax
    from tvm.relax import expr_functor
    from tvm.relax.frontend.torch import from_exported_program

    started = time.perf_counter()
    mod = from_exported_program(
        exported,
        keep_params_as_input=True,
        run_ep_decomposition=True,
        unwrap_unit_return_tuple=True,
    )
    print(
        json.dumps(
            {
                "stage": "relax-import",
                "seconds": round(time.perf_counter() - started, 3),
                "type": str(mod["main"].ret_ty),
                "params": len(mod["main"].params),
            }
        ),
        flush=True,
    )
    mod = narrow_take_indices(mod, tvm, relax, expr_functor)
    (output_dir / "relax-module.py").write_text(mod.script())
    param_names = [param.name for param in mod["main"].params]
    parameters = {
        spec.arg.name: exported.state_dict[spec.target]
        for spec in exported.graph_signature.input_specs
        if spec.kind is torch.export.graph_signature.InputKind.PARAMETER
    }
    if set(parameters).difference(param_names):
        raise RuntimeError(
            f"missing TVM parameters: {sorted(set(parameters).difference(param_names))[:10]}"
        )
    return mod


def build_webgpu_executable(mod, tvm):
    target = tvm.target.Target(
        "webgpu", host={"kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"}
    )
    from tvm.relax.backend.gpu_generic import pipeline as gpu_pipeline

    def dynamic_allocation_pipeline(module):
        passes = (
            gpu_pipeline.library_dispatch_passes(target)
            + gpu_pipeline.legalize_passes(target)
            + gpu_pipeline.dataflow_lower_passes(target)
            + [
                tvm.relax.transform.LowerAllocTensor(),
                tvm.relax.transform.KillAfterLastUse(),
                tvm.relax.transform.LowerRuntimeBuiltin(),
                tvm.relax.transform.ComputePrimValue(),
                tvm.relax.transform.VMShapeLower(),
                tvm.relax.transform.AttachGlobalSymbol(),
            ]
        )
        return tvm.transform.Sequential(passes)(module)

    return tvm.relax.build(
        mod, target=target, relax_pipeline=dynamic_allocation_pipeline
    )


def build_parameter_manifest(exported):
    manifest = []
    for spec in exported.graph_signature.input_specs:
        if spec.kind.name != "PARAMETER":
            continue
        tensor = exported.state_dict[spec.target]
        item = {
            "kind": spec.kind.name,
            "parameterName": spec.arg.name,
            "target": spec.target,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "length": tensor.numel() * tensor.element_size(),
        }
        if spec.kind.name == "PARAMETER":
            item["name"] = "model.visual." + spec.target.removeprefix("vision.")
            manifest.append(item)
    return manifest


def write_graph_artifacts(
    output_dir, manifest, expected, pixel_values, grid, wrapped, config, torch, np
):
    constants = {}
    (output_dir / "external-weight-manifest.json").write_text(
        json.dumps({"weights": manifest}, indent=2) + "\n"
    )
    (output_dir / "graph-constants.json").write_text(
        json.dumps(constants, separators=(",", ":")) + "\n"
    )
    (output_dir / "pytorch-reference.json").write_text(
        json.dumps(
            {
                "shape": list(expected.shape),
                "values": expected.detach().cpu().float().reshape(-1).tolist(),
                "pixelShape": list(pixel_values.shape),
                "grid": grid.tolist(),
            }
        )
        + "\n"
    )
    write_cases(output_dir, wrapped, config, torch, np)
    return constants


def export_webgpu_wasm(exe, output_dir, tvm):
    wasm_path = output_dir / "kanana-vision-model.wasm"
    from tvm.contrib import tvmjs

    previous_tvm_home = os.environ.get("TVM_HOME")
    tvm_home = ROOT / ".toolchains/apache-tvm-src"
    os.environ["TVM_HOME"] = str(tvm_home)
    try:
        exe.export_library(
            str(wasm_path),
            fcompile=tvmjs.create_tvmjs_wasm,
            options=[f"-ffile-prefix-map={ROOT}=/src"],
        )
    finally:
        if previous_tvm_home is None:
            os.environ.pop("TVM_HOME", None)
        else:
            os.environ["TVM_HOME"] = previous_tvm_home
    return wasm_path


def verify_checkpoint_ranges(checkpoint, output_dir):
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


def write_build_report(
    output_dir,
    manifest,
    max_patch_count,
    wasm_path,
    checkpoint_hash,
    custom_hashes,
    torch,
    tvm,
):
    report = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "checkpointSha256": checkpoint_hash,
        "customFileSha256": custom_hashes,
        "torchVersion": torch.__version__,
        "tvmVersion": tvm.__version__,
        "visionWeights": len(manifest),
        "maxPatchCount": max_patch_count,
        "graphParameters": [weight["parameterName"] for weight in manifest],
        "memoryPipeline": "dynamic allocations without StaticPlanBlockMemory",
        "testCases": ["synthetic tensor parity at 256, 384, and 512 patches"],
        "wasmBytes": wasm_path.stat().st_size,
    }
    report = add_tvm_build_evidence(
        report,
        model_id="kanana-2-v-decide-3b-2611",
        repository=MODEL_ID,
        revision=MODEL_REVISION,
        checkpoint_sha256=checkpoint_hash,
        graph="vision",
        wasm_path=wasm_path,
        build_root=output_dir,
        tvm_version=tvm.__version__,
    )
    (output_dir / "build-report.json").write_text(json.dumps(report, indent=2) + "\n")


def restore_export_attention(previous_flash):
    if previous_flash is None:
        ALL_ATTENTION_FUNCTIONS.pop("flash_attention_2", None)
    else:
        ALL_ATTENTION_FUNCTIONS.register("flash_attention_2", previous_flash)


def main():
    args = parse_args()
    (
        model_dir,
        output_dir,
        checkpoint,
        checkpoint_hash,
        custom_hashes,
        max_patch_count,
    ) = validate_model_inputs(args)
    torch.set_num_threads(1)
    config, vision = load_vision_tower(model_dir, checkpoint, torch)
    patch_count = 256
    grid, pixel_values, position_ids, interp_indices, interp_weights = (
        make_vision_inputs(config, torch, patch_count)
    )
    previous_flash = register_export_attention(config)
    wrapped = make_dynamic_vision(vision, patch_count, torch)
    merged_tokens = torch.zeros((patch_count // 4,), dtype=torch.int32)
    inputs = (pixel_values, position_ids, interp_indices, interp_weights, merged_tokens)
    expected, exported, pixel_values, grid = export_vision_graph(
        wrapped, inputs, grid, max_patch_count, output_dir, torch
    )
    mod = import_relax_graph(exported, output_dir, torch, tvm)
    exe = build_webgpu_executable(mod, tvm)
    manifest = build_parameter_manifest(exported)
    constants = write_graph_artifacts(
        output_dir, manifest, expected, pixel_values, grid, wrapped, config, torch, np
    )
    wasm_path = export_webgpu_wasm(exe, output_dir, tvm)
    print(
        json.dumps(
            {
                "stage": "webgpu-wasm-export",
                "bytes": wasm_path.stat().st_size,
                "parameters": len(manifest),
                "constants": list(constants),
            }
        ),
        flush=True,
    )
    verify_checkpoint_ranges(checkpoint, output_dir)
    write_build_report(
        output_dir,
        manifest,
        max_patch_count,
        wasm_path,
        checkpoint_hash,
        custom_hashes,
        torch,
        tvm,
    )
    restore_export_attention(previous_flash)


if __name__ == "__main__":
    main()
