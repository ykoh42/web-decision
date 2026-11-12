#!/usr/bin/env python3
"""Probe TVM 0.27's PyTorch frontend on the pinned Kanana vision branch.

This uses the custom vision model code and config from the local pinned HF
snapshot, a one-layer reduced-width config, and random bfloat16 weights. It
reuses the cached processor patch/grid tensors. It never opens the 3B checkpoint
or uses a GPU. The second capture mode supplies the vision utilities' existing
precomputed kwargs for this one fixed image and uses eager attention for its
single sequence so export can reach the tensor graph.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import math
import sys
import time
import types
from pathlib import Path

try:
    from pinned_model_files import PINNED_KANANA_MODEL_FILES, verify_pinned_model_files
except ModuleNotFoundError:
    from scripts.kanana.pinned_model_files import (
        PINNED_KANANA_MODEL_FILES,
        verify_pinned_model_files,
    )

MODEL_ID = "kakaocorp/kanana-2-v-decide-3b-2611"
MODEL_REVISION = "12820d654ed9789a0f7f48daeee1703c3483c464"
DEFAULT_MODEL_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--kakaocorp--kanana-2-v-decide-3b-2611"
    / "snapshots"
    / MODEL_REVISION
)
DEFAULT_INPUTS = (
    Path.home() / ".cache/web-decision/kanana-vision-probe/prepared-inputs.npz"
)
PACKAGE = "kanana_cached_vision_probe"
PINNED_PYTHON_MODULES = frozenset(
    Path(name).stem for name in PINNED_KANANA_MODEL_FILES if name.endswith(".py")
)


class PinnedPythonSourceLoader(importlib.machinery.SourceFileLoader):
    """Compile verified source bytes directly, bypassing timestamp or hash pyc files."""

    def get_code(self, fullname):
        source = self.get_data(self.path)
        filename = Path(self.path).name
        expected = PINNED_KANANA_MODEL_FILES[filename]
        actual = hashlib.sha256(source).hexdigest()
        if actual != expected:
            raise ImportError(
                f"Kanana custom source changed before execution: {filename}"
            )
        return compile(source, self.path, "exec", dont_inherit=True)


class PinnedPythonSourceFinder(importlib.abc.MetaPathFinder):
    """Route every local Kanana module import through the pinned byte loader."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir

    def find_spec(self, fullname, path=None, target=None):
        prefix = f"{PACKAGE}."
        if not fullname.startswith(prefix):
            return None
        module_name = fullname[len(prefix) :]
        if "." in module_name or module_name not in PINNED_PYTHON_MODULES:
            raise ModuleNotFoundError(
                f"Unpinned Kanana custom module: {fullname}", name=fullname
            )
        source_path = self.model_dir / f"{module_name}.py"
        loader = PinnedPythonSourceLoader(fullname, str(source_path))
        return importlib.util.spec_from_loader(
            fullname, loader, origin=str(source_path)
        )


def import_pinned_vision_classes(model_dir: Path):
    verify_pinned_model_files(model_dir)
    for name in tuple(sys.modules):
        if name == PACKAGE or name.startswith(f"{PACKAGE}."):
            del sys.modules[name]
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(model_dir)]
    package.__package__ = PACKAGE
    sys.modules[PACKAGE] = package
    finder = PinnedPythonSourceFinder(model_dir)
    sys.meta_path.insert(0, finder)
    try:
        config_module = importlib.import_module(
            f"{PACKAGE}.configuration_kanana_v_decide"
        )
        model_module = importlib.import_module(f"{PACKAGE}.modeling_kanana_v_decide")
    except Exception:
        for name in tuple(sys.modules):
            if name == PACKAGE or name.startswith(f"{PACKAGE}."):
                del sys.modules[name]
        raise
    finally:
        sys.meta_path.remove(finder)
    return (
        config_module.KananaVDecideVisionConfig,
        model_module.KananaVDecideVisionModel,
    )


def summarize_error(error: BaseException) -> dict[str, str]:
    import traceback

    return {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    }


def make_export_module(torch, vision_model, precomputed=None):
    class VisionOutput(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = vision_model
            self.use_precomputed = precomputed is not None
            if precomputed is not None:
                for name, tensor in precomputed.items():
                    if name == "max_seqlen":
                        setattr(self, name, int(tensor))
                    else:
                        self.register_buffer(name, tensor, persistent=False)

        def forward(self, pixel_values, image_grid_thw):
            if not self.use_precomputed:
                output = self.vision(pixel_values, image_grid_thw)
            else:
                output = self.vision(
                    pixel_values,
                    image_grid_thw,
                    position_ids=self.position_ids,
                    interp_indices=self.interp_indices,
                    interp_weights=self.interp_weights,
                    cu_seqlens=self.cu_seqlens,
                    max_seqlen=self.max_seqlen,
                )
            return output.pooler_output

    return VisionOutput().eval()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prepared-inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=1)
    return parser.parse_args()


def load_probe_runtime():
    import numpy as np
    import torch
    import tvm
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        ALL_ATTENTION_FUNCTIONS,
        eager_attention_forward,
    )
    from transformers.vision_utils import (
        get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )

    return {
        "np": np,
        "torch": torch,
        "tvm": tvm,
        "attention_registry": ALL_ATTENTION_FUNCTIONS,
        "eager_attention": eager_attention_forward,
        "get_interpolation": get_vision_interpolation_indices_and_weights,
        "get_position_ids": get_vision_position_ids,
    }


def validate_runtime(args, tvm):
    if not tvm.__version__.startswith("0.27."):
        raise SystemExit(f"expected TVM 0.27, found {tvm.__version__}")
    model_dir = args.model_dir.resolve()
    if model_dir.name != MODEL_REVISION or not (model_dir / "config.json").is_file():
        raise SystemExit(
            f"expected pinned local model config at revision {MODEL_REVISION}: {model_dir}"
        )
    if not args.prepared_inputs.is_file():
        raise SystemExit(f"cached processor fixture is missing: {args.prepared_inputs}")
    return model_dir


def load_reduced_vision(args, model_dir, np, torch):
    with (model_dir / "config.json").open() as file:
        full_config = json.load(file)
    if full_config.get("model_type") != "kanana_v_decide":
        raise SystemExit(
            f"unexpected pinned custom model type: {full_config.get('model_type')}"
        )
    vision_values = dict(full_config["vision_config"])
    vision_values.pop("model_type", None)
    vision_values.update(
        {
            "depth": args.depth,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "num_heads": args.num_heads,
            "out_hidden_size": min(256, args.hidden_size * 2),
            "deepstack_visual_indexes": [],
        }
    )
    with np.load(args.prepared_inputs) as stored:
        pixels_np = np.asarray(stored["pixel_values"], dtype=np.float32)
        grid_np = np.asarray(stored["image_grid_thw"], dtype=np.int64)
    if (
        pixels_np.shape != (256, 1536)
        or grid_np.shape != (1, 3)
        or tuple(grid_np[0]) != (1, 16, 16)
    ):
        raise SystemExit(
            f"cached image patch geometry changed: pixels={pixels_np.shape}, grid={grid_np}"
        )
    if pixels_np.shape[0] != int(np.prod(grid_np[0])):
        raise SystemExit("cached patch count does not match image_grid_thw")

    VisionConfig, VisionModel = import_pinned_vision_classes(model_dir)
    config = VisionConfig(**vision_values)
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    torch.set_num_threads(1)
    vision = VisionModel(config).eval().to(dtype=torch.bfloat16)
    pixel_values = torch.from_numpy(pixels_np)
    image_grid_thw = torch.from_numpy(grid_np)
    return (
        vision_values,
        config,
        vision,
        pixel_values,
        image_grid_thw,
        pixels_np,
        grid_np,
    )


def create_report(args, vision_values, vision, pixels_np, grid_np, torch, tvm):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "scope": "pinned custom vision module, reduced one-layer/96-wide config, random init",
        "checkpoint_opened": False,
        "checkpoint_weights_loaded": False,
        "gpu_used": False,
        "torch_version": torch.__version__,
        "tvm_version": tvm.__version__,
        "frontend": "torch.export.export -> tvm.relax.frontend.torch.from_exported_program",
        "processor_fixture": str(args.prepared_inputs.resolve()),
        "inputs": {
            "pixel_values": list(pixels_np.shape),
            "image_grid_thw": list(grid_np.shape),
        },
        "reduced_config": {
            key: vision_values.get(key)
            for key in (
                "depth",
                "hidden_size",
                "intermediate_size",
                "num_heads",
                "in_channels",
                "patch_size",
                "temporal_patch_size",
                "spatial_merge_size",
                "out_hidden_size",
            )
        },
        "parameter_dtype": str(next(vision.parameters()).dtype),
        "random_parameter_count": sum(
            parameter.numel() for parameter in vision.parameters()
        ),
        "random_parameter_bytes": sum(
            parameter.numel() * parameter.element_size()
            for parameter in vision.parameters()
        ),
    }


def capture_baseline(torch, vision, pixel_values, image_grid_thw):
    # Baseline: the pinned Transformers utility reads tensor values with .tolist().
    baseline = make_export_module(torch, vision)
    baseline_result = {"status": "failed"}
    try:
        started = time.perf_counter()
        baseline_ep = torch.export.export(
            baseline, (pixel_values, image_grid_thw), strict=True
        )
        baseline_result["status"] = "passed"
        baseline_result["seconds"] = round(time.perf_counter() - started, 3)
        baseline_result["graph"] = baseline_ep.graph_module.code
    except Exception as error:
        baseline_result["failure_phase"] = "torch.export.export"
        baseline_result["error"] = summarize_error(error)
    return baseline_result


def create_precomputed_inputs(
    grid, config, patch_count, torch, get_position_ids, get_interpolation
):
    # Retry only the fixed single-image fixture. Transformers explicitly lets
    # callers supply these grid-derived tensors in kwargs to bypass host .tolist().
    position_ids = get_position_ids(grid, config.spatial_merge_size)
    interp_indices, interp_weights = get_interpolation(
        grid,
        num_grid_per_side=int(math.sqrt(config.num_position_embeddings)),
        mode="bilinear",
        align_corners=True,
        spatial_merge_size=config.spatial_merge_size,
    )
    precomputed = {
        "position_ids": position_ids,
        "interp_indices": interp_indices,
        "interp_weights": interp_weights,
        "cu_seqlens": torch.tensor([0, patch_count], dtype=torch.int32),
        "max_seqlen": patch_count,
    }
    return precomputed


def restore_flash_attention(registry, previous_flash):
    if previous_flash is None:
        registry.pop("flash_attention_2", None)
    else:
        registry.register("flash_attention_2", previous_flash)


def fixed_failure_phase(fixed_result):
    if fixed_result["capture_status"] == "failed":
        return "torch.export.export"
    if "import_status" not in fixed_result:
        return "from_exported_program"
    return "relax_cpu_build"


def followup_failure_phase(followup):
    if "capture_status" not in followup:
        return "torch.export.export"
    if "import_status" not in followup:
        return "from_exported_program"
    if "cpu_build_status" not in followup:
        return "relax_cpu_build"
    return "relax_cpu_dispatch"


def run_float32_followup(
    args,
    vision,
    precomputed,
    pixel_values,
    image_grid_thw,
    pixels_np,
    grid_np,
    torch,
    tvm,
    np,
    from_exported_program,
):
    followup = {"parameter_dtype": "torch.float32", "checkpoint_opened": False}
    try:
        if from_exported_program is None:
            from tvm.relax.frontend.torch import from_exported_program

        float32_vision = copy.deepcopy(vision).to(dtype=torch.float32)
        float32_module = make_export_module(torch, float32_vision, precomputed).to(
            dtype=torch.float32
        )
        started = time.perf_counter()
        float32_ep = torch.export.export(
            float32_module, (pixel_values, image_grid_thw), strict=True
        )
        followup["capture_status"] = "passed"
        followup["capture_seconds"] = round(time.perf_counter() - started, 3)
        (args.output_dir / "vision_exported_graph_float32.py.txt").write_text(
            float32_ep.graph_module.code
        )
        started = time.perf_counter()
        float32_mod = from_exported_program(float32_ep, keep_params_as_input=True)
        followup["import_status"] = "passed"
        followup["import_seconds"] = round(time.perf_counter() - started, 3)
        followup["relax_return_type"] = str(float32_mod["main"].ret_ty)
        followup["relax_parameter_count"] = len(float32_mod["main"].params)
        (args.output_dir / "vision_relax_main_float32.r").write_text(
            float32_mod["main"].script()
        )
        started = time.perf_counter()
        float32_executable = tvm.relax.build(float32_mod, target="llvm")
        followup["cpu_build_status"] = "passed"
        followup["cpu_build_seconds"] = round(time.perf_counter() - started, 3)
        followup["status"] = "passed"
        record_float32_dispatch(
            followup,
            float32_module,
            float32_ep,
            float32_executable,
            pixel_values,
            image_grid_thw,
            pixels_np,
            grid_np,
            torch,
            tvm,
            np,
        )
    except Exception as error:
        followup["failure_phase"] = followup_failure_phase(followup)
        followup["error"] = summarize_error(error)
        followup["status"] = "failed"
    return followup


def record_float32_dispatch(
    followup,
    float32_module,
    float32_ep,
    float32_executable,
    pixel_values,
    image_grid_thw,
    pixels_np,
    grid_np,
    torch,
    tvm,
    np,
):
    with torch.no_grad():
        torch_output = (
            float32_module(pixel_values, image_grid_thw).detach().cpu().numpy()
        )
    parameter_by_target = dict(float32_module.named_parameters())
    parameter_tensors = []
    for input_spec in float32_ep.graph_signature.input_specs:
        if input_spec.kind.name != "PARAMETER":
            continue
        target_name = str(input_spec.target)
        parameter = parameter_by_target[target_name]
        parameter_tensors.append(
            tvm.runtime.tensor(np.ascontiguousarray(parameter.detach().cpu().numpy()))
        )
    vm = tvm.relax.VirtualMachine(float32_executable, tvm.cpu())
    vm_result = vm["main"](
        tvm.runtime.tensor(pixels_np),
        tvm.runtime.tensor(grid_np),
        *parameter_tensors,
    )
    tvm_output = np.asarray(vm_result[0].numpy())
    if tvm_output.shape != torch_output.shape:
        raise RuntimeError(
            f"TVM/PyTorch output shape mismatch: {tvm_output.shape} != {torch_output.shape}"
        )
    difference = tvm_output.astype(np.float64) - torch_output.astype(np.float64)
    reference_norm = float(np.linalg.norm(torch_output.astype(np.float64)))
    output_cosine = float(
        np.dot(tvm_output.reshape(-1), torch_output.reshape(-1))
        / (
            np.linalg.norm(tvm_output.reshape(-1))
            * np.linalg.norm(torch_output.reshape(-1))
            + 1e-30
        )
    )
    followup["cpu_dispatch_status"] = "passed"
    followup["torch_output_shape"] = list(torch_output.shape)
    followup["tvm_output_shape"] = list(tvm_output.shape)
    followup["max_abs_error"] = float(np.max(np.abs(difference)))
    followup["mean_abs_error"] = float(np.mean(np.abs(difference)))
    followup["cosine_similarity"] = output_cosine
    followup["torch_output_l2_norm"] = reference_norm


def run_fixed_capture(
    args,
    vision,
    precomputed,
    pixel_values,
    image_grid_thw,
    pixels_np,
    grid_np,
    torch,
    tvm,
    np,
):
    fixed_module = make_export_module(torch, vision, precomputed)
    fixed_result = {"capture_status": "failed"}
    from_exported_program = None
    try:
        started = time.perf_counter()
        exported_program = torch.export.export(
            fixed_module, (pixel_values, image_grid_thw), strict=True
        )
        fixed_result["capture_status"] = "passed"
        fixed_result["capture_seconds"] = round(time.perf_counter() - started, 3)
        graph_nodes = list(exported_program.graph_module.graph.nodes)
        operators = collections.Counter(
            str(node.target)
            for node in graph_nodes
            if node.op in ("call_function", "call_method", "call_module")
        )
        fixed_result["exported_graph"] = {
            "node_count": len(graph_nodes),
            "operator_count": sum(operators.values()),
            "top_operators": operators.most_common(40),
            "output": str(exported_program.graph_signature.user_outputs),
        }
        (args.output_dir / "vision_exported_graph.py.txt").write_text(
            exported_program.graph_module.code
        )
        from tvm.relax.frontend.torch import from_exported_program

        started = time.perf_counter()
        mod = from_exported_program(exported_program, keep_params_as_input=True)
        fixed_result["import_status"] = "passed"
        fixed_result["import_seconds"] = round(time.perf_counter() - started, 3)
        fixed_result["relax_return_type"] = str(mod["main"].ret_ty)
        fixed_result["relax_parameter_count"] = len(mod["main"].params)
        (args.output_dir / "vision_relax_main.r").write_text(mod["main"].script())
        started = time.perf_counter()
        tvm.relax.build(mod, target="llvm")
        fixed_result["cpu_build_status"] = "passed"
        fixed_result["cpu_build_seconds"] = round(time.perf_counter() - started, 3)
        fixed_result["status"] = "passed"
    except Exception as error:
        fixed_result["failure_phase"] = fixed_failure_phase(fixed_result)
        fixed_result["error"] = summarize_error(error)
        fixed_result["status"] = "failed"
        if fixed_result.get("capture_status") == "passed":
            fixed_result["float32_followup_probe"] = run_float32_followup(
                args,
                vision,
                precomputed,
                pixel_values,
                image_grid_thw,
                pixels_np,
                grid_np,
                torch,
                tvm,
                np,
                from_exported_program,
            )
    return fixed_result


def run_fixed_grid_probe(
    args,
    vision,
    config,
    pixel_values,
    image_grid_thw,
    pixels_np,
    grid_np,
    torch,
    tvm,
    np,
    attention_registry,
    eager_attention,
    get_position_ids,
    get_interpolation,
):
    patch_count = int(pixels_np.shape[0])
    precomputed = create_precomputed_inputs(
        image_grid_thw,
        config,
        patch_count,
        torch,
        get_position_ids,
        get_interpolation,
    )
    # The custom attention has a Python split based on cu_seqlens.tolist(). For
    # one image there is one attention segment, so route the flash control path
    # to eager attention; this is an export-only fixed-fixture specialization.
    previous_flash = attention_registry.get("flash_attention_2")
    attention_registry.register("flash_attention_2", eager_attention)
    config._attn_implementation = "flash_attention_2"
    try:
        return run_fixed_capture(
            args,
            vision,
            precomputed,
            pixel_values,
            image_grid_thw,
            pixels_np,
            grid_np,
            torch,
            tvm,
            np,
        )
    finally:
        restore_flash_attention(attention_registry, previous_flash)


def finalize_report(report, fixed_result):
    report["fixed_grid_precompute_probe"] = fixed_result
    float32_result = fixed_result.get("float32_followup_probe", {})
    report["status"] = (
        "passed"
        if fixed_result.get("status") == "passed"
        else "passed_with_float32_diagnostic"
        if float32_result.get("status") == "passed"
        else "failed"
    )


def write_report(args, report):
    (args.output_dir / "probe-report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] == "failed":
        raise SystemExit(1)


def main():
    args = parse_args()
    runtime = load_probe_runtime()
    model_dir = validate_runtime(args, runtime["tvm"])
    vision_values, config, vision, pixel_values, image_grid_thw, pixels_np, grid_np = (
        load_reduced_vision(args, model_dir, runtime["np"], runtime["torch"])
    )
    report = create_report(
        args,
        vision_values,
        vision,
        pixels_np,
        grid_np,
        runtime["torch"],
        runtime["tvm"],
    )
    report["baseline_capture"] = capture_baseline(
        runtime["torch"], vision, pixel_values, image_grid_thw
    )
    fixed_result = run_fixed_grid_probe(
        args,
        vision,
        config,
        pixel_values,
        image_grid_thw,
        pixels_np,
        grid_np,
        runtime["torch"],
        runtime["tvm"],
        runtime["np"],
        runtime["attention_registry"],
        runtime["eager_attention"],
        runtime["get_position_ids"],
        runtime["get_interpolation"],
    )
    finalize_report(report, fixed_result)
    write_report(args, report)


if __name__ == "__main__":
    main()
