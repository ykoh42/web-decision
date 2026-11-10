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


def _parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prepared-inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=1)
    return parser.parse_args()


def _load_probe_dependencies():
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
        "attentionFunctions": ALL_ATTENTION_FUNCTIONS,
        "eagerAttention": eager_attention_forward,
        "getInterpolation": get_vision_interpolation_indices_and_weights,
        "getPositionIds": get_vision_position_ids,
    }


def _read_probe_inputs(args, dependencies):
    tvm_version = dependencies["tvm"].__version__
    if not tvm_version.startswith("0.27."):
        raise SystemExit(f"expected TVM 0.27, found {tvm_version}")
    model_dir = args.model_dir.resolve()
    if model_dir.name != MODEL_REVISION or not (model_dir / "config.json").is_file():
        raise SystemExit(
            f"expected pinned local model config at revision {MODEL_REVISION}: {model_dir}"
        )
    if not args.prepared_inputs.is_file():
        raise SystemExit(f"cached processor fixture is missing: {args.prepared_inputs}")
    with (model_dir / "config.json").open() as file:
        full_config = json.load(file)
    model_type = full_config.get("model_type")
    if model_type != "kanana_v_decide":
        raise SystemExit(f"unexpected pinned custom model type: {model_type}")
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
    with dependencies["np"].load(args.prepared_inputs) as stored:
        pixels_np = dependencies["np"].asarray(
            stored["pixel_values"], dtype=dependencies["np"].float32
        )
        grid_np = dependencies["np"].asarray(
            stored["image_grid_thw"], dtype=dependencies["np"].int64
        )
    valid_geometry = (
        pixels_np.shape == (256, 1536)
        and grid_np.shape == (1, 3)
        and tuple(grid_np[0]) == (1, 16, 16)
    )
    if not valid_geometry:
        raise SystemExit(
            f"cached image patch geometry changed: pixels={pixels_np.shape}, grid={grid_np}"
        )
    if pixels_np.shape[0] != int(dependencies["np"].prod(grid_np[0])):
        raise SystemExit("cached patch count does not match image_grid_thw")
    return model_dir, full_config, vision_values, pixels_np, grid_np


def _make_probe_model(args, dependencies, inputs):
    model_dir, full_config, vision_values, pixels_np, grid_np = inputs
    VisionConfig, VisionModel = import_pinned_vision_classes(model_dir)
    config = VisionConfig(**vision_values)
    config._attn_implementation = "eager"
    torch = dependencies["torch"]
    torch.manual_seed(0)
    torch.set_num_threads(1)
    vision = VisionModel(config).eval().to(dtype=torch.bfloat16)
    pixel_values = torch.from_numpy(pixels_np)
    image_grid_thw = torch.from_numpy(grid_np)
    return config, vision, pixel_values, image_grid_thw


def _make_probe_report(
    args, dependencies, config, vision, vision_values, pixels_np, grid_np
):
    torch, tvm = dependencies["torch"], dependencies["tvm"]
    reduced_config = {
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
    }
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
        "reduced_config": reduced_config,
        "parameter_dtype": str(next(vision.parameters()).dtype),
        "random_parameter_count": sum(
            parameter.numel() for parameter in vision.parameters()
        ),
        "random_parameter_bytes": sum(
            parameter.numel() * parameter.element_size()
            for parameter in vision.parameters()
        ),
    }


def _prepare_probe(args):
    dependencies = _load_probe_dependencies()
    inputs = _read_probe_inputs(args, dependencies)
    config, vision, pixels, grid = _make_probe_model(args, dependencies, inputs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = _make_probe_report(
        args, dependencies, config, vision, inputs[2], inputs[3], inputs[4]
    )
    return {
        "args": args,
        "dependencies": dependencies,
        "modelDir": inputs[0],
        "config": config,
        "vision": vision,
        "pixelValues": pixels,
        "imageGrid": grid,
        "pixelsNumpy": inputs[3],
        "gridNumpy": inputs[4],
        "report": report,
    }


def _run_baseline_capture(state):
    torch = state["dependencies"]["torch"]
    baseline = make_export_module(torch, state["vision"])
    result = {"status": "failed"}
    try:
        started = time.perf_counter()
        exported = torch.export.export(
            baseline, (state["pixelValues"], state["imageGrid"]), strict=True
        )
        result["status"] = "passed"
        result["seconds"] = round(time.perf_counter() - started, 3)
        result["graph"] = exported.graph_module.code
    except Exception as error:
        result["failure_phase"] = "torch.export.export"
        result["error"] = summarize_error(error)
    return result


def _precompute_fixed_grid(state):
    deps = state["dependencies"]
    torch, config = deps["torch"], state["config"]
    grid = state["imageGrid"]
    positions = deps["getPositionIds"](grid, config.spatial_merge_size)
    indices, weights = deps["getInterpolation"](
        grid,
        num_grid_per_side=int(math.sqrt(config.num_position_embeddings)),
        mode="bilinear",
        align_corners=True,
        spatial_merge_size=config.spatial_merge_size,
    )
    patch_count = int(state["pixelsNumpy"].shape[0])
    precomputed = {
        "position_ids": positions,
        "interp_indices": indices,
        "interp_weights": weights,
        "cu_seqlens": torch.tensor([0, patch_count], dtype=torch.int32),
        "max_seqlen": patch_count,
    }
    attention_functions = deps["attentionFunctions"]
    previous_flash = attention_functions.get("flash_attention_2")
    attention_functions.register("flash_attention_2", deps["eagerAttention"])
    config._attn_implementation = "flash_attention_2"
    return precomputed, previous_flash


def _restore_attention_function(state, previous_flash):
    attention_functions = state["dependencies"]["attentionFunctions"]
    if previous_flash is None:
        attention_functions.pop("flash_attention_2", None)
    else:
        attention_functions.register("flash_attention_2", previous_flash)


def _capture_fixed_program(state, result):
    torch = state["dependencies"]["torch"]
    started = time.perf_counter()
    exported = torch.export.export(
        make_export_module(torch, state["vision"], state["precomputed"]),
        (state["pixelValues"], state["imageGrid"]),
        strict=True,
    )
    result["capture_status"] = "passed"
    result["capture_seconds"] = round(time.perf_counter() - started, 3)
    nodes = list(exported.graph_module.graph.nodes)
    operators = collections.Counter(
        str(node.target)
        for node in nodes
        if node.op in ("call_function", "call_method", "call_module")
    )
    result["exported_graph"] = {
        "node_count": len(nodes),
        "operator_count": sum(operators.values()),
        "top_operators": operators.most_common(40),
        "output": str(exported.graph_signature.user_outputs),
    }
    path = state["args"].output_dir / "vision_exported_graph.py.txt"
    path.write_text(exported.graph_module.code)
    return exported


def _import_fixed_program(state, result, exported):
    from tvm.relax.frontend.torch import from_exported_program

    started = time.perf_counter()
    module = from_exported_program(exported, keep_params_as_input=True)
    result["import_status"] = "passed"
    result["import_seconds"] = round(time.perf_counter() - started, 3)
    result["relax_return_type"] = str(module["main"].ret_ty)
    result["relax_parameter_count"] = len(module["main"].params)
    path = state["args"].output_dir / "vision_relax_main.r"
    path.write_text(module["main"].script())
    return module


def _build_fixed_cpu(tvm, result, module):
    started = time.perf_counter()
    tvm.relax.build(module, target="llvm")
    result["cpu_build_status"] = "passed"
    result["cpu_build_seconds"] = round(time.perf_counter() - started, 3)
    result["status"] = "passed"


def _fixed_failure_phase(result):
    if result["capture_status"] == "failed":
        return "torch.export.export"
    if "import_status" not in result:
        return "from_exported_program"
    return "relax_cpu_build"


def _run_fixed_probe(state, previous_flash):
    result = {"capture_status": "failed"}
    try:
        exported = _capture_fixed_program(state, result)
        module = _import_fixed_program(state, result, exported)
        _build_fixed_cpu(state["dependencies"]["tvm"], result, module)
    except Exception as error:
        result["failure_phase"] = _fixed_failure_phase(result)
        result["error"] = summarize_error(error)
        result["status"] = "failed"
        if result.get("capture_status") == "passed":
            result["float32_followup_probe"] = _run_float32_followup(state)
    finally:
        _restore_attention_function(state, previous_flash)
    return result


def _export_float32_followup(state, followup):
    torch = state["dependencies"]["torch"]
    tvm = state["dependencies"]["tvm"]
    vision = copy.deepcopy(state["vision"]).to(dtype=torch.float32)
    module = make_export_module(torch, vision, state["precomputed"]).to(
        dtype=torch.float32
    )
    started = time.perf_counter()
    exported = torch.export.export(
        module, (state["pixelValues"], state["imageGrid"]), strict=True
    )
    followup["capture_status"] = "passed"
    followup["capture_seconds"] = round(time.perf_counter() - started, 3)
    graph_path = state["args"].output_dir / "vision_exported_graph_float32.py.txt"
    graph_path.write_text(exported.graph_module.code)
    started = time.perf_counter()
    relax_module = tvm.relax.frontend.torch.from_exported_program(
        exported, keep_params_as_input=True
    )
    followup["import_status"] = "passed"
    followup["import_seconds"] = round(time.perf_counter() - started, 3)
    followup["relax_return_type"] = str(relax_module["main"].ret_ty)
    followup["relax_parameter_count"] = len(relax_module["main"].params)
    relax_path = state["args"].output_dir / "vision_relax_main_float32.r"
    relax_path.write_text(relax_module["main"].script())
    started = time.perf_counter()
    executable = tvm.relax.build(relax_module, target="llvm")
    followup["cpu_build_status"] = "passed"
    followup["cpu_build_seconds"] = round(time.perf_counter() - started, 3)
    followup["status"] = "passed"
    return module, exported, executable


def _dispatch_float32_followup(state, followup, module, exported, executable):
    torch, tvm, np = (
        state["dependencies"]["torch"],
        state["dependencies"]["tvm"],
        state["dependencies"]["np"],
    )
    with torch.no_grad():
        torch_output = (
            module(state["pixelValues"], state["imageGrid"]).detach().cpu().numpy()
        )
    parameter_by_target = dict(module.named_parameters())
    parameters = []
    for input_spec in exported.graph_signature.input_specs:
        if input_spec.kind.name != "PARAMETER":
            continue
        parameter = parameter_by_target[str(input_spec.target)]
        parameters.append(
            tvm.runtime.tensor(np.ascontiguousarray(parameter.detach().cpu().numpy()))
        )
    vm = tvm.relax.VirtualMachine(executable, tvm.cpu())
    result = vm["main"](
        tvm.runtime.tensor(state["pixelsNumpy"]),
        tvm.runtime.tensor(state["gridNumpy"]),
        *parameters,
    )
    tvm_output = np.asarray(result[0].numpy())
    if tvm_output.shape != torch_output.shape:
        raise RuntimeError(
            f"TVM/PyTorch output shape mismatch: {tvm_output.shape} != {torch_output.shape}"
        )
    difference = tvm_output.astype(np.float64) - torch_output.astype(np.float64)
    torch_flat = torch_output.reshape(-1).astype(np.float64)
    tvm_flat = tvm_output.reshape(-1).astype(np.float64)
    followup["cpu_dispatch_status"] = "passed"
    followup["torch_output_shape"] = list(torch_output.shape)
    followup["tvm_output_shape"] = list(tvm_output.shape)
    followup["max_abs_error"] = float(np.max(np.abs(difference)))
    followup["mean_abs_error"] = float(np.mean(np.abs(difference)))
    followup["cosine_similarity"] = float(
        np.dot(tvm_flat, torch_flat)
        / (np.linalg.norm(tvm_flat) * np.linalg.norm(torch_flat) + 1e-30)
    )
    followup["torch_output_l2_norm"] = float(np.linalg.norm(torch_flat))


def _float32_failure_phase(followup):
    if "capture_status" not in followup:
        return "torch.export.export"
    if "import_status" not in followup:
        return "from_exported_program"
    if "cpu_build_status" not in followup:
        return "relax_cpu_build"
    return "relax_cpu_dispatch"


def _run_float32_followup(state):
    followup = {"parameter_dtype": "torch.float32", "checkpoint_opened": False}
    try:
        module, exported, executable = _export_float32_followup(state, followup)
        _dispatch_float32_followup(state, followup, module, exported, executable)
    except Exception as error:
        followup["failure_phase"] = _float32_failure_phase(followup)
        followup["error"] = summarize_error(error)
        followup["status"] = "failed"
    return followup


def _final_probe_status(fixed_result):
    float32_result = fixed_result.get("float32_followup_probe", {})
    if fixed_result.get("status") == "passed":
        return "passed"
    if float32_result.get("status") == "passed":
        return "passed_with_float32_diagnostic"
    return "failed"


def _write_probe_report(state, fixed_result):
    report = state["report"]
    report["fixed_grid_precompute_probe"] = fixed_result
    report["status"] = _final_probe_status(fixed_result)
    report_path = state["args"].output_dir / "probe-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] == "failed":
        raise SystemExit(1)


def main():
    args = _parse_arguments()
    state = _prepare_probe(args)
    baseline = _run_baseline_capture(state)
    state["report"]["baseline_capture"] = baseline
    state["precomputed"], previous_flash = _precompute_fixed_grid(state)
    fixed_result = _run_fixed_probe(state, previous_flash)
    _write_probe_report(state, fixed_result)


if __name__ == "__main__":
    main()
