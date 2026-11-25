"""Build the private E2B v1 causal graph with unchanged TVM WebGPU/WASM."""

import argparse
import hashlib
import json
import os
import sys
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path
from subprocess import check_output
from tempfile import TemporaryDirectory

import torch
import tvm
import tvm_ffi
from canonical_shape_stores import canonical_stores
from causal_layer_ir import CausalLayerSource
from pointer_head_ir import import_pointer_head, load_pointer_head
from safetensors import safe_open
from source_inventory import read_pinned_inventory
from tiled_causal_decoder import load_norm
from tiled_causal_layer import import_stage, import_tiled_layer
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding
from tvm import relax
from webgpu_model_grid import verify_and_guard_grids

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.extend(str(SCRIPTS / name) for name in ("gemma4_v2", "qwen35"))

TEXT = "model.language_model."
LAYER_BASE = TEXT + "layers."
LAYER_ADAPTER = "base_model.model.layers."
FAMILIES = {
    0: "local_owner",
    4: "global_owner",
    15: "local_shared",
    19: "global_shared",
}


def family_name(index, kind, config):
    locality = "global" if kind == "full_attention" else "local"
    sharing = (
        "shared"
        if index >= config.num_hidden_layers - config.num_kv_shared_layers
        else "owner"
    )
    return f"{locality}_{sharing}"


def remove_unused(program):
    module, params = program
    runtime = int(module["main"].attrs["num_input"])
    original = module["main"].params[runtime:]
    values = {var.name: value for var, value in zip(original, params, strict=True)}
    module = relax.transform.RemoveUnusedParameters()(module)
    return module, [
        (var.name, values[var.name]) for var in module["main"].params[runtime:]
    ]


def export_name(name, stage):
    if stage == "project":
        if name == "input_layernorm.weight":
            return "p_norm_weight"
        if not name.startswith("self_attn."):
            return None
        name = "attention." + name.removeprefix("self_attn.")
    elif stage == "tail":
        name = "block." + name
    return "p_" + name.replace(".", "_").lower()


def layer_names(inventory, index, stage):
    candidates = {}
    for source, prefix in (("base", LAYER_BASE), ("adapter", LAYER_ADAPTER)):
        layer = f"{prefix}{index}."
        for full, meta in inventory[source]["tensors"].items():
            if not full.startswith(layer):
                continue
            short = full.removeprefix(layer)
            if source == "adapter":
                targets = [short.removesuffix(".weight") + ".default.weight"]
            else:
                targets = (
                    [short, short.removesuffix(".weight") + ".base_layer.weight"]
                    if short.endswith(".weight")
                    else [short]
                )
            for target in targets:
                if key := export_name(target, stage):
                    candidates[key] = (source, full, meta)
    return candidates


def binding_record(param, candidate, verify=None):
    source, name, meta = candidate
    shape = list(map(int, param.shape))
    dtype = str(param.dtype)
    if meta["dtype"] == "BF16":
        expected_shape, expected_dtype = [(meta["length"] + 3) // 4], "uint32"
    else:
        expected_shape, expected_dtype = meta["shape"], "float32"
    if (shape, dtype) != (expected_shape, expected_dtype):
        raise ValueError(f"Unexpected transport shape for {name}")
    if verify is not None:
        offset, length = meta["offset"], meta["length"]
        expected = os.pread(verify[source].fileno(), length, offset)
        padding = (-len(expected)) % 4
        if len(expected) != length or param.numpy().tobytes() != expected + bytes(
            padding
        ):
            raise ValueError(f"Converted parameter bytes differ for {name}")
    return {"source": source, "tensor": name, "shape": shape, "dtype": dtype}


def collect_families(source, inventory, handles):
    config = source.config
    hidden = torch.zeros((1, 3, config.hidden_size), dtype=torch.bfloat16)
    per_layer = torch.zeros(
        (1, 3, config.hidden_size_per_layer_input), dtype=torch.bfloat16
    )
    positions = torch.arange(3).unsqueeze(0)
    rotary = Gemma4TextRotaryEmbedding(config)
    modules, layer_params, attentions = [], {}, set()
    for index, family in FAMILIES.items():
        layer = source.load(index)
        if family_name(index, layer.layer_type, config) != family:
            raise ValueError("The pinned layer-family representative differs")
        cosine, sine = rotary(hidden, positions, layer_type=layer.layer_type)
        project, (attention, _), tail = import_tiled_layer(
            layer, (hidden, per_layer, cosine, sine), config.max_position_embeddings
        )
        for stage, program in (("project", project), ("tail", tail)):
            module, parameters = remove_unused(program)
            modules.append((family + "_" + stage, module))
            candidates = layer_names(inventory, index, stage)
            for label, value in parameters:
                binding_record(value, candidates[label], handles)
            layer_params[family, stage] = parameters
        locality = family.split("_")[0]
        if locality not in attentions:
            attentions.add(locality)
            modules.append((locality + "_attention", attention))
    return modules, layer_params, rotary


class PerLayerProjection(torch.nn.Module):
    def __init__(self, rows, norm, config):
        super().__init__()
        self.linear = torch.nn.Linear(
            config.hidden_size, rows.shape[0], bias=False, dtype=torch.bfloat16
        )
        self.linear.weight.data.copy_(rows)
        self.norm = norm
        self.scale = config.hidden_size**-0.5
        self.width = config.hidden_size_per_layer_input

    def forward(self, hidden, embedding):
        projected = (self.linear(hidden) * self.scale).reshape(
            *hidden.shape[:2], -1, self.width
        )
        identity = embedding.reshape(*hidden.shape[:2], -1, self.width)
        return (self.norm(projected) + identity) * 2.0**-0.5


def build_embedding_endpoints(source, inventory, handles):
    config = source.config
    vocab_path = source.base / "model.safetensors"
    hidden = torch.zeros((1, 3, config.hidden_size), dtype=torch.bfloat16)
    identity = torch.zeros(
        (1, 3, config.num_hidden_layers * config.hidden_size_per_layer_input),
        dtype=torch.bfloat16,
    )
    with safe_open(vocab_path, framework="pt", device="cpu") as weights:
        final = load_norm(weights, config, config.hidden_size, "norm.weight")
        norm = load_norm(
            weights,
            config,
            config.hidden_size_per_layer_input,
            "per_layer_projection_norm.weight",
        )
        projection = weights.get_tensor(TEXT + "per_layer_model_projection.weight")
    endpoints = {
        "final": (import_stage(final, (hidden,)), {"p_weight": TEXT + "norm.weight"}),
        "ple": (
            import_stage(
                PerLayerProjection(projection, norm, config), (hidden, identity)
            ),
            {
                "p_linear_weight": TEXT + "per_layer_model_projection.weight",
                "p_norm_weight": TEXT + "per_layer_projection_norm.weight",
            },
        ),
    }
    modules, bindings = [], {}
    for name, (program, expected) in endpoints.items():
        module, parameters = remove_unused(program)
        modules.append((name, module))
        bindings[name] = [
            binding_record(
                value, ("base", tensor, inventory["base"]["tensors"][tensor]), handles
            )
            for label, value in parameters
            for tensor in (expected[label],)
        ]
    return modules, bindings


def layer_records(config, programs, inventory):
    layers = []
    producers = {}
    first_shared = config.num_hidden_layers - config.num_kv_shared_layers
    last_owners = {
        kind: index for index, kind in enumerate(config.layer_types[:first_shared])
    }
    for index, kind in enumerate(config.layer_types):
        family = family_name(index, kind, config)
        shared = index >= first_shared
        records = {
            "family": family,
            "window": config.sliding_window if kind == "sliding_attention" else None,
            "producer": producers[kind] if shared else None,
        }
        for stage in ("project", "tail"):
            candidates = layer_names(inventory, index, stage)
            records[stage] = [
                binding_record(value, candidates[name])
                for name, value in programs[family, stage]
            ]
        if last_owners[kind] == index:
            producers[kind] = index
        layers.append(records)
    return layers


def build(args):
    from build_model_tvm import link_modules, verify_build_environment
    from qwen35_tvm_build import (
        export_tvmjs,
        verify_emsdk_checkout,
        verify_tvm_checkout,
    )

    verify_build_environment()
    if (
        version("peft") != "0.21.0"
        or version("strands-decider") != "0.2.1.dev7+g13c0514"
    ):
        raise ValueError(
            "Gemma v1 requires the pinned PEFT and official pointer model."
        )
    repo = SCRIPTS.parents[1]

    def git(*arguments):
        return check_output(["git", "-C", str(repo), *arguments], text=True).strip()

    code = {
        "commit": git("rev-parse", "HEAD"),
        "clean": not git("status", "--porcelain"),
    }
    pinned_tvm = verify_tvm_checkout(args.tvm_source)
    pinned_emsdk = verify_emsdk_checkout(args.emsdk)
    raw, inventory = read_pinned_inventory(args.inventory)
    source = CausalLayerSource(args.base, args.checkpoint)
    with ExitStack() as stack:
        handles = {
            "base": stack.enter_context((args.base / "model.safetensors").open("rb")),
            "adapter": stack.enter_context(
                (args.checkpoint / "lora/adapter_model.safetensors").open("rb")
            ),
        }
        modules, params, rotary = collect_families(source, inventory, handles)
        embedding_graphs, bindings = build_embedding_endpoints(
            source, inventory, handles
        )
        modules.extend(embedding_graphs)
    layers = layer_records(source.config, params, inventory)
    modules.append(
        (
            "pointer",
            import_pointer_head(
                load_pointer_head(args.checkpoint / "head.safetensors")
            ),
        )
    )
    graph = link_modules(modules)
    host = {"kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"}
    if args.cpu_wasm:
        host["mattr"] = ["+simd128"]
    target = (
        tvm.target.Target(host)
        if args.cpu_wasm
        else tvm.target.Target("webgpu", host=host)
    )
    lower = relax.get_default_pipeline(target)
    reordered = []

    @tvm.transform.module_pass(opt_level=0)
    def pipeline(module, _context):
        lowered = lower(module)
        normalized = canonical_stores(lowered)
        reordered.extend(
            name.name_hint
            for name, function in lowered.functions.items()
            if isinstance(function, tvm.tirx.PrimFunc)
            and not tvm_ffi.structural_equal(function, normalized[name])
        )
        if args.cpu_wasm:
            return normalized
        return verify_and_guard_grids(normalized, source.config.max_position_embeddings)

    with tvm.transform.PassContext(disabled_pass=["StaticPlanBlockMemory", "FuseOps"]):
        executable = relax.build(graph, target=target, relax_pipeline=pipeline)
    args.output.mkdir()
    with TemporaryDirectory(dir=args.output) as staging:
        wasm = Path(staging) / "graph.wasm"
        compiler = export_tvmjs(executable, wasm, args.emsdk, args.tvm_source)
        verify_tvm_checkout(args.tvm_source)
        if code != {
            "commit": git("rev-parse", "HEAD"),
            "clean": not git("status", "--porcelain"),
        }:
            raise ValueError("The conversion source changed during compilation")
        manifest = {
            "format": "web-decision-gemma4-e2b-v1-graph-v1",
            "tvmCommit": pinned_tvm,
            "source": code,
            "emsdkCommit": pinned_emsdk,
            "emscriptenVersion": compiler,
            "archiveApproved": False,
            "target": "cpu-wasm" if args.cpu_wasm else "webgpu-wasm",
            "maxLength": int(source.config.max_position_embeddings),
            "inventorySha256": hashlib.sha256(raw).hexdigest(),
            "graphSha256": hashlib.sha256(wasm.read_bytes()).hexdigest(),
            "entryPoints": [name + "_main" for name, _ in modules],
            "reorderedShapeFunctions": sorted(reordered),
            "layers": layers,
            **bindings,
            "rope": {
                kind: getattr(rotary, kind + "_inv_freq").tolist()
                for kind in set(source.config.layer_types)
            },
        }
        (Path(staging) / "graph.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
        )
        for name in ("graph.wasm", "graph.json"):
            (Path(staging) / name).replace(args.output / name)
    print(f"Built {len(layers)} E2B v1 layers using {len(modules)} native entry points")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "checkpoint", "inventory", "tvm-source", "emsdk", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--cpu-wasm",
        action="store_true",
        help="Build the same ABI for independent CPU checks",
    )
    build(parser.parse_args())
