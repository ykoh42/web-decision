#!/usr/bin/env python3
"""Prototype the pinned Kanana text branch through torch.export and TVM Relax."""

import argparse
import hashlib
import json
import math
import os
import re
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

try:
    from build_evidence import add_tvm_build_evidence
    from pinned_model_files import verify_pinned_model_files
except ModuleNotFoundError:
    from scripts.kanana.build_evidence import add_tvm_build_evidence
    from scripts.kanana.pinned_model_files import verify_pinned_model_files

KANANA_MODEL_MAX_LENGTH = 20_480
KANANA_VOCAB_SIZE = 128_256


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def align_parameter_storage(torch, module):
    for child in module.modules():
        for name, parameter in child.named_parameters(recurse=False):
            aligned = parameter.detach().clone(memory_format=torch.contiguous_format)
            setattr(
                child,
                name,
                torch.nn.Parameter(aligned, requires_grad=parameter.requires_grad),
            )


def _mrope_axis_for_channel(channel, sections):
    if channel < sections[1] * 3 and channel % 3 == 1:
        return 1
    if channel < sections[2] * 3 and channel % 3 == 2:
        return 2
    return 0


def _mrope_channel_layout(sections, half_dimension):
    axes = tuple(
        _mrope_axis_for_channel(channel, sections) for channel in range(half_dimension)
    )
    channels = tuple(
        tuple(channel for channel, axis in enumerate(axes) if axis == axis_index)
        for axis_index in range(3)
    )
    return axes, channels


def _select_mrope_frequencies(torch, freq, channels_by_axis):
    return tuple(
        torch.index_select(
            freq[axis_index],
            -1,
            torch.tensor(channels, dtype=torch.int64, device=freq.device),
        )
        for axis_index, channels in enumerate(channels_by_axis)
    )


def _has_interleaved_mrope_prefix(axis_by_channel, triplet_count):
    return axis_by_channel[: triplet_count * 3] == (0, 1, 2) * triplet_count


def _interleave_mrope_tail(torch, selected, axis_by_channel, triplet_count):
    interleaved = torch.stack(
        tuple(axis[..., :triplet_count] for axis in selected), dim=-1
    ).flatten(start_dim=-2)
    offsets = [triplet_count, triplet_count, triplet_count]
    tail = []
    for axis_index in axis_by_channel[triplet_count * 3 :]:
        offset = offsets[axis_index]
        tail.append(selected[axis_index][..., offset : offset + 1])
        offsets[axis_index] += 1
    if not tail:
        return interleaved
    return torch.cat((interleaved, torch.cat(tail, dim=-1)), dim=-1)


def _gather_mrope_channels(torch, freq, axis_by_channel):
    pieces = [
        freq[axis_index : axis_index + 1, :, :, channel : channel + 1]
        for channel, axis_index in enumerate(axis_by_channel)
    ]
    return torch.cat(pieces, dim=-1).squeeze(0)


def _make_mrope_recomposition(torch, axis_by_channel, channels_by_axis):
    def recompose(freq):
        selected = _select_mrope_frequencies(torch, freq, channels_by_axis)
        triplet_count = min(len(channels) for channels in channels_by_axis)
        if _has_interleaved_mrope_prefix(axis_by_channel, triplet_count):
            frequencies = _interleave_mrope_tail(
                torch, selected, axis_by_channel, triplet_count
            )
        else:
            frequencies = _gather_mrope_channels(torch, freq, axis_by_channel)
        return torch.cat((frequencies, frequencies), dim=-1)

    return recompose


def install_static_mrope_recomposition(torch, module):
    patched = 0
    for rotary in module.modules():
        sections = getattr(rotary, "mrope_section", None)
        inverse_frequencies = getattr(rotary, "inv_freq", None)
        if sections is None or inverse_frequencies is None:
            continue
        axes, channels = _mrope_channel_layout(
            sections, int(inverse_frequencies.numel())
        )
        rotary.recomposition_frequencies = _make_mrope_recomposition(
            torch, axes, channels
        )
        patched += 1

    if patched != 1:
        raise RuntimeError(f"Expected one text M-RoPE module, patched {patched}.")
    return patched


def _normalize_webgpu_constant(relax, np, constant, counts):
    if str(constant.ty.dtype) != "int64":
        return None
    values = constant.value.numpy()
    bounds = np.iinfo(np.int32)
    if values.size and (values.min() < bounds.min or values.max() > bounds.max):
        raise RuntimeError(
            "The Relax graph contains an int64 constant outside the int32 index range."
        )
    counts["constants"] += 1
    return relax.const(values.astype(np.int32))


def _normalize_webgpu_slice(relax, np, call, counts):
    args = [list(argument.fields) for argument in call.args[1:]]
    int64_max = np.iinfo(np.int64).max
    ends = [
        np.iinfo(np.int32).max if int(value) == int64_max else value
        for value in args[2]
    ]
    counts["sliceEndSentinels"] += sum(int(value) == int64_max for value in args[2])
    if args[2] == ends:
        return call
    return relax.op.strided_slice(
        call.args[0],
        args[0],
        args[1],
        ends,
        args[3],
        assume_inbound=call.attrs.assume_inbound,
    )


def _normalize_webgpu_call(relax, np, call, op_name, counts):
    dtype = str(call.attrs.dtype) if hasattr(call.attrs, "dtype") else ""
    if op_name == "relax.arange" and dtype == "int64":
        counts["aranges"] += 1
        return relax.op.arange(*call.args, dtype="int32")
    if op_name == "relax.astype" and dtype == "int64":
        counts["casts"] += 1
        return relax.op.astype(call.args[0], "int32")
    if op_name == "relax.astype" and dtype == "int8":
        source_type = getattr(call.args[0], "ty", None)
        if str(getattr(source_type, "dtype", "")) == "bool":
            counts["booleanMaskCasts"] += 1
            return relax.op.astype(call.args[0], "int32")
    if op_name == "relax.strided_slice":
        return _normalize_webgpu_slice(relax, np, call, counts)
    return call


def normalize_webgpu_dtypes(tvm, np, module, max_sequence_length):
    if max_sequence_length > np.iinfo(np.int32).max:
        raise RuntimeError(
            "The pinned sequence bound exceeds the WebGPU int32 index range."
        )

    counts = {
        "aranges": 0,
        "casts": 0,
        "booleanMaskCasts": 0,
        "constants": 0,
        "sliceEndSentinels": 0,
    }
    relax = tvm.relax
    expr_functor = relax.expr_functor

    @expr_functor.mutator
    class IndexDtypeNormalizer(expr_functor.PyExprMutator):
        def visit_generic_const_(self, constant):
            normalized = _normalize_webgpu_constant(relax, np, constant, counts)
            return (
                normalized
                if normalized is not None
                else super().visit_generic_const_(constant)
            )

        def visit_call_(self, call):
            op_name = call.op.name if hasattr(call.op, "name") else ""
            normalized_call = super().visit_call_(call)
            return _normalize_webgpu_call(relax, np, normalized_call, op_name, counts)

    mutator = IndexDtypeNormalizer(module)
    for global_var, function in module.functions.items():
        if isinstance(function, relax.Function):
            module.update_func(global_var, mutator.visit_expr(function))

    script = module.script()
    if 'dtype="int64"' in script or str(np.iinfo(np.int64).max) in script:
        raise RuntimeError(
            "The WebGPU Relax graph still contains int64 tensor operations or slice sentinels."
        )
    counts["maxSequenceLength"] = max_sequence_length
    counts["graphHasInt64TensorTypes"] = False
    return module, counts


def _run_cpu_parity_case(
    tvm,
    torch,
    np,
    vm,
    tvm_parameters,
    parameter_names,
    wrapped,
    inputs,
    sequence_length,
    name,
    image_tokens,
    allow_parity_delta,
    cases,
):
    with torch.inference_mode():
        expected = wrapped(*inputs).detach().float().numpy()
    input_tensors = {
        parameter_name: tvm.runtime.tensor(
            value.detach().cpu().numpy(), device=tvm.cpu()
        )
        for parameter_name, value in zip(
            parameter_names[: len(inputs)], inputs, strict=True
        )
    }
    arguments = [
        input_tensors[name] if name in input_tensors else tvm_parameters[name]
        for name in parameter_names
    ]
    started = time.perf_counter()
    output = vm["main"](*arguments)
    if not callable(getattr(output, "numpy", None)):
        output = output[0]
    actual = np.asarray(output.numpy(), dtype=np.float32)
    delta = actual - expected
    max_abs_delta = float(np.max(np.abs(delta)))
    rmse = float(np.sqrt(np.mean(np.square(delta))))
    finite_output = (
        np.all(np.isfinite(actual))
        and math.isfinite(max_abs_delta)
        and math.isfinite(rmse)
    )
    parity_exceeded = max_abs_delta >= 0.03 or rmse >= 0.005
    if not finite_output or (parity_exceeded and not allow_parity_delta):
        raise RuntimeError(
            f"TVM CPU parity failed for {name} at length {sequence_length}: "
            f"{max_abs_delta=}, {rmse=}"
        )
    cases.append(
        {
            "name": name,
            "sequenceLength": sequence_length,
            "imageTokens": image_tokens,
            "shape": list(actual.shape),
            "inferenceSeconds": round(time.perf_counter() - started, 3),
            "maxAbsDelta": max_abs_delta,
            "rmse": rmse,
            "withinTolerance": not parity_exceeded,
            "inputIds": inputs[0][0].tolist(),
            "attentionMask": inputs[1][0].tolist(),
            "imageEmbeddings": inputs[2].reshape(-1).tolist()
            if len(inputs) == 5
            else [],
            "imagePositions": inputs[3].reshape(-1).tolist()
            if len(inputs) == 5
            else [],
            "positionIds": inputs[4].reshape(-1).tolist() if len(inputs) == 5 else [],
            "referenceValues": expected.reshape(-1).tolist(),
        }
    )


def _run_sequence_parity_cases(
    tvm,
    torch,
    np,
    vm,
    tvm_parameters,
    parameter_names,
    wrapped,
    multimodal_example,
    sequence_length,
    dynamic_sequence,
    allow_parity_delta,
    cases,
):
    vocab_size = wrapped.language_model.config.vocab_size
    input_ids = (
        (torch.arange(sequence_length, dtype=torch.int32, device="cpu") * 2 + 1)
        .remainder(vocab_size)
        .reshape(1, sequence_length)
    )
    attention_mask = torch.ones((1, sequence_length), dtype=torch.int32, device="cpu")
    if multimodal_example is None:
        _run_cpu_parity_case(
            tvm,
            torch,
            np,
            vm,
            tvm_parameters,
            parameter_names,
            wrapped,
            (input_ids, attention_mask),
            sequence_length,
            "text-only",
            0,
            allow_parity_delta,
            cases,
        )
        return

    if sequence_length == 8:
        _run_cpu_parity_case(
            tvm,
            torch,
            np,
            vm,
            tvm_parameters,
            parameter_names,
            wrapped,
            multimodal_example,
            sequence_length,
            "two-image-tokens",
            2,
            allow_parity_delta,
            cases,
        )
    elif not dynamic_sequence:
        raise RuntimeError(
            "the fixed multimodal graph only accepts its exported sequence length"
        )
    if dynamic_sequence:
        hidden_size = multimodal_example[2].shape[1]
        image_embeddings = torch.empty((0, hidden_size), dtype=torch.float32)
        image_positions = torch.empty((0,), dtype=torch.int32)
        linear_positions = torch.arange(sequence_length, dtype=torch.int32).view(
            1, 1, -1
        )
        position_ids = linear_positions.expand(4, 1, -1)
        inputs = (
            input_ids,
            attention_mask,
            image_embeddings,
            image_positions,
            position_ids,
        )
        _run_cpu_parity_case(
            tvm,
            torch,
            np,
            vm,
            tvm_parameters,
            parameter_names,
            wrapped,
            inputs,
            sequence_length,
            "no-images",
            0,
            allow_parity_delta,
            cases,
        )


def run_cpu_parity(
    tvm,
    torch,
    np,
    executable,
    parameters,
    parameter_names,
    wrapped,
    sequence_lengths,
    multimodal_example=None,
    dynamic_sequence=False,
    allow_parity_delta=False,
):
    vm = tvm.relax.VirtualMachine(executable, tvm.cpu())
    tvm_parameters = {
        name: tvm.runtime.from_dlpack(torch.utils.dlpack.to_dlpack(parameter.detach()))
        for name, parameter in parameters.items()
    }
    vocab_size = wrapped.language_model.config.vocab_size
    if not isinstance(vocab_size, int) or vocab_size < 1:
        raise RuntimeError(
            "The exported text branch must define a positive vocabulary size."
        )
    cases = []
    for sequence_length in sequence_lengths:
        _run_sequence_parity_cases(
            tvm,
            torch,
            np,
            vm,
            tvm_parameters,
            parameter_names,
            wrapped,
            multimodal_example,
            sequence_length,
            dynamic_sequence,
            allow_parity_delta,
            cases,
        )
    return cases


def _parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="cached pinned model directory; reads config and custom code only",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for export graph, Relax IR, and evidence",
    )
    parser.add_argument(
        "--full-width-one-layer",
        action="store_true",
        help="keep pinned text tensor dimensions and rope config, reducing only the transformer layer count to one",
    )
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--checkpoint-one-layer",
        action="store_true",
        help="load and hash the pinned checkpoint, then export its first text layer",
    )
    checkpoint_group.add_argument(
        "--checkpoint-full",
        action="store_true",
        help="load and hash the pinned checkpoint, then export all text layers",
    )
    parser.add_argument(
        "--runtime-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
        help="parameter dtype exposed as TVM graph inputs",
    )
    parser.add_argument(
        "--weight-source",
        choices=("asset", "safetensors"),
        default="asset",
        help="store graph weights in a separate asset or read the pinned checkpoint by verified ranges",
    )
    parser.add_argument(
        "--dynamic-sequence",
        action="store_true",
        help="export one shared sequence dimension through the pinned model limit",
    )
    parser.add_argument(
        "--multimodal-inputs",
        action="store_true",
        help="accept vision features, image token positions, and four-axis M-RoPE positions",
    )
    parser.add_argument(
        "--sequence-sweep",
        action="store_true",
        help="run direct TVM CPU parity at lengths 1, 8, 9, 33, 64, and 128",
    )
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        help="run direct TVM CPU parity at these sequence lengths",
    )
    parser.add_argument(
        "--continue-after-cpu-parity-delta",
        action="store_true",
        help="record CPU output deltas and continue to WebGPU build for diagnosis",
    )
    parser.add_argument(
        "--target",
        choices=("llvm", "webgpu"),
        default="llvm",
        help="compile the direct Relax graph for CPU validation or WebGPU/wasm32",
    )
    args = parser.parse_args()
    _validate_arguments(args)
    return args


def _validate_arguments(args):
    if args.checkpoint_one_layer and not args.full_width_one_layer:
        raise SystemExit("--checkpoint-one-layer requires --full-width-one-layer")
    if args.checkpoint_full and args.full_width_one_layer:
        raise SystemExit(
            "--checkpoint-full cannot be combined with --full-width-one-layer"
        )
    if args.weight_source == "safetensors" and not args.checkpoint_full:
        raise SystemExit("--weight-source safetensors requires --checkpoint-full")
    if args.sequence_sweep and not args.dynamic_sequence:
        raise SystemExit("--sequence-sweep requires --dynamic-sequence")
    if args.sequence_lengths and not args.dynamic_sequence:
        raise SystemExit("--sequence-lengths requires --dynamic-sequence")
    if args.sequence_sweep and args.sequence_lengths:
        raise SystemExit("--sequence-sweep cannot be combined with --sequence-lengths")


def _load_dependencies():
    import numpy as np
    import torch
    import tvm
    from transformers import AutoConfig, AutoModel
    from tvm.relax.frontend.torch import from_exported_program

    return {
        "np": np,
        "torch": torch,
        "tvm": tvm,
        "AutoConfig": AutoConfig,
        "AutoModel": AutoModel,
        "fromExportedProgram": from_exported_program,
    }


def _pinned_text_config(config, text_config):
    return {
        "vocabSize": text_config.vocab_size,
        "hiddenSize": text_config.hidden_size,
        "intermediateSize": text_config.intermediate_size,
        "layers": text_config.num_hidden_layers,
        "attentionHeads": text_config.num_attention_heads,
        "keyValueHeads": text_config.num_key_value_heads,
        "headDim": text_config.head_dim,
        "maxPositionEmbeddings": text_config.max_position_embeddings,
        "padTokenId": text_config.pad_token_id,
        "ropeParameters": dict(text_config.rope_parameters),
        "embeddingDimension": config.embedding_dimension,
    }


def _apply_text_config_mode(args, config, text_config):
    if args.checkpoint_full:
        return
    if args.full_width_one_layer:
        if not args.checkpoint_one_layer:
            text_config.num_hidden_layers = 1
        return
    text_config.vocab_size = 64
    text_config.hidden_size = 32
    text_config.intermediate_size = 64
    text_config.num_hidden_layers = 1
    text_config.num_attention_heads = 2
    text_config.num_key_value_heads = 1
    text_config.head_dim = 16
    text_config.max_position_embeddings = 64
    text_config.pad_token_id = 0
    text_config.rope_parameters = {
        "rope_type": "yarn",
        "rope_theta": 10000,
        "factor": 2.0,
        "original_max_position_embeddings": 32,
        "mrope_interleaved": True,
        "mrope_section": [2, 2, 2],
    }
    config.embedding_dimension = 32
    config.max_length = 64


def _text_dimensions_match_pinned(args, config, text_config, pinned):
    if not (args.full_width_one_layer or args.checkpoint_full):
        return False
    pairs = (
        (text_config.vocab_size, "vocabSize"),
        (text_config.hidden_size, "hiddenSize"),
        (text_config.intermediate_size, "intermediateSize"),
        (text_config.num_attention_heads, "attentionHeads"),
        (text_config.num_key_value_heads, "keyValueHeads"),
        (text_config.head_dim, "headDim"),
        (text_config.max_position_embeddings, "maxPositionEmbeddings"),
        (text_config.pad_token_id, "padTokenId"),
        (dict(text_config.rope_parameters), "ropeParameters"),
        (config.embedding_dimension, "embeddingDimension"),
    )
    return all(current == pinned[name] for current, name in pairs)


def _shrink_vision_config(args, config, text_config, checkpoint_enabled):
    if checkpoint_enabled:
        return
    vision = config.vision_config
    vision.depth = 1
    vision.hidden_size = 32
    vision.intermediate_size = 64
    vision.num_heads = 2
    vision.patch_size = 2
    vision.temporal_patch_size = 1
    vision.spatial_merge_size = 1
    vision.out_hidden_size = text_config.hidden_size if args.multimodal_inputs else 32
    vision.num_position_embeddings = 16
    vision.in_channels = 3
    vision.gated_attention = True


def _load_model_configuration(args, config, checkpoint_enabled, config_started):
    text_config = config.text_config
    text_config._attn_implementation = "eager"
    pinned = _pinned_text_config(config, text_config)
    _apply_text_config_mode(args, config, text_config)
    full_width_match = _text_dimensions_match_pinned(args, config, text_config, pinned)
    max_sequence_length = min(
        KANANA_MODEL_MAX_LENGTH, text_config.max_position_embeddings
    )
    _shrink_vision_config(args, config, text_config, checkpoint_enabled)
    return (
        config,
        text_config,
        pinned,
        full_width_match,
        max_sequence_length,
        time.perf_counter() - config_started,
    )


def _create_initial_report(
    args,
    model_dir,
    code_hashes,
    config,
    text_config,
    pinned,
    full_width_match,
    max_sequence_length,
    dependencies,
    config_seconds,
):
    torch = dependencies["torch"]
    tvm = dependencies["tvm"]
    return {
        "modelId": "kakaocorp/kanana-2-v-decide-3b-2611",
        "modelRevision": "12820d654ed9789a0f7f48daeee1703c3483c464",
        "expectedCheckpointSha256": "528930fa261b31ed1f47eecbe7d08efd2e226207508124ae186e1a65dbbbda3b",
        "modelDirectory": model_dir.name,
        "customSourceSha256": code_hashes,
        "mode": "checkpoint_one_layer"
        if args.checkpoint_one_layer
        else "checkpoint_full"
        if args.checkpoint_full
        else "full_width_one_layer"
        if args.full_width_one_layer
        else "reduced",
        "fullWidthDimensionsUnchanged": full_width_match,
        "multimodalInputs": args.multimodal_inputs,
        "weightSource": "pinned-safetensors"
        if args.weight_source == "safetensors"
        else "external-asset",
        "onlyLayerCountReduced": args.full_width_one_layer
        and text_config.num_hidden_layers == 1
        and pinned["layers"] > 1
        and full_width_match,
        "pinnedTextConfig": pinned,
        "checkpointLoaded": False,
        "checkpointBytesRead": False,
        "onnxImportOrExportUsed": False,
        "device": "webgpu" if args.target == "webgpu" else "cpu",
        "target": "webgpu / wasm32" if args.target == "webgpu" else "llvm",
        "versions": {
            "torch": torch.__version__,
            "tvm": tvm.__version__,
            "transformers": __import__("transformers").__version__,
        },
        "effectiveTextConfig": {
            "vocabSize": text_config.vocab_size,
            "hiddenSize": text_config.hidden_size,
            "intermediateSize": text_config.intermediate_size,
            "layers": text_config.num_hidden_layers,
            "attentionHeads": text_config.num_attention_heads,
            "keyValueHeads": text_config.num_key_value_heads,
            "headDim": text_config.head_dim,
            "maxPositionEmbeddings": text_config.max_position_embeddings,
            "positionEmbeddingType": text_config.rope_parameters["rope_type"],
            "ropeParameters": dict(text_config.rope_parameters),
            "embeddingDimension": config.embedding_dimension,
            "attentionImplementation": text_config._attn_implementation,
        },
        "inputShape": [1, 8],
        "dynamicSequence": args.dynamic_sequence,
        "sequenceLengthBounds": [1, max_sequence_length]
        if args.dynamic_sequence
        else None,
        "cpuParityTolerance": {"maxAbsDelta": 0.03, "rmse": 0.005},
        "stageMetrics": {},
        "configLoadSeconds": round(config_seconds, 3),
        "peakRssBytes": peak_rss_bytes(),
    }


def _save_report(state):
    report = state["report"]
    report["peakRssBytes"] = peak_rss_bytes()
    report["elapsedSeconds"] = round(time.perf_counter() - state["runStarted"], 3)
    state["reportPath"].write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def _finish_stage(state, name, started):
    state["report"]["stageMetrics"][name] = {
        "seconds": round(time.perf_counter() - started, 3),
        "peakRssBytes": peak_rss_bytes(),
    }
    _save_report(state)


def _initialize_run(
    args,
    checkpoint_enabled,
    model_dir,
    code_hashes,
    output_dir,
    dependencies,
    run_started,
    config,
    config_started,
):
    (
        config,
        text_config,
        pinned,
        full_width_match,
        max_sequence_length,
        config_seconds,
    ) = _load_model_configuration(args, config, checkpoint_enabled, config_started)
    report = _create_initial_report(
        args,
        model_dir,
        code_hashes,
        config,
        text_config,
        pinned,
        full_width_match,
        max_sequence_length,
        dependencies,
        config_seconds,
    )
    state = {
        "args": args,
        "checkpointEnabled": checkpoint_enabled,
        "modelDir": model_dir,
        "outputDir": output_dir,
        "dependencies": dependencies,
        "config": config,
        "textConfig": text_config,
        "pinnedTextConfig": pinned,
        "maxSequenceLength": max_sequence_length,
        "report": report,
        "reportPath": output_dir / "prototype-report.json",
        "runStarted": run_started,
        "currentStage": "model-init",
    }
    _save_report(state)
    return state


def _load_text_model(state, stage_started):
    args = state["args"]
    deps = state["dependencies"]
    torch = deps["torch"]
    report = state["report"]
    model_dir = state["modelDir"]
    config = state["config"]
    model = None
    branch = None
    if state["checkpointEnabled"]:
        checkpoint_path = model_dir / "model.safetensors"
        checkpoint_size = checkpoint_path.stat().st_size
        checkpoint_hash = sha256_file(checkpoint_path)
        if (
            checkpoint_size != 7_330_945_212
            or checkpoint_hash != report["expectedCheckpointSha256"]
        ):
            raise RuntimeError(
                "local Safetensors checkpoint does not match the pinned Kanana artifact"
            )
        model = deps["AutoModel"].from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        branch = model.language_model
        if len(branch.layers) != state["pinnedTextConfig"]["layers"]:
            raise RuntimeError(
                "loaded Kanana text layer count differs from the pinned config"
            )
        if args.checkpoint_one_layer:
            branch.layers = torch.nn.ModuleList([branch.layers[0]])
            branch.config.num_hidden_layers = 1
            state["config"].text_config.num_hidden_layers = 1
        if not args.multimodal_inputs:
            model.visual = None
        _record_checkpoint_load(
            state, checkpoint_path, checkpoint_size, checkpoint_hash, branch
        )
    else:
        model = deps["AutoModel"].from_config(config, trust_remote_code=True)
        branch = model.language_model
    branch.config._attn_implementation = "eager"
    report["effectiveTextConfig"]["attentionImplementation"] = (
        branch.config._attn_implementation
    )
    model.eval()
    report["modelClass"] = type(model).__name__
    report["textBranchClass"] = type(branch).__name__
    report["parameterCount"] = sum(
        parameter.numel() for parameter in branch.parameters()
    )
    report["parameterBytes"] = sum(
        parameter.numel() * parameter.element_size()
        for parameter in branch.parameters()
    )
    _finish_stage(state, "modelInit", stage_started)
    return model, branch


def _record_checkpoint_load(
    state, checkpoint_path, checkpoint_size, checkpoint_hash, branch
):
    report = state["report"]
    args = state["args"]
    report["effectiveTextConfig"]["layers"] = len(branch.layers)
    report["onlyLayerCountReduced"] = (
        args.checkpoint_one_layer and report["fullWidthDimensionsUnchanged"]
    )
    report["checkpointLoaded"] = True
    report["checkpointBytesRead"] = True
    report["checkpointSizeBytes"] = checkpoint_size
    report["checkpointSha256"] = checkpoint_hash
    state["checkpointPath"] = checkpoint_path


def _create_text_branch(torch, language_model, multimodal_inputs):
    class TextBranch(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.language_model = model

        def finish(self, input_ids, inputs_embeds, attention_mask, position_ids):
            batch_size, sequence_length = input_ids.shape
            positions = torch.arange(
                sequence_length, dtype=torch.int64, device=input_ids.device
            )
            query_positions = positions.view(sequence_length, 1)
            key_positions = positions.view(1, sequence_length)
            mask_dtype = inputs_embeds.dtype
            mask_scale = -torch.finfo(mask_dtype).min
            causal_mask = (
                (query_positions - key_positions)
                .clamp(min=-1, max=0)
                .to(mask_dtype)
                .mul(mask_scale)
                .view(1, 1, sequence_length, sequence_length)
            )
            padding_mask = (
                (attention_mask.to(mask_dtype) - 1)
                .view(batch_size, 1, 1, sequence_length)
                .mul(mask_scale)
            )
            attention_mask = torch.minimum(causal_mask, padding_mask)
            output = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            return output.last_hidden_state.float()

        if multimodal_inputs:

            def forward(
                self,
                input_ids,
                attention_mask,
                image_embeddings,
                image_positions,
                position_ids,
            ):
                inputs_embeds = self.language_model.embed_tokens(input_ids)
                hidden_size = inputs_embeds.shape[-1]
                scatter_positions = (
                    image_positions.to(torch.int64)
                    .view(1, -1, 1)
                    .expand(input_ids.shape[0], -1, hidden_size)
                )
                inputs_embeds = inputs_embeds.scatter(
                    1,
                    scatter_positions,
                    image_embeddings.to(inputs_embeds.dtype).unsqueeze(0),
                )
                return self.finish(
                    input_ids,
                    inputs_embeds,
                    attention_mask,
                    position_ids.to(torch.int64),
                )

        else:

            def forward(self, input_ids, attention_mask):
                inputs_embeds = self.language_model.embed_tokens(input_ids)
                positions = torch.arange(
                    input_ids.shape[1], dtype=torch.int64, device=input_ids.device
                )
                position_ids = positions.view(1, 1, -1).expand(
                    4, input_ids.shape[0], -1
                )
                return self.finish(
                    input_ids, inputs_embeds, attention_mask, position_ids
                )

    return TextBranch(language_model).eval()


def _prepare_multimodal_example(state, model):
    config = state["config"]
    torch = state["dependencies"]["torch"]
    if not state["checkpointEnabled"]:
        config.image_token_id = 63
    image_token_id = config.image_token_id
    image_grid = [1, 2, 4] if state["checkpointEnabled"] else [1, 1, 2]
    image_patch_count = math.prod(image_grid)
    input_ids = torch.tensor(
        [[1, image_token_id, image_token_id, 3, 5, 7, 9, 11]],
        dtype=torch.int32,
        device="cpu",
    )
    image_grid_thw = torch.tensor([image_grid], dtype=torch.int64, device="cpu")
    patch_values = (
        config.vision_config.in_channels
        * config.vision_config.temporal_patch_size
        * config.vision_config.patch_size**2
    )
    pixel_values = torch.randn(
        (image_patch_count, patch_values), dtype=torch.float32, device="cpu"
    )
    model.visual.config._attn_implementation = "eager"
    with torch.inference_mode():
        image_embeddings = (
            model.visual(pixel_values, image_grid_thw, return_dict=True)
            .pooler_output.detach()
            .float()
        )
    image_positions = torch.tensor([1, 2], dtype=torch.int32, device="cpu")
    mrope_positions, _ = model.get_rope_index(
        input_ids.to(torch.int64),
        mm_token_type_ids=(input_ids == image_token_id).int(),
        image_grid_thw=image_grid_thw,
        attention_mask=torch.ones_like(input_ids),
    )
    text_positions = torch.arange(input_ids.shape[1], dtype=torch.int64).view(1, 1, -1)
    position_ids = torch.cat([text_positions, mrope_positions], dim=0).to(torch.int32)
    example = (
        input_ids,
        state["attentionMask"],
        image_embeddings,
        image_positions,
        position_ids,
    )
    return example, pixel_values, image_grid_thw


def _prepare_text_example(state):
    torch = state["dependencies"]["torch"]
    input_ids = torch.tensor(
        [[1, 3, 5, 7, 9, 11, 13, 15]], dtype=torch.int32, device="cpu"
    )
    return (input_ids, state["attentionMask"]), None, None


def _initialize_model_phase(state):
    print("stage model-init start", flush=True)
    stage_started = time.perf_counter()
    model, branch = _load_text_model(state, stage_started)
    args = state["args"]
    torch = state["dependencies"]["torch"]
    wrapped = _create_text_branch(torch, branch, args.multimodal_inputs)
    if state["checkpointEnabled"] and args.runtime_dtype == "bfloat16":
        align_parameter_storage(torch, wrapped)
    state["model"] = model
    state["branch"] = branch
    state["wrapped"] = wrapped
    state["attentionMask"] = torch.ones((1, 8), dtype=torch.int32, device="cpu")
    if args.multimodal_inputs:
        example, pixel_values, image_grid = _prepare_multimodal_example(state, model)
    else:
        example, pixel_values, image_grid = _prepare_text_example(state)
    state["example"] = example
    state["pixelValues"] = pixel_values
    state["imageGridThw"] = image_grid
    state["inputIds"] = example[0]
    state["currentStage"] = "pytorch-eager"


def _run_protocol_output(state):
    args = state["args"]
    torch = state["dependencies"]["torch"]
    model = state["model"]
    input_ids, attention_mask = state["inputIds"], state["attentionMask"]
    with torch.inference_mode():
        if args.multimodal_inputs:
            output = model(
                input_ids=input_ids.to(torch.int64),
                attention_mask=attention_mask,
                pixel_values=state["pixelValues"],
                image_grid_thw=state["imageGridThw"],
                use_cache=False,
            )
        else:
            output = model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            )
    return output.last_hidden_state.float()


def _run_wrapped_output(state):
    torch = state["dependencies"]["torch"]
    with torch.inference_mode():
        return state["wrapped"](*state["example"])


def _write_pytorch_reference(state, runtime_values, runtime_dtype):
    args = state["args"]
    output_dir = state["outputDir"]
    example = state["example"]
    reference = {
        "modelId": state["report"]["modelId"],
        "modelRevision": state["report"]["modelRevision"],
        "mode": state["report"]["mode"],
        "inputIds": example[0][0].tolist(),
        "attentionMask": example[1][0].tolist(),
        "shape": list(state["torchOutput"].shape),
        "dtype": "float32",
        "parameterDtype": runtime_dtype,
        "values": runtime_values.reshape(-1).tolist(),
    }
    if args.multimodal_inputs:
        reference["imagePositions"] = example[3].tolist()
        reference["positionIds"] = example[4].tolist()
        reference["imageEmbeddings"] = example[2].tolist()
        reference["imageGridThw"] = state["imageGridThw"].tolist()
    reference_path = output_dir / "pytorch-reference.json"
    reference_path.write_text(
        json.dumps(reference, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    state["report"]["pytorchReferencePath"] = str(reference_path)


def _run_pytorch_reference_phase(state):
    torch = state["dependencies"]["torch"]
    np = state["dependencies"]["np"]
    wrapped = state["wrapped"]
    print("stage pytorch-eager start", flush=True)
    stage_started = time.perf_counter()
    protocol_output = _run_protocol_output(state)
    _run_wrapped_output(state)
    if state["args"].target == "webgpu":
        count = install_static_mrope_recomposition(torch, state["branch"])
        state["report"]["staticMropeRecompositionModules"] = count
    source_output = _run_wrapped_output(state)
    source_dtype = str(next(wrapped.parameters()).dtype).removeprefix("torch.")
    if state["args"].runtime_dtype == "float16":
        wrapped.to(dtype=torch.float16)
    runtime_dtype = str(next(wrapped.parameters()).dtype).removeprefix("torch.")
    torch_output = _run_wrapped_output(state)
    protocol_values = protocol_output.detach().float().numpy()
    source_values = source_output.detach().float().numpy()
    runtime_values = torch_output.detach().float().numpy()
    protocol_delta = np.abs(source_values - protocol_values)
    if not np.allclose(source_values, protocol_values, rtol=1e-6, atol=1e-6):
        raise RuntimeError(
            "The exported text wrapper differs from the pinned model protocol output."
        )
    report = state["report"]
    report["textProtocolMaxAbsDelta"] = float(np.max(protocol_delta))
    report["textProtocolRmse"] = float(
        np.sqrt(np.mean(np.square(source_values - protocol_values)))
    )
    report["sourceParameterDtype"] = source_dtype
    report["runtimeParameterDtype"] = runtime_dtype
    report["runtimeDtypeOutputMaxAbsDelta"] = float(
        np.max(np.abs(runtime_values - source_values))
    )
    report["runtimeDtypeOutputRmse"] = float(
        np.sqrt(np.mean(np.square(runtime_values - source_values)))
    )
    report["pytorchOutputShape"] = list(torch_output.shape)
    report["pytorchOutputDtype"] = str(torch_output.dtype)
    state["torchOutput"] = torch_output
    _write_pytorch_reference(state, runtime_values, runtime_dtype)
    report["pytorchEagerCpu"] = "passed"
    _finish_stage(state, "pytorchEager", stage_started)
    state["sourceParameterDtype"] = source_dtype


def _export_dynamic_options(args, torch, max_sequence_length):
    if not args.dynamic_sequence:
        return {}
    sequence_length = torch.export.Dim(
        "sequence_length", min=1, max=max_sequence_length
    )
    if args.multimodal_inputs:
        image_count = torch.export.Dim("image_count", min=0, max=max_sequence_length)
        dynamic_shapes = (
            {1: sequence_length},
            {1: sequence_length},
            {0: image_count},
            {0: image_count},
            {2: sequence_length},
        )
    else:
        dynamic_shapes = ({1: sequence_length}, {1: sequence_length})
    return {"dynamic_shapes": dynamic_shapes}


def _record_export_operators(state, exported_program):
    operators = {}
    for node in exported_program.graph_module.graph.nodes:
        if node.op == "call_function":
            key = str(node.target)
            operators[key] = operators.get(key, 0) + 1
    report = state["report"]
    report["torchExport"] = "passed"
    report["exportGraphNodeCount"] = len(
        list(exported_program.graph_module.graph.nodes)
    )
    report["exportOperators"] = operators
    report["exportOperatorCount"] = sum(operators.values())


def _torch_export_phase(state):
    torch = state["dependencies"]["torch"]
    args = state["args"]
    print("stage torch-export start", flush=True)
    started = time.perf_counter()
    export_inputs = state["example"]
    options = _export_dynamic_options(args, torch, state["maxSequenceLength"])
    exported = torch.export.export(state["wrapped"], export_inputs, **options)
    output_path = state["outputDir"] / "exported-program.py"
    output_path.write_text(exported.graph_module.code, encoding="utf-8")
    _record_export_operators(state, exported)
    _finish_stage(state, "torchExport", started)
    state["exportedProgram"] = exported
    state["currentStage"] = "relax-import"


def _read_exported_parameters(torch, exported):
    return {
        spec.arg.name: exported.state_dict[spec.target]
        for spec in exported.graph_signature.input_specs
        if spec.kind is torch.export.graph_signature.InputKind.PARAMETER
    }


def _validate_relax_parameters(relax_module, exported_parameters):
    parameter_names = [parameter.name for parameter in relax_module["main"].params]
    if set(exported_parameters).difference(parameter_names):
        raise RuntimeError("TVM main omitted exported model parameters.")
    if "input_ids" not in parameter_names or "attention_mask" not in parameter_names:
        raise RuntimeError("TVM main omitted input_ids or attention_mask.")
    return parameter_names


def _write_external_weights(state, exported_program, exported_parameters):
    torch = state["dependencies"]["torch"]
    args = state["args"]
    output_dir = state["outputDir"]
    source_dtype = state["sourceParameterDtype"]
    payload = None if args.weight_source == "safetensors" else bytearray()
    manifest = []
    for spec in exported_program.graph_signature.input_specs:
        if spec.kind is not torch.export.graph_signature.InputKind.PARAMETER:
            continue
        tensor = exported_program.state_dict[spec.target].detach().cpu().contiguous()
        record = {
            "name": f"model.{spec.target}",
            "parameterName": spec.arg.name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "sourceDtype": source_dtype,
            "length": tensor.numel() * tensor.element_size(),
        }
        if payload is not None:
            weight_bytes = tensor.view(torch.uint8).numpy().tobytes()
            if record["length"] != len(weight_bytes):
                raise RuntimeError(
                    f"external weight byte count differs for {record['name']}"
                )
            record["assetOffset"] = len(payload)
            record["assetLength"] = len(weight_bytes)
            record["assetSha256"] = hashlib.sha256(weight_bytes).hexdigest()
            payload.extend(weight_bytes)
        manifest.append(record)
    report = state["report"]
    if payload is not None:
        asset = output_dir / "external-weights.bin"
        asset.write_bytes(payload)
        report["externalWeightAssetPath"] = str(asset)
        report["externalWeightAssetBytes"] = len(payload)
        report["externalWeightAssetSha256"] = hashlib.sha256(payload).hexdigest()
    else:
        report["externalWeightAssetPath"] = None
        report["externalWeightAssetBytes"] = 0
    report["externalWeightManifest"] = manifest
    external = {
        "format": "kanana-tvm-external-weight-manifest-v1",
        "modelId": report["modelId"],
        "modelRevision": report["modelRevision"],
        "weights": report["externalWeightManifest"],
    }
    path = output_dir / "external-weight-manifest.json"
    path.write_text(json.dumps(external, indent=2) + "\n", encoding="utf-8")
    report["externalWeightManifestPath"] = str(path)
    report["externalWeightManifestSha256"] = sha256_file(path)


def _record_relax_operators(state, relax_script):
    operators = {}
    for name in re.findall(r"\bR\.((?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*)\(", relax_script):
        operators[name] = operators.get(name, 0) + 1
    state["report"]["relaxOperators"] = operators


def _relax_import_phase(state):
    args = state["args"]
    deps = state["dependencies"]
    tvm, np = deps["tvm"], deps["np"]
    exported = state["exportedProgram"]
    print("stage relax-import start", flush=True)
    started = time.perf_counter()
    module = deps["fromExportedProgram"](
        exported,
        keep_params_as_input=True,
        run_ep_decomposition=True,
        unwrap_unit_return_tuple=True,
    )
    if args.target == "webgpu":
        module, normalization = normalize_webgpu_dtypes(
            tvm, np, module, state["maxSequenceLength"]
        )
        state["report"]["webgpuDtypeNormalization"] = normalization
    script = module.script()
    (state["outputDir"] / "relax-module.py").write_text(script, encoding="utf-8")
    exported_parameters = _read_exported_parameters(deps["torch"], exported)
    parameter_names = _validate_relax_parameters(module, exported_parameters)
    report = state["report"]
    report["relaxImport"] = "passed"
    report["relaxFunctionNames"] = [
        global_var.name_hint for global_var in module.functions
    ]
    report["relaxMainParameterNames"] = parameter_names
    report["externalParameterCount"] = len(exported_parameters)
    report["externalParameterBytes"] = sum(
        tensor.numel() * tensor.element_size()
        for tensor in exported_parameters.values()
    )
    _write_external_weights(state, exported, exported_parameters)
    _record_relax_operators(state, script)
    _finish_stage(state, "relaxImport", started)
    state["relaxModule"] = module
    state["exportedParameters"] = exported_parameters
    state["parameterNames"] = parameter_names
    state["currentStage"] = "llvm-build"


def _sequence_lengths(args, max_sequence_length):
    values = args.sequence_lengths or (
        [length for length in (1, 8, 9, 33, 64, 128) if length <= max_sequence_length]
        if args.sequence_sweep
        else [8]
    )
    if len(set(values)) != len(values) or any(
        length < 1 or length > max_sequence_length for length in values
    ):
        raise SystemExit(
            f"sequence lengths must be unique values from 1 to {max_sequence_length}"
        )
    return values


def _sequence_case_summary(case):
    omitted = (
        "inputIds",
        "attentionMask",
        "imageEmbeddings",
        "imagePositions",
        "positionIds",
        "referenceValues",
    )
    return {key: value for key, value in case.items() if key not in omitted}


def _run_cpu_phase(state):
    tvm = state["dependencies"]["tvm"]
    print("stage llvm-build start", flush=True)
    started = time.perf_counter()
    executable = tvm.relax.build(state["relaxModule"], target="llvm")
    state["report"]["relaxLlvmBuild"] = "passed"
    _finish_stage(state, "llvmBuild", started)
    state["currentStage"] = "cpu-vm-run"
    print("stage cpu-vm-run start", flush=True)
    started = time.perf_counter()
    cases = run_cpu_parity(
        tvm,
        state["dependencies"]["torch"],
        state["dependencies"]["np"],
        executable,
        state["exportedParameters"],
        state["parameterNames"],
        state["wrapped"],
        _sequence_lengths(state["args"], state["maxSequenceLength"]),
        multimodal_example=state["example"]
        if state["args"].multimodal_inputs
        else None,
        dynamic_sequence=state["args"].dynamic_sequence,
        allow_parity_delta=state["args"].continue_after_cpu_parity_delta,
    )
    baseline = next(case for case in cases if case["sequenceLength"] == 8)
    case_path = state["outputDir"] / "sequence-parity-cases.json"
    case_path.write_text(
        json.dumps({"cases": cases}, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    report = state["report"]
    report["sequenceParityCasesPath"] = str(case_path)
    report["sequenceParityCasesSha256"] = sha256_file(case_path)
    report["sequenceCases"] = [_sequence_case_summary(case) for case in cases]
    report["relaxCpuRun"] = (
        "passed"
        if all(case["withinTolerance"] for case in cases)
        else "completed_with_parity_deltas"
    )
    report["tvmOutputShape"] = baseline["shape"]
    report["tvmOutputDtype"] = "float32"
    report["outputMaxAbsDelta"] = baseline["maxAbsDelta"]
    report["outputRmse"] = baseline["rmse"]
    _finish_stage(state, "cpuVmRun", started)
    state["executable"] = executable


def _webgpu_target(tvm):
    return tvm.target.Target(
        "webgpu",
        host={"kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"},
    )


def _export_wasm(state, executable):
    from tvm.contrib import tvmjs

    output_dir = state["outputDir"]
    wasm_path = output_dir / "kanana-text-model.wasm"
    tvm_home = Path(__file__).resolve().parents[2] / ".toolchains" / "apache-tvm-src"
    runtime_dir = tvm_home / "web" / "dist" / "wasm"
    repo_root = Path(__file__).resolve().parents[2]
    runtime_assets = [
        runtime_dir / "wasm_runtime.bc",
        runtime_dir / "tvmjs_support.bc",
        runtime_dir / "webgpu_runtime.bc",
    ]
    if any(not asset.is_file() for asset in runtime_assets):
        raise RuntimeError("The pinned TVM WebGPU/wasm32 runtime assets are not built.")
    previous_tvm_home = os.environ.get("TVM_HOME")
    try:
        os.environ["TVM_HOME"] = str(tvm_home)
        executable.export_library(
            str(wasm_path),
            fcompile=tvmjs.create_tvmjs_wasm,
            options=[f"-ffile-prefix-map={repo_root}=/src"],
        )
    finally:
        if previous_tvm_home is None:
            os.environ.pop("TVM_HOME", None)
        else:
            os.environ["TVM_HOME"] = previous_tvm_home
    report = state["report"]
    report["webgpuWasmPath"] = str(wasm_path)
    report["webgpuWasmBytes"] = wasm_path.stat().st_size
    report["webgpuWasmSha256"] = sha256_file(wasm_path)
    report["tvmWasmRuntimeAssets"] = [
        {
            "path": str(asset),
            "bytes": asset.stat().st_size,
            "sha256": sha256_file(asset),
        }
        for asset in runtime_assets
    ]
    report["relaxWebgpuWasmExport"] = "passed"


def _run_webgpu_phase(state):
    tvm = state["dependencies"]["tvm"]
    state["currentStage"] = "webgpu-build"
    print("stage webgpu-build start", flush=True)
    started = time.perf_counter()
    executable = tvm.relax.build(state["relaxModule"], target=_webgpu_target(tvm))
    state["report"]["relaxWebgpuBuild"] = "passed"
    _finish_stage(state, "webgpuBuild", started)
    state["currentStage"] = "webgpu-wasm-export"
    print("stage webgpu-wasm-export start", flush=True)
    started = time.perf_counter()
    _export_wasm(state, executable)
    _finish_stage(state, "webgpuWasmExport", started)


def _write_safetensors_manifest(state):
    checkpoint_path = state["checkpointPath"]
    manifest_path = state["outputDir"] / "range-manifest.json"
    subprocess.run(
        [
            "node",
            "--import",
            "tsx",
            str(Path(__file__).resolve().parent / "hash_safetensors_ranges.mts"),
            str(checkpoint_path),
            str(manifest_path),
            "--verify-fp16-rn",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
    )
    state["report"]["rangeManifestPath"] = str(manifest_path)
    state["report"]["rangeManifestSha256"] = sha256_file(manifest_path)
    state["report"]["weightSourceConversionVerification"] = "passed"


def _run_optional_stages(state):
    args = state["args"]
    if args.target == "webgpu":
        _run_webgpu_phase(state)
    if args.weight_source == "safetensors":
        state["currentStage"] = "safetensors-range-manifest"
        _write_safetensors_manifest(state)


def _capture_pipeline_error(state, error):
    report = state["report"]
    report["failedStage"] = state["currentStage"]
    report["errorType"] = type(error).__name__
    report["error"] = str(error)
    report["traceback"] = traceback.format_exc()


def _attach_build_evidence(state):
    report = state["report"]
    if report.get("relaxWebgpuWasmExport") != "passed":
        return
    wasm_path = Path(report["webgpuWasmPath"])
    state["report"] = add_tvm_build_evidence(
        report,
        model_id="kanana-2-v-decide-3b-2611",
        repository="kakaocorp/kanana-2-v-decide-3b-2611",
        revision=report["modelRevision"],
        checkpoint_sha256=report["expectedCheckpointSha256"],
        graph="text",
        wasm_path=wasm_path,
        build_root=state["outputDir"],
        tvm_version=report["versions"]["tvm"],
    )


def _print_final_report(state):
    report = state["report"]
    _save_report(state)
    output = {
        key: value
        for key, value in report.items()
        if key not in ("exportOperators", "externalWeightManifest", "traceback")
    }
    print(json.dumps(output, indent=2))
    if "error" in report:
        raise SystemExit(1)


def _run_pipeline(
    args,
    checkpoint_enabled,
    model_dir,
    code_hashes,
    output_dir,
    dependencies,
    run_started,
    config,
    config_started,
):
    state = _initialize_run(
        args,
        checkpoint_enabled,
        model_dir,
        code_hashes,
        output_dir,
        dependencies,
        run_started,
        config,
        config_started,
    )
    try:
        _initialize_model_phase(state)
        _run_pytorch_reference_phase(state)
        _torch_export_phase(state)
        _relax_import_phase(state)
        _run_cpu_phase(state)
        _run_optional_stages(state)
    except Exception as error:
        _capture_pipeline_error(state, error)
    _attach_build_evidence(state)
    _print_final_report(state)


def main():
    args = _parse_arguments()
    checkpoint_enabled = args.checkpoint_one_layer or args.checkpoint_full
    model_dir = args.model_dir.expanduser().resolve(strict=True)
    code_hashes = verify_pinned_model_files(model_dir)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dependencies = _load_dependencies()
    run_started = time.perf_counter()
    torch = dependencies["torch"]
    torch.manual_seed(20261109)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config_started = time.perf_counter()
    config = dependencies["AutoConfig"].from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )
    _run_pipeline(
        args,
        checkpoint_enabled,
        model_dir,
        code_hashes,
        output_dir,
        dependencies,
        run_started,
        config,
        config_started,
    )


if __name__ == "__main__":
    main()
