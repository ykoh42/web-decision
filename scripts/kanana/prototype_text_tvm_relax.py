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
import types
from pathlib import Path

try:
    from build_evidence import add_tvm_build_evidence
    from online_attention_tir import install_online_attention_tir
    from pinned_model_files import verify_pinned_model_files
except ModuleNotFoundError:
    from scripts.kanana.build_evidence import add_tvm_build_evidence
    from scripts.kanana.online_attention_tir import install_online_attention_tir
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


def install_float32_compute(torch, module):
    linear_layers = 0
    rms_norm_layers = 0

    def linear_forward(layer, inputs):
        bias = None if layer.bias is None else layer.bias.float()
        return torch.nn.functional.linear(inputs.float(), layer.weight.float(), bias)

    def rms_norm_forward(layer, hidden_states):
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + layer.variance_epsilon)
        return layer.weight.float() * normalized

    for child in module.modules():
        if isinstance(child, torch.nn.Linear):
            child.forward = types.MethodType(linear_forward, child)
            linear_layers += 1
        elif type(child).__name__ == "Qwen3VLTextRMSNorm":
            child.forward = types.MethodType(rms_norm_forward, child)
            rms_norm_layers += 1

    layer_count = len(module.layers)
    expected_linear_layers = 7 * layer_count
    expected_rms_norm_layers = 4 * layer_count + 1
    if (
        linear_layers != expected_linear_layers
        or rms_norm_layers != expected_rms_norm_layers
    ):
        raise RuntimeError(
            f"Kanana F32 compute expected {expected_linear_layers} text linear layers and "
            f"{expected_rms_norm_layers} text RMSNorm layers; "
            f"found {linear_layers} and {rms_norm_layers}."
        )

    return {"linearLayers": linear_layers, "rmsNormLayers": rms_norm_layers}


def install_torch_online_attention(torch, transformers_attention_functions):
    def kanana_online_attention(
        module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
    ):
        groups = module.num_key_value_groups
        if groups != 1:
            batch, key_heads, sequence_length, head_dim = key.shape
            key = key[:, :, None, :, :].expand(
                batch, key_heads, groups, sequence_length, head_dim
            )
            value = value[:, :, None, :, :].expand(
                batch, key_heads, groups, sequence_length, head_dim
            )
            key = key.reshape(batch, key_heads * groups, sequence_length, head_dim)
            value = value.reshape(batch, key_heads * groups, sequence_length, head_dim)
        compiling = torch.compiler.is_compiling()
        if compiling:
            output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=dropout,
                is_causal=True,
                scale=scaling,
            )
            return output.transpose(1, 2).contiguous(), None

        query_length, key_length = query.shape[-2], key.shape[-2]
        causal = torch.ones(
            (query_length, key_length), dtype=torch.bool, device=query.device
        ).tril()
        causal_bias = torch.where(
            causal.view(1, 1, query_length, key_length),
            torch.zeros((), dtype=query.dtype, device=query.device),
            torch.full((), -torch.inf, dtype=query.dtype, device=query.device),
        )
        attention_mask = (
            causal_bias if attention_mask is None else attention_mask + causal_bias
        )
        scores = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            scores = scores + attention_mask
        weights = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        weights = torch.nn.functional.dropout(
            weights, p=dropout, training=module.training
        )
        output = torch.matmul(weights, value)
        return output.transpose(1, 2).contiguous(), None

    transformers_attention_functions.register(
        "web_decision_online", kanana_online_attention
    )


def restore_kanana_causal_attention(tvm):
    relax = tvm.relax
    conversion = {"attentionOps": 0}

    def convert_attention(node, importer):
        if (
            len(node.args) < 6
            or node.args[3] is not None
            or node.args[4] != 0.0
            or node.args[5] is not True
        ):
            raise RuntimeError(
                "Kanana online attention requires unmasked causal SDPA with zero dropout."
            )
        query, key, value = (importer.env[node.args[index]] for index in range(3))
        query = importer.block_builder.emit(
            relax.op.permute_dims(query, axes=[0, 2, 1, 3])
        )
        key = importer.block_builder.emit(relax.op.permute_dims(key, axes=[0, 2, 1, 3]))
        value = importer.block_builder.emit(
            relax.op.permute_dims(value, axes=[0, 2, 1, 3])
        )
        query = importer.block_builder.emit(relax.op.astype(query, "float32"))
        key = importer.block_builder.emit(relax.op.astype(key, "float32"))
        value = importer.block_builder.emit(relax.op.astype(value, "float32"))
        mask_node = next(
            (
                graph_node
                for graph_node in node.graph.nodes
                if graph_node.op == "placeholder"
                and graph_node.name == "attention_mask"
            ),
            None,
        )
        if mask_node is None or mask_node not in importer.env:
            raise RuntimeError(
                "Kanana online attention cannot find the attention_mask graph input."
            )
        padding_mask = importer.block_builder.emit(
            relax.op.astype(importer.env[mask_node], "float32")
        )
        padding_mask = importer.block_builder.emit(
            relax.op.reshape(padding_mask, (1, 1, 1, -1))
        )
        padding_mask = importer.block_builder.emit(
            relax.op.multiply(
                relax.op.subtract(padding_mask, relax.const(1.0, dtype="float32")),
                relax.const(3.402823466e38, dtype="float32"),
            )
        )
        result = importer.block_builder.emit(
            relax.op.nn.attention(
                query,
                key,
                value,
                bias=padding_mask,
                scale=node.kwargs.get("scale"),
                causal_mask="TopLeft",
            )
        )
        conversion["attentionOps"] += 1
        return importer.block_builder.emit(
            relax.op.permute_dims(result, axes=[0, 2, 1, 3])
        )

    return convert_attention, conversion


def _mrope_axis_mapping(sections, half_dimension):
    axis_by_channel = []
    for channel in range(half_dimension):
        if channel < sections[1] * 3 and channel % 3 == 1:
            axis_by_channel.append(1)
        elif channel < sections[2] * 3 and channel % 3 == 2:
            axis_by_channel.append(2)
        else:
            axis_by_channel.append(0)

    axis_by_channel = tuple(axis_by_channel)
    channels_by_axis = tuple(
        tuple(
            channel
            for channel, axis in enumerate(axis_by_channel)
            if axis == axis_index
        )
        for axis_index in range(3)
    )
    return axis_by_channel, channels_by_axis


def _select_mrope_frequencies(torch, freq, channels_by_axis):
    return tuple(
        torch.index_select(
            freq[axis_index],
            -1,
            torch.tensor(channels, dtype=torch.int64, device=freq.device),
        )
        for axis_index, channels in enumerate(channels_by_axis)
    )


def _interleave_mrope_frequencies(
    torch, selected, axis_by_channel, channels_by_axis, triplet_count
):
    interleaved = torch.stack(
        tuple(axis[..., :triplet_count] for axis in selected), dim=-1
    ).flatten(start_dim=-2)
    channel_offsets = [triplet_count, triplet_count, triplet_count]
    tail = []
    for axis_index in axis_by_channel[triplet_count * 3 :]:
        offset = channel_offsets[axis_index]
        tail.append(selected[axis_index][..., offset : offset + 1])
        channel_offsets[axis_index] += 1
    if tail:
        return torch.cat((interleaved, torch.cat(tail, dim=-1)), dim=-1)
    return interleaved


def _ordered_mrope_frequencies(torch, freq, axis_by_channel):
    pieces = [
        freq[axis_index : axis_index + 1, :, :, channel : channel + 1]
        for channel, axis_index in enumerate(axis_by_channel)
    ]
    return torch.cat(pieces, dim=-1).squeeze(0)


def _recompose_mrope_frequencies(torch, freq, axis_by_channel, channels_by_axis):
    selected = _select_mrope_frequencies(torch, freq, channels_by_axis)
    triplet_count = min(len(channels) for channels in channels_by_axis)
    is_interleaved = axis_by_channel[: triplet_count * 3] == (0, 1, 2) * triplet_count
    if is_interleaved:
        frequencies = _interleave_mrope_frequencies(
            torch, selected, axis_by_channel, channels_by_axis, triplet_count
        )
    else:
        frequencies = _ordered_mrope_frequencies(torch, freq, axis_by_channel)
    return torch.cat((frequencies, frequencies), dim=-1)


def install_static_mrope_recomposition(torch, module):
    patched = 0
    for rotary in module.modules():
        sections = getattr(rotary, "mrope_section", None)
        inverse_frequencies = getattr(rotary, "inv_freq", None)
        if sections is None or inverse_frequencies is None:
            continue

        axis_by_channel, channels_by_axis = _mrope_axis_mapping(
            sections, int(inverse_frequencies.numel())
        )

        def recompose(
            freq, axis_by_channel=axis_by_channel, channels_by_axis=channels_by_axis
        ):
            return _recompose_mrope_frequencies(
                torch, freq, axis_by_channel, channels_by_axis
            )

        rotary.recomposition_frequencies = recompose
        patched += 1

    if patched != 1:
        raise RuntimeError(f"Expected one text M-RoPE module, patched {patched}.")
    return patched


def _normalize_int64_constant(relax, np, constant, counts):
    values = constant.value.numpy()
    if values.size and (
        values.min() < np.iinfo(np.int32).min or values.max() > np.iinfo(np.int32).max
    ):
        raise RuntimeError(
            "The Relax graph contains an int64 constant outside the int32 index range."
        )
    counts["constants"] += 1
    return relax.const(values.astype(np.int32))


def _normalize_index_cast(relax, np, call, op_name, counts):
    if op_name == "relax.arange":
        if str(call.attrs.dtype) == "int64":
            counts["aranges"] += 1
            return relax.op.arange(*call.args, dtype="int32")
        return call

    if str(call.attrs.dtype) == "int64":
        counts["casts"] += 1
        return relax.op.astype(call.args[0], "int32")
    if str(call.attrs.dtype) != "int8":
        return call

    source_type = getattr(call.args[0], "ty", None)
    source_dtype = str(getattr(source_type, "dtype", ""))
    if source_dtype == "bool":
        counts["booleanMaskCasts"] += 1
        return relax.op.astype(call.args[0], "int32")
    return call


def _normalize_slice_ends(relax, np, call, counts):
    args = [list(argument.fields) for argument in call.args[1:]]
    ends = [
        np.iinfo(np.int32).max if int(value) == np.iinfo(np.int64).max else value
        for value in args[2]
    ]
    counts["sliceEndSentinels"] += sum(
        int(before) == np.iinfo(np.int64).max for before in args[2]
    )
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


def _normalize_relax_call(relax, np, call, op_name, counts):
    if op_name in ("relax.arange", "relax.astype"):
        return _normalize_index_cast(relax, np, call, op_name, counts)
    if op_name == "relax.strided_slice":
        return _normalize_slice_ends(relax, np, call, counts)
    return call


def _validate_webgpu_dtypes(np, module, counts, max_sequence_length):
    script = module.script()
    if 'dtype="int64"' in script or str(np.iinfo(np.int64).max) in script:
        raise RuntimeError(
            "The WebGPU Relax graph still contains int64 tensor operations or slice sentinels."
        )
    counts["maxSequenceLength"] = max_sequence_length
    counts["graphHasInt64TensorTypes"] = False


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
            if str(constant.ty.dtype) != "int64":
                return super().visit_generic_const_(constant)
            return _normalize_int64_constant(relax, np, constant, counts)

        def visit_call_(self, call):
            op_name = call.op.name if hasattr(call.op, "name") else ""
            call = super().visit_call_(call)
            return _normalize_relax_call(relax, np, call, op_name, counts)

    mutator = IndexDtypeNormalizer(module)
    for global_var, function in module.functions.items():
        if isinstance(function, relax.Function):
            module.update_func(global_var, mutator.visit_expr(function))

    _validate_webgpu_dtypes(np, module, counts, max_sequence_length)
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


def _make_cpu_parity_inputs(torch, sequence_length, vocab_size):
    input_ids = (
        (torch.arange(sequence_length, dtype=torch.int32, device="cpu") * 2 + 1)
        .remainder(vocab_size)
        .reshape(1, sequence_length)
    )
    attention_mask = torch.ones((1, sequence_length), dtype=torch.int32, device="cpu")
    return input_ids, attention_mask


def _run_multimodal_parity_cases(
    tvm,
    torch,
    np,
    vm,
    tvm_parameters,
    parameter_names,
    wrapped,
    sequence_length,
    input_ids,
    attention_mask,
    multimodal_example,
    dynamic_sequence,
    allow_parity_delta,
    cases,
):
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
    cases = []
    vocab_size = wrapped.language_model.config.vocab_size
    if not isinstance(vocab_size, int) or vocab_size < 1:
        raise RuntimeError(
            "The exported text branch must define a positive vocabulary size."
        )

    for sequence_length in sequence_lengths:
        input_ids, attention_mask = _make_cpu_parity_inputs(
            torch, sequence_length, vocab_size
        )
        if multimodal_example is None:
            inputs = (input_ids, attention_mask)
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
                "text-only",
                0,
                allow_parity_delta,
                cases,
            )
            continue
        _run_multimodal_parity_cases(
            tvm,
            torch,
            np,
            vm,
            tvm_parameters,
            parameter_names,
            wrapped,
            sequence_length,
            input_ids,
            attention_mask,
            multimodal_example,
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
        "--checkpoint-prefix-layers",
        type=int,
        help="diagnose a prefix of the checkpoint text layers while preserving pinned tensor dimensions",
    )
    parser.add_argument(
        "--runtime-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
        help="parameter dtype exposed as TVM graph inputs",
    )
    parser.add_argument(
        "--float32-compute",
        action="store_true",
        help="keep Kanana text activations and linear/RMSNorm compute in float32 with float16 parameters",
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
        "--online-attention",
        action="store_true",
        help="use the model-local online-softmax attention path for WebGPU lowering",
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
    _validate_checkpoint_arguments(args)
    _validate_runtime_arguments(args)
    _validate_sequence_arguments(args)


def _validate_checkpoint_arguments(args):
    if args.checkpoint_one_layer and not args.full_width_one_layer:
        raise SystemExit("--checkpoint-one-layer requires --full-width-one-layer")
    if args.checkpoint_full and args.full_width_one_layer:
        raise SystemExit(
            "--checkpoint-full cannot be combined with --full-width-one-layer"
        )
    if args.checkpoint_prefix_layers is not None and not args.checkpoint_full:
        raise SystemExit("--checkpoint-prefix-layers requires --checkpoint-full")
    if args.checkpoint_prefix_layers is not None and args.checkpoint_one_layer:
        raise SystemExit(
            "--checkpoint-prefix-layers cannot be combined with --checkpoint-one-layer"
        )
    if args.weight_source == "safetensors" and not args.checkpoint_full:
        raise SystemExit("--weight-source safetensors requires --checkpoint-full")


def _validate_runtime_arguments(args):
    if args.online_attention and args.runtime_dtype != "float16":
        raise SystemExit(
            "--online-attention requires the WebGPU-supported float16 runtime dtype."
        )
    if args.float32_compute and args.runtime_dtype != "float16":
        raise SystemExit(
            "--float32-compute requires the float16 runtime parameter dtype."
        )
    if args.online_attention and not args.float32_compute:
        raise SystemExit(
            "--online-attention requires --float32-compute for Kanana checkpoint parity."
        )


def _validate_sequence_arguments(args):
    if args.sequence_sweep and not args.dynamic_sequence:
        raise SystemExit("--sequence-sweep requires --dynamic-sequence")
    if args.sequence_lengths and not args.dynamic_sequence:
        raise SystemExit("--sequence-lengths requires --dynamic-sequence")
    if args.sequence_sweep and args.sequence_lengths:
        raise SystemExit("--sequence-sweep cannot be combined with --sequence-lengths")


def _save_report(run):
    run.report["peakRssBytes"] = peak_rss_bytes()
    run.report["elapsedSeconds"] = round(time.perf_counter() - run.run_started, 3)
    run.report_path.write_text(
        json.dumps(run.report, indent=2) + "\n", encoding="utf-8"
    )


def _finish_stage(run, name, started):
    run.report["stageMetrics"][name] = {
        "seconds": round(time.perf_counter() - started, 3),
        "peakRssBytes": peak_rss_bytes(),
    }
    _save_report(run)


def _load_config(run):
    run.config_started = time.perf_counter()
    run.config = run.AutoConfig.from_pretrained(
        run.model_dir, trust_remote_code=True, local_files_only=True
    )
    run.text_config = run.config.text_config
    run.text_config._attn_implementation = (
        "web_decision_online" if run.args.online_attention else "eager"
    )
    run.pinned_text_config = {
        "vocabSize": run.text_config.vocab_size,
        "hiddenSize": run.text_config.hidden_size,
        "intermediateSize": run.text_config.intermediate_size,
        "layers": run.text_config.num_hidden_layers,
        "attentionHeads": run.text_config.num_attention_heads,
        "keyValueHeads": run.text_config.num_key_value_heads,
        "headDim": run.text_config.head_dim,
        "maxPositionEmbeddings": run.text_config.max_position_embeddings,
        "padTokenId": run.text_config.pad_token_id,
        "ropeParameters": dict(run.text_config.rope_parameters),
        "embeddingDimension": run.config.embedding_dimension,
    }
    _configure_text_dimensions(run)
    run.full_width_dimensions_unchanged = _full_width_dimensions_unchanged(run)
    run.max_sequence_length = min(
        KANANA_MODEL_MAX_LENGTH, run.text_config.max_position_embeddings
    )
    _configure_tiny_vision(run)


def _configure_text_dimensions(run):
    args = run.args
    if args.checkpoint_full:
        return
    if args.full_width_one_layer:
        # Preserve every text tensor dimension and pinned RoPE setting. Only remove
        # the remaining 31 transformer blocks to bound CPU memory and compile cost.
        if not args.checkpoint_one_layer:
            run.text_config.num_hidden_layers = 1
        return

    # Reduced mode checks frontend coverage cheaply without allocating a large embedding table.
    run.text_config.vocab_size = 64
    run.text_config.hidden_size = 32
    run.text_config.intermediate_size = 64
    run.text_config.num_hidden_layers = 1
    run.text_config.num_attention_heads = 2
    run.text_config.num_key_value_heads = 1
    run.text_config.head_dim = 16
    run.text_config.max_position_embeddings = 64
    run.text_config.pad_token_id = 0
    run.text_config.rope_parameters = {
        "rope_type": "yarn",
        "rope_theta": 10000,
        "factor": 2.0,
        "original_max_position_embeddings": 32,
        "mrope_interleaved": True,
        "mrope_section": [2, 2, 2],
    }
    run.config.embedding_dimension = 32
    run.config.max_length = 64


def _full_width_dimensions_unchanged(run):
    if not (run.args.full_width_one_layer or run.args.checkpoint_full):
        return False
    pairs = (
        (run.text_config.vocab_size, "vocabSize"),
        (run.text_config.hidden_size, "hiddenSize"),
        (run.text_config.intermediate_size, "intermediateSize"),
        (run.text_config.num_attention_heads, "attentionHeads"),
        (run.text_config.num_key_value_heads, "keyValueHeads"),
        (run.text_config.head_dim, "headDim"),
        (run.text_config.max_position_embeddings, "maxPositionEmbeddings"),
        (run.text_config.pad_token_id, "padTokenId"),
        (dict(run.text_config.rope_parameters), "ropeParameters"),
        (run.config.embedding_dimension, "embeddingDimension"),
    )
    return all(
        current == run.pinned_text_config[pinned_key] for current, pinned_key in pairs
    )


def _configure_tiny_vision(run):
    if run.checkpoint_enabled:
        return
    vision_config = run.config.vision_config
    vision_config.depth = 1
    vision_config.hidden_size = 32
    vision_config.intermediate_size = 64
    vision_config.num_heads = 2
    vision_config.patch_size = 2
    vision_config.temporal_patch_size = 1
    vision_config.spatial_merge_size = 1
    vision_config.out_hidden_size = (
        run.text_config.hidden_size if run.args.multimodal_inputs else 32
    )
    vision_config.num_position_embeddings = 16
    vision_config.in_channels = 3
    vision_config.gated_attention = True


def _build_initial_report(run):
    args = run.args
    text_config = run.text_config
    config = run.config
    run.report = {
        "modelId": "kakaocorp/kanana-2-v-decide-3b-2611",
        "modelRevision": "12820d654ed9789a0f7f48daeee1703c3483c464",
        "expectedCheckpointSha256": "528930fa261b31ed1f47eecbe7d08efd2e226207508124ae186e1a65dbbbda3b",
        "modelDirectory": run.model_dir.name,
        "customSourceSha256": run.code_hashes,
        "mode": _report_mode(args),
        "fullWidthDimensionsUnchanged": run.full_width_dimensions_unchanged,
        "multimodalInputs": args.multimodal_inputs,
        "weightSource": "pinned-safetensors"
        if args.weight_source == "safetensors"
        else "external-asset",
        "onlyLayerCountReduced": _only_layer_count_reduced(run),
        "pinnedTextConfig": run.pinned_text_config,
        "checkpointLoaded": False,
        "checkpointBytesRead": False,
        "onnxImportOrExportUsed": False,
        "device": "webgpu" if args.target == "webgpu" else "cpu",
        "target": "webgpu / wasm32" if args.target == "webgpu" else "llvm",
        "versions": {
            "torch": run.torch.__version__,
            "tvm": run.tvm.__version__,
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
        "sequenceLengthBounds": [1, run.max_sequence_length]
        if args.dynamic_sequence
        else None,
        "exportedSequenceLengthUpperBound": None
        if args.dynamic_sequence
        else run.max_sequence_length,
        "cpuParityTolerance": {"maxAbsDelta": 0.03, "rmse": 0.005},
        "stageMetrics": {},
        "configLoadSeconds": round(time.perf_counter() - run.config_started, 3),
        "peakRssBytes": peak_rss_bytes(),
    }


def _report_mode(args):
    return (
        "checkpoint_prefix_layers"
        if args.checkpoint_prefix_layers is not None
        else "checkpoint_one_layer"
        if args.checkpoint_one_layer
        else "checkpoint_full"
        if args.checkpoint_full
        else "full_width_one_layer"
        if args.full_width_one_layer
        else "reduced"
    )


def _only_layer_count_reduced(run):
    return (
        run.args.full_width_one_layer
        and run.text_config.num_hidden_layers == 1
        and run.pinned_text_config["layers"] > 1
        and run.full_width_dimensions_unchanged
    )


def _prepare_run(args):
    run = types.SimpleNamespace(args=args)
    run.checkpoint_enabled = args.checkpoint_one_layer or args.checkpoint_full
    run.model_dir = args.model_dir.expanduser().resolve(strict=True)
    run.code_hashes = verify_pinned_model_files(run.model_dir)
    run.output_dir = args.output_dir.expanduser().resolve()
    run.output_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    import tvm
    from transformers import AutoConfig, AutoModel
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from tvm.relax.frontend.torch import from_exported_program

    run.np = np
    run.torch = torch
    run.tvm = tvm
    run.AutoConfig = AutoConfig
    run.AutoModel = AutoModel
    run.attention_functions = ALL_ATTENTION_FUNCTIONS
    run.from_exported_program = from_exported_program
    if args.online_attention:
        install_torch_online_attention(torch, ALL_ATTENTION_FUNCTIONS)

    run.run_started = time.perf_counter()
    torch.manual_seed(20261109)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    _load_config(run)
    _build_initial_report(run)
    run.report_path = run.output_dir / "prototype-report.json"
    run.current_stage = "model-init"
    return run


def _trim_checkpoint_layers(run, branch):
    args = run.args
    torch = run.torch
    if args.checkpoint_prefix_layers is not None:
        if not 1 <= args.checkpoint_prefix_layers <= len(branch.layers):
            raise SystemExit(
                f"--checkpoint-prefix-layers must be from 1 to {len(branch.layers)}"
            )
        branch.layers = torch.nn.ModuleList(
            list(branch.layers)[: args.checkpoint_prefix_layers]
        )
        branch.config.num_hidden_layers = args.checkpoint_prefix_layers
        run.config.text_config.num_hidden_layers = args.checkpoint_prefix_layers
    if args.checkpoint_one_layer:
        branch.layers = torch.nn.ModuleList([branch.layers[0]])
        branch.config.num_hidden_layers = 1
        run.config.text_config.num_hidden_layers = 1
    if not args.multimodal_inputs:
        run.model.visual = None


def _load_checkpoint_model(run):
    checkpoint_path = run.model_dir / "model.safetensors"
    checkpoint_size = checkpoint_path.stat().st_size
    checkpoint_hash = sha256_file(checkpoint_path)
    if (
        checkpoint_size != 7_330_945_212
        or checkpoint_hash != run.report["expectedCheckpointSha256"]
    ):
        raise RuntimeError(
            "local Safetensors checkpoint does not match the pinned Kanana artifact"
        )
    model = run.AutoModel.from_pretrained(
        run.model_dir,
        trust_remote_code=True,
        local_files_only=True,
        dtype=run.torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    branch = model.language_model
    if len(branch.layers) != run.pinned_text_config["layers"]:
        raise RuntimeError(
            "loaded Kanana text layer count differs from the pinned config"
        )
    run.model = model
    _trim_checkpoint_layers(run, branch)
    run.report["effectiveTextConfig"]["layers"] = len(branch.layers)
    run.report["onlyLayerCountReduced"] = (
        run.args.checkpoint_one_layer or run.args.checkpoint_prefix_layers is not None
    ) and run.report["fullWidthDimensionsUnchanged"]
    run.report["checkpointLoaded"] = True
    run.report["checkpointBytesRead"] = True
    run.report["checkpointSizeBytes"] = checkpoint_size
    run.report["checkpointSha256"] = checkpoint_hash
    run.checkpoint_path = checkpoint_path
    return model, branch


def _load_model_stage(run):
    run.current_stage = "model-init"
    print("stage model-init start", flush=True)
    stage_started = time.perf_counter()
    if run.checkpoint_enabled:
        model, branch = _load_checkpoint_model(run)
    else:
        model = run.AutoModel.from_config(run.config, trust_remote_code=True)
        branch = model.language_model
        run.model = model
    branch.config._attn_implementation = (
        "web_decision_online" if run.args.online_attention else "eager"
    )
    run.report["effectiveTextConfig"]["attentionImplementation"] = (
        branch.config._attn_implementation
    )
    model.eval()
    run.branch = branch
    run.report["modelClass"] = type(model).__name__
    run.report["textBranchClass"] = type(branch).__name__
    run.report["parameterCount"] = sum(
        parameter.numel() for parameter in branch.parameters()
    )
    run.report["parameterBytes"] = sum(
        parameter.numel() * parameter.element_size()
        for parameter in branch.parameters()
    )
    _finish_stage(run, "modelInit", stage_started)


def _make_text_branch_class(torch, args):
    class TextBranch(torch.nn.Module):
        def __init__(self, language_model):
            super().__init__()
            self.language_model = language_model
            self.float32_compute = False

        def finish(self, input_ids, inputs_embeds, attention_mask, position_ids):
            batch_size, sequence_length = input_ids.shape
            mask_dtype = inputs_embeds.dtype
            mask_scale = -torch.finfo(mask_dtype).min
            padding_mask = (
                (attention_mask.to(mask_dtype) - 1)
                .view(batch_size, 1, 1, sequence_length)
                .mul(mask_scale)
            )
            if args.online_attention:
                attention_mask = padding_mask
            else:
                positions = torch.arange(
                    sequence_length, dtype=torch.int64, device=input_ids.device
                )
                query_positions = positions.view(sequence_length, 1)
                key_positions = positions.view(1, sequence_length)
                causal_mask = (
                    (query_positions - key_positions)
                    .clamp(min=-1, max=0)
                    .to(mask_dtype)
                    .mul(mask_scale)
                    .view(1, 1, sequence_length, sequence_length)
                )
                attention_mask = torch.minimum(causal_mask, padding_mask)
            output = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            return output.last_hidden_state.float()

        if args.multimodal_inputs:

            def forward(
                self,
                input_ids,
                attention_mask,
                image_embeddings,
                image_positions,
                position_ids,
            ):
                inputs_embeds = self.language_model.embed_tokens(input_ids)
                if self.float32_compute:
                    inputs_embeds = inputs_embeds.float()
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
                if self.float32_compute:
                    inputs_embeds = inputs_embeds.float()
                positions = torch.arange(
                    input_ids.shape[1], dtype=torch.int64, device=input_ids.device
                )
                position_ids = positions.view(1, 1, -1).expand(
                    4, input_ids.shape[0], -1
                )
                return self.finish(
                    input_ids, inputs_embeds, attention_mask, position_ids
                )

    return TextBranch


def _build_example_inputs(run):
    torch = run.torch
    args = run.args
    run.wrapped = _make_text_branch_class(torch, args)(run.branch).eval()
    if run.checkpoint_enabled and args.runtime_dtype == "bfloat16":
        align_parameter_storage(torch, run.wrapped)
    run.attention_mask = torch.ones((1, 8), dtype=torch.int32, device="cpu")
    if not args.multimodal_inputs:
        run.input_ids = torch.tensor(
            [[1, 3, 5, 7, 9, 11, 13, 15]], dtype=torch.int32, device="cpu"
        )
        run.multimodal_example = None
        return

    _build_multimodal_example(run)


def _build_multimodal_example(run):
    torch = run.torch
    if not run.checkpoint_enabled:
        run.config.image_token_id = 63
    image_token_id = run.config.image_token_id
    image_grid = [1, 2, 4] if run.checkpoint_enabled else [1, 1, 2]
    image_patch_count = math.prod(image_grid)
    run.input_ids = torch.tensor(
        [[1, image_token_id, image_token_id, 3, 5, 7, 9, 11]],
        dtype=torch.int32,
        device="cpu",
    )
    run.image_grid_thw = torch.tensor([image_grid], dtype=torch.int64, device="cpu")
    patch_values = (
        run.config.vision_config.in_channels
        * run.config.vision_config.temporal_patch_size
        * run.config.vision_config.patch_size**2
    )
    run.pixel_values = torch.randn(
        (image_patch_count, patch_values), dtype=torch.float32, device="cpu"
    )
    run.model.visual.config._attn_implementation = "eager"
    with torch.inference_mode():
        image_embeddings = run.model.visual(
            run.pixel_values, run.image_grid_thw, return_dict=True
        ).pooler_output
        run.image_embeddings = image_embeddings.detach().float()
    run.image_positions = torch.tensor([1, 2], dtype=torch.int32, device="cpu")
    mrope_positions, _ = run.model.get_rope_index(
        run.input_ids.to(torch.int64),
        mm_token_type_ids=(run.input_ids == image_token_id).int(),
        image_grid_thw=run.image_grid_thw,
        attention_mask=torch.ones_like(run.input_ids),
    )
    text_positions = torch.arange(run.input_ids.shape[1], dtype=torch.int64).view(
        1, 1, -1
    )
    run.position_ids = torch.cat([text_positions, mrope_positions], dim=0).to(
        torch.int32
    )
    run.multimodal_example = (
        run.input_ids,
        run.attention_mask,
        run.image_embeddings,
        run.image_positions,
        run.position_ids,
    )


def _run_model_protocol_reference(run):
    torch = run.torch
    if run.args.multimodal_inputs:
        return run.model(
            input_ids=run.input_ids.to(torch.int64),
            attention_mask=run.attention_mask,
            pixel_values=run.pixel_values,
            image_grid_thw=run.image_grid_thw,
            use_cache=False,
        ).last_hidden_state.float()
    return run.model(
        input_ids=run.input_ids,
        attention_mask=run.attention_mask,
        use_cache=False,
    ).last_hidden_state.float()


def _run_wrapped_reference(run):
    if run.args.multimodal_inputs:
        return run.wrapped(*run.multimodal_example)
    return run.wrapped(run.input_ids, run.attention_mask)


def _record_pytorch_metrics(run):
    np = run.np
    report = run.report
    protocol_values = run.protocol_output.detach().float().numpy()
    source_values = run.source_output.detach().float().numpy()
    runtime_values = run.torch_output.detach().float().numpy()
    report["textProtocolMaxAbsDelta"] = float(
        np.max(np.abs(source_values - protocol_values))
    )
    report["textProtocolRmse"] = float(
        np.sqrt(np.mean(np.square(source_values - protocol_values)))
    )
    if not np.allclose(source_values, protocol_values, rtol=1e-6, atol=1e-6):
        raise RuntimeError(
            "The exported text wrapper differs from the pinned model protocol output."
        )
    report["sourceParameterDtype"] = run.source_parameter_dtype
    report["runtimeParameterDtype"] = run.runtime_parameter_dtype
    report["computeDtype"] = (
        "float32" if run.args.float32_compute else run.runtime_parameter_dtype
    )
    report["runtimeDtypeOutputMaxAbsDelta"] = float(
        np.max(np.abs(runtime_values - source_values))
    )
    report["runtimeDtypeOutputRmse"] = float(
        np.sqrt(np.mean(np.square(runtime_values - source_values)))
    )
    report["pytorchOutputShape"] = list(run.torch_output.shape)
    report["pytorchOutputDtype"] = str(run.torch_output.dtype)
    run.runtime_output_values = runtime_values


def _write_pytorch_reference(run):
    pytorch_reference = {
        "modelId": run.report["modelId"],
        "modelRevision": run.report["modelRevision"],
        "mode": run.report["mode"],
        "inputIds": run.input_ids[0].tolist(),
        "attentionMask": run.attention_mask[0].tolist(),
        "shape": list(run.torch_output.shape),
        "dtype": "float32",
        "parameterDtype": run.runtime_parameter_dtype,
        "values": run.runtime_output_values.reshape(-1).tolist(),
    }
    if run.args.multimodal_inputs:
        pytorch_reference["imagePositions"] = run.image_positions.tolist()
        pytorch_reference["positionIds"] = run.position_ids.tolist()
        pytorch_reference["imageEmbeddings"] = run.image_embeddings.tolist()
        pytorch_reference["imageGridThw"] = run.image_grid_thw.tolist()
    reference_path = run.output_dir / "pytorch-reference.json"
    reference_path.write_text(
        json.dumps(pytorch_reference, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    run.report["pytorchReferencePath"] = str(reference_path)
    run.report["pytorchEagerCpu"] = "passed"


def _run_pytorch_stage(run):
    run.current_stage = "pytorch-eager"
    print("stage pytorch-eager start", flush=True)
    stage_started = time.perf_counter()
    run.online_implementation = run.branch.config._attn_implementation
    run.branch.config._attn_implementation = "eager"
    with run.torch.inference_mode():
        run.protocol_output = _run_model_protocol_reference(run)
    run.branch.config._attn_implementation = run.online_implementation
    if run.args.target == "webgpu":
        run.report["staticMropeRecompositionModules"] = (
            install_static_mrope_recomposition(run.torch, run.branch)
        )
    with run.torch.inference_mode():
        run.source_output = _run_wrapped_reference(run)
    run.source_parameter_dtype = str(next(run.wrapped.parameters()).dtype).removeprefix(
        "torch."
    )
    if run.args.runtime_dtype == "float16":
        run.wrapped.to(dtype=run.torch.float16)
    run.runtime_parameter_dtype = str(
        next(run.wrapped.parameters()).dtype
    ).removeprefix("torch.")
    if run.args.float32_compute:
        run.report["float32ComputeModules"] = install_float32_compute(
            run.torch, run.branch
        )
        run.wrapped.float32_compute = True
    with run.torch.inference_mode():
        run.torch_output = _run_wrapped_reference(run)
    _record_pytorch_metrics(run)
    _write_pytorch_reference(run)
    _finish_stage(run, "pytorchEager", stage_started)


def _make_export_options(run):
    export_options = {}
    export_inputs = (
        run.multimodal_example
        if run.args.multimodal_inputs
        else (run.input_ids, run.attention_mask)
    )
    if not run.args.dynamic_sequence:
        return export_inputs, export_options

    sequence_length = run.torch.export.Dim("sequence_length", min=1)
    if run.args.multimodal_inputs:
        image_count = run.torch.export.Dim("image_count", min=0)
        export_options["dynamic_shapes"] = (
            {1: sequence_length},
            {1: sequence_length},
            {0: image_count},
            {0: image_count},
            {2: sequence_length},
        )
    else:
        export_options["dynamic_shapes"] = (
            {1: sequence_length},
            {1: sequence_length},
        )
    return export_inputs, export_options


def _count_export_operators(nodes):
    operators = {}
    for node in nodes:
        if node.op == "call_function":
            key = str(node.target)
            operators[key] = operators.get(key, 0) + 1
    return operators


def _run_torch_export_stage(run):
    run.current_stage = "torch-export"
    print("stage torch-export start", flush=True)
    stage_started = time.perf_counter()
    export_inputs, export_options = _make_export_options(run)
    run.exported_program = run.torch.export.export(
        run.wrapped, export_inputs, **export_options
    )
    graph = run.exported_program.graph_module.graph
    (run.output_dir / "exported-program.py").write_text(
        run.exported_program.graph_module.code, encoding="utf-8"
    )
    operators = _count_export_operators(graph.nodes)
    run.report["torchExport"] = "passed"
    run.report["exportGraphNodeCount"] = len(list(graph.nodes))
    run.report["exportOperators"] = operators
    run.report["exportOperatorCount"] = sum(operators.values())
    _finish_stage(run, "torchExport", stage_started)


def _prepare_relax_import_program(run):
    custom_convert_map = None
    conversion = None
    program_for_import = run.exported_program
    if run.args.online_attention:
        attention_converter, conversion = restore_kanana_causal_attention(run.tvm)
        custom_convert_map = {
            "scaled_dot_product_attention.default": attention_converter,
        }
        decomposition_table = run.torch.export.default_decompositions()
        attention_decomposition = decomposition_table.pop(
            run.torch.ops.aten.scaled_dot_product_attention.default, None
        )
        program_for_import = run.exported_program.run_decompositions(
            decomp_table=decomposition_table
        )
        conversion["defaultAttentionDecompositionRemoved"] = (
            attention_decomposition is not None
        )
    run.online_attention_conversion = conversion
    return program_for_import, custom_convert_map


def _import_relax_module(run):
    program_for_import, custom_convert_map = _prepare_relax_import_program(run)
    run.relax_module = run.from_exported_program(
        program_for_import,
        keep_params_as_input=True,
        unwrap_unit_return_tuple=True,
        custom_convert_map=custom_convert_map,
        run_ep_decomposition=not run.args.online_attention,
    )
    if run.args.online_attention:
        run.report["onlineAttentionGraphRewrite"] = run.online_attention_conversion
    if run.args.target == "webgpu":
        run.relax_module, normalization = normalize_webgpu_dtypes(
            run.tvm, run.np, run.relax_module, run.max_sequence_length
        )
        run.report["webgpuDtypeNormalization"] = normalization


def _collect_exported_parameters(run):
    parameter_kind = run.torch.export.graph_signature.InputKind.PARAMETER
    return {
        spec.arg.name: run.exported_program.state_dict[spec.target]
        for spec in run.exported_program.graph_signature.input_specs
        if spec.kind is parameter_kind
    }


def _record_relax_signature(run):
    main_parameters = list(run.relax_module["main"].params)
    parameter_names = [parameter.name for parameter in main_parameters]
    exported_parameters = _collect_exported_parameters(run)
    if set(exported_parameters).difference(parameter_names):
        raise RuntimeError("TVM main omitted exported model parameters.")
    if "input_ids" not in parameter_names or "attention_mask" not in parameter_names:
        raise RuntimeError("TVM main omitted input_ids or attention_mask.")
    run.main_parameter_names = parameter_names
    run.exported_parameters = exported_parameters
    run.report["relaxImport"] = "passed"
    run.report["relaxFunctionNames"] = [
        global_var.name_hint for global_var in run.relax_module.functions
    ]
    run.report["relaxMainParameterNames"] = parameter_names
    run.report["externalParameterCount"] = len(exported_parameters)
    run.report["externalParameterBytes"] = sum(
        parameter.numel() * parameter.element_size()
        for parameter in exported_parameters.values()
    )


def _make_external_weight_record(run, spec, tensor, payload):
    record = {
        "name": f"model.{spec.target}",
        "parameterName": spec.arg.name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "sourceDtype": run.source_parameter_dtype,
        "length": tensor.numel() * tensor.element_size(),
    }
    if payload is not None:
        weight_bytes = tensor.view(run.torch.uint8).numpy().tobytes()
        if record["length"] != len(weight_bytes):
            raise RuntimeError(
                f"external weight byte count differs for {record['name']}"
            )
        record["assetOffset"] = len(payload)
        record["assetLength"] = len(weight_bytes)
        record["assetSha256"] = hashlib.sha256(weight_bytes).hexdigest()
        payload.extend(weight_bytes)
    return record


def _record_external_weights(run):
    payload = None if run.args.weight_source == "safetensors" else bytearray()
    manifest = []
    parameter_kind = run.torch.export.graph_signature.InputKind.PARAMETER
    for spec in run.exported_program.graph_signature.input_specs:
        if spec.kind is not parameter_kind:
            continue
        tensor = (
            run.exported_program.state_dict[spec.target].detach().cpu().contiguous()
        )
        manifest.append(_make_external_weight_record(run, spec, tensor, payload))

    if payload is not None:
        asset_path = run.output_dir / "external-weights.bin"
        asset_path.write_bytes(payload)
        run.report["externalWeightAssetPath"] = str(asset_path)
        run.report["externalWeightAssetBytes"] = len(payload)
        run.report["externalWeightAssetSha256"] = hashlib.sha256(payload).hexdigest()
    else:
        run.report["externalWeightAssetPath"] = None
        run.report["externalWeightAssetBytes"] = 0
    run.report["externalWeightManifest"] = manifest
    return manifest


def _write_external_weight_manifest(run, weights):
    external_manifest = {
        "format": "kanana-tvm-external-weight-manifest-v1",
        "modelId": run.report["modelId"],
        "modelRevision": run.report["modelRevision"],
        "weights": weights,
    }
    path = run.output_dir / "external-weight-manifest.json"
    path.write_text(json.dumps(external_manifest, indent=2) + "\n", encoding="utf-8")
    run.report["externalWeightManifestPath"] = str(path)
    run.report["externalWeightManifestSha256"] = sha256_file(path)


def _count_relax_operators(relax_script):
    operators = {}
    for operator in re.findall(
        r"\bR\.((?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*)\(", relax_script
    ):
        operators[operator] = operators.get(operator, 0) + 1
    return operators


def _run_relax_import_stage(run):
    run.current_stage = "relax-import"
    print("stage relax-import start", flush=True)
    stage_started = time.perf_counter()
    _import_relax_module(run)
    run.relax_script = run.relax_module.script()
    (run.output_dir / "relax-module.py").write_text(run.relax_script, encoding="utf-8")
    _record_relax_signature(run)
    weights = _record_external_weights(run)
    _write_external_weight_manifest(run, weights)
    run.report["relaxOperators"] = _count_relax_operators(run.relax_script)
    _finish_stage(run, "relaxImport", stage_started)


def _run_llvm_build_stage(run):
    run.current_stage = "llvm-build"
    print(f"stage {run.current_stage} start", flush=True)
    stage_started = time.perf_counter()
    run.executable = run.tvm.relax.build(run.relax_module, target="llvm")
    run.report["relaxLlvmBuild"] = "passed"
    _finish_stage(run, "llvmBuild", stage_started)


def _sequence_lengths(run):
    if run.args.sequence_lengths:
        return run.args.sequence_lengths
    if run.args.sequence_sweep:
        return [
            length
            for length in (1, 8, 9, 33, 64, 128)
            if length <= run.max_sequence_length
        ]
    return [8]


def _record_cpu_parity_report(run, sequence_cases):
    baseline = next(case for case in sequence_cases if case["sequenceLength"] == 8)
    case_path = run.output_dir / "sequence-parity-cases.json"
    case_path.write_text(
        json.dumps({"cases": sequence_cases}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    report = run.report
    report["sequenceParityCasesPath"] = str(case_path)
    report["sequenceParityCasesSha256"] = sha256_file(case_path)
    removed_fields = (
        "inputIds",
        "attentionMask",
        "imageEmbeddings",
        "imagePositions",
        "positionIds",
        "referenceValues",
    )
    report["sequenceCases"] = [
        {key: value for key, value in case.items() if key not in removed_fields}
        for case in sequence_cases
    ]
    report["relaxCpuRun"] = (
        "passed"
        if all(case["withinTolerance"] for case in sequence_cases)
        else "completed_with_parity_deltas"
    )
    report["tvmOutputShape"] = baseline["shape"]
    report["tvmOutputDtype"] = "float32"
    report["outputMaxAbsDelta"] = baseline["maxAbsDelta"]
    report["outputRmse"] = baseline["rmse"]


def _run_cpu_vm_stage(run):
    run.current_stage = "cpu-vm-run"
    print(f"stage {run.current_stage} start", flush=True)
    stage_started = time.perf_counter()
    sequence_lengths = _sequence_lengths(run)
    if len(set(sequence_lengths)) != len(sequence_lengths) or any(
        length < 1 or length > run.max_sequence_length for length in sequence_lengths
    ):
        raise SystemExit(
            f"sequence lengths must be unique values from 1 to {run.max_sequence_length}"
        )
    sequence_cases = run_cpu_parity(
        run.tvm,
        run.torch,
        run.np,
        run.executable,
        run.exported_parameters,
        run.main_parameter_names,
        run.wrapped,
        sequence_lengths,
        multimodal_example=run.multimodal_example,
        dynamic_sequence=run.args.dynamic_sequence,
        allow_parity_delta=run.args.continue_after_cpu_parity_delta,
    )
    _record_cpu_parity_report(run, sequence_cases)
    _finish_stage(run, "cpuVmRun", stage_started)


def _build_webgpu_target(run):
    return run.tvm.target.Target(
        "webgpu",
        host={"kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"},
    )


def _build_webgpu_module(run, target):
    run.current_stage = "webgpu-build"
    print(f"stage {run.current_stage} start", flush=True)
    stage_started = time.perf_counter()
    run.executable = run.tvm.relax.build(run.relax_module, target=target)
    run.report["relaxWebgpuBuild"] = "passed"
    _finish_stage(run, "webgpuBuild", stage_started)


def _wasm_runtime_assets(tvm_home):
    runtime_directory = tvm_home / "web" / "dist" / "wasm"
    return [
        runtime_directory / "wasm_runtime.bc",
        runtime_directory / "tvmjs_support.bc",
        runtime_directory / "webgpu_runtime.bc",
    ]


def _export_webgpu_wasm(run):
    run.current_stage = "webgpu-wasm-export"
    print(f"stage {run.current_stage} start", flush=True)
    stage_started = time.perf_counter()
    from tvm.contrib import tvmjs

    wasm_path = run.output_dir / "kanana-text-model.wasm"
    tvm_home = Path(__file__).resolve().parents[2] / ".toolchains" / "apache-tvm-src"
    repo_root = Path(__file__).resolve().parents[2]
    runtime_assets = _wasm_runtime_assets(tvm_home)
    if any(not asset.is_file() for asset in runtime_assets):
        raise RuntimeError("The pinned TVM WebGPU/wasm32 runtime assets are not built.")
    previous_tvm_home = os.environ.get("TVM_HOME")
    os.environ["TVM_HOME"] = str(tvm_home)
    try:
        run.executable.export_library(
            str(wasm_path),
            fcompile=tvmjs.create_tvmjs_wasm,
            options=[f"-ffile-prefix-map={repo_root}=/src"],
        )
    finally:
        if previous_tvm_home is None:
            os.environ.pop("TVM_HOME", None)
        else:
            os.environ["TVM_HOME"] = previous_tvm_home
    run.report["webgpuWasmPath"] = str(wasm_path)
    run.report["webgpuWasmBytes"] = wasm_path.stat().st_size
    run.report["webgpuWasmSha256"] = sha256_file(wasm_path)
    run.report["tvmWasmRuntimeAssets"] = [
        {
            "path": str(asset),
            "bytes": asset.stat().st_size,
            "sha256": sha256_file(asset),
        }
        for asset in runtime_assets
    ]
    run.report["relaxWebgpuWasmExport"] = "passed"
    _finish_stage(run, "webgpuWasmExport", stage_started)


def _run_webgpu_stage(run):
    if run.args.target != "webgpu":
        return
    target = _build_webgpu_target(run)
    if run.args.online_attention:
        run.report["webgpuAttentionLowering"] = install_online_attention_tir(
            run.tvm, run.relax_module, target
        )
        (run.output_dir / "relax-module.py").write_text(
            run.relax_module.script(), encoding="utf-8"
        )
        _save_report(run)
    _build_webgpu_module(run, target)
    _export_webgpu_wasm(run)


def _run_safetensors_manifest_stage(run):
    if run.args.weight_source != "safetensors":
        return
    run.current_stage = "safetensors-range-manifest"
    manifest_path = run.output_dir / "range-manifest.json"
    subprocess.run(
        [
            "node",
            "--import",
            "tsx",
            str(Path(__file__).resolve().parent / "hash_safetensors_ranges.mts"),
            str(run.checkpoint_path),
            str(manifest_path),
            "--verify-fp16-rn",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
    )
    run.report["rangeManifestPath"] = str(manifest_path)
    run.report["rangeManifestSha256"] = sha256_file(manifest_path)
    run.report["weightSourceConversionVerification"] = "passed"


def _record_pipeline_error(run, error):
    run.report["failedStage"] = run.current_stage
    run.report["errorType"] = type(error).__name__
    run.report["error"] = str(error)
    run.report["traceback"] = traceback.format_exc()


def _add_build_evidence(run):
    if run.report.get("relaxWebgpuWasmExport") != "passed":
        return
    wasm_path = Path(run.report["webgpuWasmPath"])
    run.report = add_tvm_build_evidence(
        run.report,
        model_id="kanana-2-v-decide-3b-2611",
        repository="kakaocorp/kanana-2-v-decide-3b-2611",
        revision=run.report["modelRevision"],
        checkpoint_sha256=run.report["expectedCheckpointSha256"],
        graph="text",
        wasm_path=wasm_path,
        build_root=run.output_dir,
        tvm_version=run.report["versions"]["tvm"],
    )


def _print_report(run):
    print(
        json.dumps(
            {
                key: value
                for key, value in run.report.items()
                if key not in ("exportOperators", "externalWeightManifest", "traceback")
            },
            indent=2,
        )
    )


def _finalize_run(run):
    _add_build_evidence(run)
    _save_report(run)
    _print_report(run)
    if "error" in run.report:
        raise SystemExit(1)


def _run_pipeline(run):
    _save_report(run)
    try:
        _load_model_stage(run)
        _build_example_inputs(run)
        _run_pytorch_stage(run)
        _run_torch_export_stage(run)
        _run_relax_import_stage(run)
        _run_llvm_build_stage(run)
        _run_cpu_vm_stage(run)
        _run_webgpu_stage(run)
        _run_safetensors_manifest_stage(run)
    except Exception as error:
        _record_pipeline_error(run, error)
    _finalize_run(run)


def main():
    run = _prepare_run(_parse_arguments())
    _run_pipeline(run)


if __name__ == "__main__":
    main()
