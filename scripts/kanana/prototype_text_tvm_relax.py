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
            setattr(child, name, torch.nn.Parameter(aligned, requires_grad=parameter.requires_grad))


def install_static_mrope_recomposition(torch, module):
    patched = 0
    for rotary in module.modules():
        sections = getattr(rotary, "mrope_section", None)
        inverse_frequencies = getattr(rotary, "inv_freq", None)
        if sections is None or inverse_frequencies is None:
            continue

        half_dimension = int(inverse_frequencies.numel())
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
            tuple(channel for channel, axis in enumerate(axis_by_channel) if axis == axis_index)
            for axis_index in range(3)
        )

        def recompose(freq, axis_by_channel=axis_by_channel, channels_by_axis=channels_by_axis):
            selected = tuple(
                torch.index_select(
                    freq[axis_index],
                    -1,
                    torch.tensor(channels, dtype=torch.int64, device=freq.device),
                )
                for axis_index, channels in enumerate(channels_by_axis)
            )

            triplet_count = min(len(channels) for channels in channels_by_axis)
            if axis_by_channel[: triplet_count * 3] == (0, 1, 2) * triplet_count:
                interleaved = torch.stack(
                    tuple(axis[..., :triplet_count] for axis in selected), dim=-1
                ).flatten(start_dim=-2)
                channel_offsets = [triplet_count, triplet_count, triplet_count]
                tail = []
                for axis_index in axis_by_channel[triplet_count * 3 :]:
                    offset = channel_offsets[axis_index]
                    tail.append(selected[axis_index][..., offset : offset + 1])
                    channel_offsets[axis_index] += 1
                frequencies = (
                    torch.cat((interleaved, torch.cat(tail, dim=-1)), dim=-1)
                    if tail
                    else interleaved
                )
            else:
                pieces = [
                    freq[axis_index : axis_index + 1, :, :, channel : channel + 1]
                    for channel, axis_index in enumerate(axis_by_channel)
                ]
                frequencies = torch.cat(pieces, dim=-1).squeeze(0)

            return torch.cat((frequencies, frequencies), dim=-1)

        rotary.recomposition_frequencies = recompose
        patched += 1

    if patched != 1:
        raise RuntimeError(f"Expected one text M-RoPE module, patched {patched}.")
    return patched


def normalize_webgpu_dtypes(tvm, np, module, max_sequence_length):
    if max_sequence_length > np.iinfo(np.int32).max:
        raise RuntimeError("The pinned sequence bound exceeds the WebGPU int32 index range.")

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
            values = constant.value.numpy()
            if values.size and (
                values.min() < np.iinfo(np.int32).min or values.max() > np.iinfo(np.int32).max
            ):
                raise RuntimeError("The Relax graph contains an int64 constant outside the int32 index range.")
            counts["constants"] += 1
            return relax.const(values.astype(np.int32))

        def visit_call_(self, call):
            op_name = call.op.name if hasattr(call.op, "name") else ""
            call = super().visit_call_(call)
            if op_name == "relax.arange" and str(call.attrs.dtype) == "int64":
                counts["aranges"] += 1
                return relax.op.arange(*call.args, dtype="int32")
            if op_name == "relax.astype" and str(call.attrs.dtype) == "int64":
                counts["casts"] += 1
                return relax.op.astype(call.args[0], "int32")
            if op_name == "relax.astype" and str(call.attrs.dtype) == "int8":
                source_type = getattr(call.args[0], "ty", None)
                source_dtype = str(getattr(source_type, "dtype", ""))
                if source_dtype == "bool":
                    counts["booleanMaskCasts"] += 1
                    return relax.op.astype(call.args[0], "int32")
            if op_name != "relax.strided_slice":
                return call

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

    mutator = IndexDtypeNormalizer(module)
    for global_var, function in module.functions.items():
        if isinstance(function, relax.Function):
            module.update_func(global_var, mutator.visit_expr(function))

    script = module.script()
    if 'dtype="int64"' in script or str(np.iinfo(np.int64).max) in script:
        raise RuntimeError("The WebGPU Relax graph still contains int64 tensor operations or slice sentinels.")
    counts["maxSequenceLength"] = max_sequence_length
    counts["graphHasInt64TensorTypes"] = False
    return module, counts


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
        raise RuntimeError("The exported text branch must define a positive vocabulary size.")

    def run_case(inputs, sequence_length, name, image_tokens=0):
        with torch.inference_mode():
            expected = wrapped(*inputs).detach().float().numpy()
        input_tensors = {
            parameter_name: tvm.runtime.tensor(value.detach().cpu().numpy(), device=tvm.cpu())
            for parameter_name, value in zip(parameter_names[:len(inputs)], inputs, strict=True)
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
        finite_output = np.all(np.isfinite(actual)) and math.isfinite(max_abs_delta) and math.isfinite(rmse)
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
                "imageEmbeddings": inputs[2].reshape(-1).tolist() if len(inputs) == 5 else [],
                "imagePositions": inputs[3].reshape(-1).tolist() if len(inputs) == 5 else [],
                "positionIds": inputs[4].reshape(-1).tolist() if len(inputs) == 5 else [],
                "referenceValues": expected.reshape(-1).tolist(),
            }
        )

    for sequence_length in sequence_lengths:
        input_ids = (
            (torch.arange(sequence_length, dtype=torch.int32, device="cpu") * 2 + 1)
            .remainder(vocab_size)
            .reshape(1, sequence_length)
        )
        attention_mask = torch.ones((1, sequence_length), dtype=torch.int32, device="cpu")
        if multimodal_example is None:
            run_case((input_ids, attention_mask), sequence_length, "text-only")
            continue

        if sequence_length == 8:
            run_case(multimodal_example, sequence_length, "two-image-tokens", image_tokens=2)
        elif not dynamic_sequence:
            raise RuntimeError("the fixed multimodal graph only accepts its exported sequence length")

        if dynamic_sequence:
            hidden_size = multimodal_example[2].shape[1]
            image_embeddings = torch.empty((0, hidden_size), dtype=torch.float32)
            image_positions = torch.empty((0,), dtype=torch.int32)
            linear_positions = torch.arange(sequence_length, dtype=torch.int32).view(1, 1, -1)
            position_ids = linear_positions.expand(4, 1, -1)
            run_case(
                (input_ids, attention_mask, image_embeddings, image_positions, position_ids),
                sequence_length,
                "no-images",
            )
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="cached pinned model directory; reads config and custom code only")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for export graph, Relax IR, and evidence")
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
    if args.checkpoint_one_layer and not args.full_width_one_layer:
        raise SystemExit("--checkpoint-one-layer requires --full-width-one-layer")
    if args.checkpoint_full and args.full_width_one_layer:
        raise SystemExit("--checkpoint-full cannot be combined with --full-width-one-layer")
    if args.weight_source == "safetensors" and not args.checkpoint_full:
        raise SystemExit("--weight-source safetensors requires --checkpoint-full")
    if args.sequence_sweep and not args.dynamic_sequence:
        raise SystemExit("--sequence-sweep requires --dynamic-sequence")
    if args.sequence_lengths and not args.dynamic_sequence:
        raise SystemExit("--sequence-lengths requires --dynamic-sequence")
    if args.sequence_sweep and args.sequence_lengths:
        raise SystemExit("--sequence-sweep cannot be combined with --sequence-lengths")
    checkpoint_enabled = args.checkpoint_one_layer or args.checkpoint_full
    model_dir = args.model_dir.expanduser().resolve(strict=True)
    code_hashes = verify_pinned_model_files(model_dir)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    import tvm
    from transformers import AutoConfig, AutoModel
    from tvm.relax.frontend.torch import from_exported_program

    run_started = time.perf_counter()
    torch.manual_seed(20261109)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    config_started = time.perf_counter()
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    text_config = config.text_config
    # Export the model's supported eager attention reference; TVM's current
    # WebGPU lowering of scaled_dot_product_attention emits an invalid WGSL i32 minimum.
    text_config._attn_implementation = "eager"
    pinned_text_config = {
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
    if args.checkpoint_full:
        pass
    elif args.full_width_one_layer:
        # Preserve every text tensor dimension and pinned RoPE setting. Only remove
        # the remaining 31 transformer blocks to bound CPU memory and compile cost.
        if not args.checkpoint_one_layer:
            text_config.num_hidden_layers = 1
    else:
        # Reduced mode checks frontend coverage cheaply without allocating a large embedding table.
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
    full_width_dimensions_unchanged = (args.full_width_one_layer or args.checkpoint_full) and all(
        current == pinned_text_config[pinned_key]
        for current, pinned_key in (
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
    )
    max_sequence_length = min(KANANA_MODEL_MAX_LENGTH, text_config.max_position_embeddings)

    # The top-level custom model constructs its actual pinned language_model and
    # vision child. The tiny vision config is instantiated but never called.
    if not checkpoint_enabled:
        config.vision_config.depth = 1
        config.vision_config.hidden_size = 32
        config.vision_config.intermediate_size = 64
        config.vision_config.num_heads = 2
        config.vision_config.patch_size = 2
        config.vision_config.temporal_patch_size = 1
        config.vision_config.spatial_merge_size = 1
        config.vision_config.out_hidden_size = text_config.hidden_size if args.multimodal_inputs else 32
        config.vision_config.num_position_embeddings = 16
        config.vision_config.in_channels = 3
        config.vision_config.gated_attention = True
    report = {
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
        "fullWidthDimensionsUnchanged": full_width_dimensions_unchanged,
        "multimodalInputs": args.multimodal_inputs,
        "weightSource": "pinned-safetensors" if args.weight_source == "safetensors" else "external-asset",
        "onlyLayerCountReduced": args.full_width_one_layer
        and text_config.num_hidden_layers == 1
        and pinned_text_config["layers"] > 1
        and full_width_dimensions_unchanged,
        "pinnedTextConfig": pinned_text_config,
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
        "sequenceLengthBounds": [1, max_sequence_length] if args.dynamic_sequence else None,
        "cpuParityTolerance": {"maxAbsDelta": 0.03, "rmse": 0.005},
        "stageMetrics": {},
        "configLoadSeconds": round(time.perf_counter() - config_started, 3),
        "peakRssBytes": peak_rss_bytes(),
    }

    report_path = output_dir / "prototype-report.json"
    current_stage = "model-init"

    def save_report():
        report["peakRssBytes"] = peak_rss_bytes()
        report["elapsedSeconds"] = round(time.perf_counter() - run_started, 3)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def finish_stage(name, started):
        report["stageMetrics"][name] = {
            "seconds": round(time.perf_counter() - started, 3),
            "peakRssBytes": peak_rss_bytes(),
        }
        save_report()

    save_report()
    try:
        print("stage model-init start", flush=True)
        stage_started = time.perf_counter()
        if checkpoint_enabled:
            checkpoint_path = model_dir / "model.safetensors"
            checkpoint_size = checkpoint_path.stat().st_size
            checkpoint_hash = sha256_file(checkpoint_path)
            if checkpoint_size != 7_330_945_212 or checkpoint_hash != report["expectedCheckpointSha256"]:
                raise RuntimeError("local Safetensors checkpoint does not match the pinned Kanana artifact")
            model = AutoModel.from_pretrained(
                model_dir,
                trust_remote_code=True,
                local_files_only=True,
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            branch = model.language_model
            if len(branch.layers) != pinned_text_config["layers"]:
                raise RuntimeError("loaded Kanana text layer count differs from the pinned config")
            if args.checkpoint_one_layer:
                branch.layers = torch.nn.ModuleList([branch.layers[0]])
                branch.config.num_hidden_layers = 1
                config.text_config.num_hidden_layers = 1
            if not args.multimodal_inputs:
                model.visual = None
            report["effectiveTextConfig"]["layers"] = len(branch.layers)
            report["onlyLayerCountReduced"] = args.checkpoint_one_layer and report["fullWidthDimensionsUnchanged"]
            report["checkpointLoaded"] = True
            report["checkpointBytesRead"] = True
            report["checkpointSizeBytes"] = checkpoint_size
            report["checkpointSha256"] = checkpoint_hash
        else:
            model = AutoModel.from_config(config, trust_remote_code=True)
            branch = model.language_model
        # from_pretrained reloads its own config unless explicitly passed one.
        # Set the actual loaded branch's backend before reference inference and export.
        branch.config._attn_implementation = "eager"
        report["effectiveTextConfig"]["attentionImplementation"] = branch.config._attn_implementation
        model.eval()
        report["modelClass"] = type(model).__name__
        report["textBranchClass"] = type(branch).__name__
        report["parameterCount"] = sum(parameter.numel() for parameter in branch.parameters())
        report["parameterBytes"] = sum(parameter.numel() * parameter.element_size() for parameter in branch.parameters())
        finish_stage("modelInit", stage_started)

        class TextBranch(torch.nn.Module):
            def __init__(self, language_model):
                super().__init__()
                self.language_model = language_model

            def finish(self, input_ids, inputs_embeds, attention_mask, position_ids):
                batch_size, sequence_length = input_ids.shape
                positions = torch.arange(sequence_length, dtype=torch.int64, device=input_ids.device)
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

            if args.multimodal_inputs:

                def forward(self, input_ids, attention_mask, image_embeddings, image_positions, position_ids):
                    inputs_embeds = self.language_model.embed_tokens(input_ids)
                    hidden_size = inputs_embeds.shape[-1]
                    scatter_positions = image_positions.to(torch.int64).view(1, -1, 1).expand(
                        input_ids.shape[0], -1, hidden_size
                    )
                    inputs_embeds = inputs_embeds.scatter(
                        1,
                        scatter_positions,
                        image_embeddings.to(inputs_embeds.dtype).unsqueeze(0),
                    )
                    return self.finish(input_ids, inputs_embeds, attention_mask, position_ids.to(torch.int64))

            else:

                def forward(self, input_ids, attention_mask):
                    inputs_embeds = self.language_model.embed_tokens(input_ids)
                    positions = torch.arange(input_ids.shape[1], dtype=torch.int64, device=input_ids.device)
                    position_ids = positions.view(1, 1, -1).expand(4, input_ids.shape[0], -1)
                    return self.finish(input_ids, inputs_embeds, attention_mask, position_ids)

        wrapped = TextBranch(branch).eval()
        if checkpoint_enabled and args.runtime_dtype == "bfloat16":
            align_parameter_storage(torch, wrapped)
        attention_mask = torch.ones((1, 8), dtype=torch.int32, device="cpu")
        if args.multimodal_inputs:
            if not checkpoint_enabled:
                config.image_token_id = 63
            image_token_id = config.image_token_id
            image_grid = [1, 2, 4] if checkpoint_enabled else [1, 1, 2]
            image_patch_count = math.prod(image_grid)
            input_ids = torch.tensor(
                [[1, image_token_id, image_token_id, 3, 5, 7, 9, 11]], dtype=torch.int32, device="cpu"
            )
            image_grid_thw = torch.tensor([image_grid], dtype=torch.int64, device="cpu")
            patch_values = (
                config.vision_config.in_channels
                * config.vision_config.temporal_patch_size
                * config.vision_config.patch_size**2
            )
            pixel_values = torch.randn((image_patch_count, patch_values), dtype=torch.float32, device="cpu")
            model.visual.config._attn_implementation = "eager"
            with torch.inference_mode():
                image_embeddings = model.visual(pixel_values, image_grid_thw, return_dict=True).pooler_output
                image_embeddings = image_embeddings.detach().float()
            image_positions = torch.tensor([1, 2], dtype=torch.int32, device="cpu")
            mrope_positions, _ = model.get_rope_index(
                input_ids.to(torch.int64),
                mm_token_type_ids=(input_ids == image_token_id).int(),
                image_grid_thw=image_grid_thw,
                attention_mask=torch.ones_like(input_ids),
            )
            text_positions = torch.arange(input_ids.shape[1], dtype=torch.int64).view(1, 1, -1)
            position_ids = torch.cat([text_positions, mrope_positions], dim=0).to(torch.int32)
            multimodal_example = (input_ids, attention_mask, image_embeddings, image_positions, position_ids)
        else:
            input_ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15]], dtype=torch.int32, device="cpu")
            multimodal_example = None
        current_stage = "pytorch-eager"
        print("stage pytorch-eager start", flush=True)
        stage_started = time.perf_counter()
        with torch.inference_mode():
            if args.multimodal_inputs:
                protocol_output = model(
                    input_ids=input_ids.to(torch.int64),
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    use_cache=False,
                ).last_hidden_state.float()
                source_output = wrapped(*multimodal_example)
            else:
                protocol_output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).last_hidden_state.float()
        if args.target == "webgpu":
            report["staticMropeRecompositionModules"] = install_static_mrope_recomposition(torch, branch)
        with torch.inference_mode():
            if args.multimodal_inputs:
                source_output = wrapped(*multimodal_example)
            else:
                source_output = wrapped(input_ids, attention_mask)
        source_parameter_dtype = str(next(wrapped.parameters()).dtype).removeprefix("torch.")
        if args.runtime_dtype == "float16":
            wrapped.to(dtype=torch.float16)
        runtime_parameter_dtype = str(next(wrapped.parameters()).dtype).removeprefix("torch.")
        with torch.inference_mode():
            torch_output = wrapped(*multimodal_example) if args.multimodal_inputs else wrapped(input_ids, attention_mask)
        protocol_output_values = protocol_output.detach().float().numpy()
        source_output_values = source_output.detach().float().numpy()
        runtime_output_values = torch_output.detach().float().numpy()
        report["textProtocolMaxAbsDelta"] = float(np.max(np.abs(source_output_values - protocol_output_values)))
        report["textProtocolRmse"] = float(np.sqrt(np.mean(np.square(source_output_values - protocol_output_values))))
        if not np.allclose(source_output_values, protocol_output_values, rtol=1e-6, atol=1e-6):
            raise RuntimeError("The exported text wrapper differs from the pinned model protocol output.")
        report["sourceParameterDtype"] = source_parameter_dtype
        report["runtimeParameterDtype"] = runtime_parameter_dtype
        report["runtimeDtypeOutputMaxAbsDelta"] = float(np.max(np.abs(runtime_output_values - source_output_values)))
        report["runtimeDtypeOutputRmse"] = float(np.sqrt(np.mean(np.square(runtime_output_values - source_output_values))))
        report["pytorchOutputShape"] = list(torch_output.shape)
        report["pytorchOutputDtype"] = str(torch_output.dtype)
        pytorch_reference = {
            "modelId": report["modelId"],
            "modelRevision": report["modelRevision"],
            "mode": report["mode"],
            "inputIds": input_ids[0].tolist(),
            "attentionMask": attention_mask[0].tolist(),
            "shape": list(torch_output.shape),
            "dtype": "float32",
            "parameterDtype": runtime_parameter_dtype,
            "values": runtime_output_values.reshape(-1).tolist(),
        }
        if args.multimodal_inputs:
            pytorch_reference["imagePositions"] = image_positions.tolist()
            pytorch_reference["positionIds"] = position_ids.tolist()
            pytorch_reference["imageEmbeddings"] = image_embeddings.tolist()
            pytorch_reference["imageGridThw"] = image_grid_thw.tolist()
        reference_path = output_dir / "pytorch-reference.json"
        reference_path.write_text(json.dumps(pytorch_reference, separators=(",", ":")) + "\n", encoding="utf-8")
        report["pytorchReferencePath"] = str(reference_path)
        report["pytorchEagerCpu"] = "passed"
        finish_stage("pytorchEager", stage_started)

        current_stage = "torch-export"
        print("stage torch-export start", flush=True)
        stage_started = time.perf_counter()
        export_options = {}
        export_inputs = multimodal_example if args.multimodal_inputs else (input_ids, attention_mask)
        if args.dynamic_sequence:
            sequence_length = torch.export.Dim("sequence_length", min=1, max=max_sequence_length)
            if args.multimodal_inputs:
                image_count = torch.export.Dim("image_count", min=0, max=max_sequence_length)
                export_options["dynamic_shapes"] = (
                    {1: sequence_length},
                    {1: sequence_length},
                    {0: image_count},
                    {0: image_count},
                    {2: sequence_length},
                )
            else:
                export_options["dynamic_shapes"] = ({1: sequence_length}, {1: sequence_length})
        exported_program = torch.export.export(wrapped, export_inputs, **export_options)
        (output_dir / "exported-program.py").write_text(exported_program.graph_module.code, encoding="utf-8")
        operators = {}
        for node in exported_program.graph_module.graph.nodes:
            if node.op == "call_function":
                key = str(node.target)
                operators[key] = operators.get(key, 0) + 1
        report["torchExport"] = "passed"
        report["exportGraphNodeCount"] = len(list(exported_program.graph_module.graph.nodes))
        report["exportOperators"] = operators
        report["exportOperatorCount"] = sum(operators.values())
        finish_stage("torchExport", stage_started)

        current_stage = "relax-import"
        print("stage relax-import start", flush=True)
        stage_started = time.perf_counter()
        relax_module = from_exported_program(
            exported_program,
            keep_params_as_input=True,
            run_ep_decomposition=True,
            unwrap_unit_return_tuple=True,
        )
        if args.target == "webgpu":
            relax_module, webgpu_normalization = normalize_webgpu_dtypes(
                tvm, np, relax_module, max_sequence_length
            )
            report["webgpuDtypeNormalization"] = webgpu_normalization
        relax_script = relax_module.script()
        (output_dir / "relax-module.py").write_text(relax_script, encoding="utf-8")
        main_parameters = list(relax_module["main"].params)
        main_parameter_names = [parameter.name for parameter in main_parameters]
        exported_parameters = {
            spec.arg.name: exported_program.state_dict[spec.target]
            for spec in exported_program.graph_signature.input_specs
            if spec.kind is torch.export.graph_signature.InputKind.PARAMETER
        }
        if set(exported_parameters).difference(main_parameter_names):
            raise RuntimeError("TVM main omitted exported model parameters.")
        if "input_ids" not in main_parameter_names or "attention_mask" not in main_parameter_names:
            raise RuntimeError("TVM main omitted input_ids or attention_mask.")
        report["relaxImport"] = "passed"
        report["relaxFunctionNames"] = [global_var.name_hint for global_var in relax_module.functions]
        report["relaxMainParameterNames"] = main_parameter_names
        report["externalParameterCount"] = len(exported_parameters)
        report["externalParameterBytes"] = sum(
            parameter.numel() * parameter.element_size() for parameter in exported_parameters.values()
        )
        external_weight_payload = None if args.weight_source == "safetensors" else bytearray()
        external_weight_manifest = []
        for spec in exported_program.graph_signature.input_specs:
            if spec.kind is not torch.export.graph_signature.InputKind.PARAMETER:
                continue
            tensor = exported_program.state_dict[spec.target].detach().cpu().contiguous()
            record = {
                "name": f"model.{spec.target}",
                "parameterName": spec.arg.name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "sourceDtype": source_parameter_dtype,
                "length": tensor.numel() * tensor.element_size(),
            }
            if external_weight_payload is not None:
                weight_bytes = tensor.view(torch.uint8).numpy().tobytes()
                if record["length"] != len(weight_bytes):
                    raise RuntimeError(f"external weight byte count differs for {record['name']}")
                record["assetOffset"] = len(external_weight_payload)
                record["assetLength"] = len(weight_bytes)
                record["assetSha256"] = hashlib.sha256(weight_bytes).hexdigest()
                external_weight_payload.extend(weight_bytes)
            external_weight_manifest.append(record)
        if external_weight_payload is not None:
            external_weight_asset = output_dir / "external-weights.bin"
            external_weight_asset.write_bytes(external_weight_payload)
            report["externalWeightAssetPath"] = str(external_weight_asset)
            report["externalWeightAssetBytes"] = len(external_weight_payload)
            report["externalWeightAssetSha256"] = hashlib.sha256(external_weight_payload).hexdigest()
        else:
            report["externalWeightAssetPath"] = None
            report["externalWeightAssetBytes"] = 0
        report["externalWeightManifest"] = external_weight_manifest
        external_manifest = {
            "format": "kanana-tvm-external-weight-manifest-v1",
            "modelId": report["modelId"],
            "modelRevision": report["modelRevision"],
            "weights": report["externalWeightManifest"],
        }
        external_manifest_path = output_dir / "external-weight-manifest.json"
        external_manifest_path.write_text(json.dumps(external_manifest, indent=2) + "\n", encoding="utf-8")
        report["externalWeightManifestPath"] = str(external_manifest_path)
        report["externalWeightManifestSha256"] = sha256_file(external_manifest_path)
        relax_operators = {}
        for operator in re.findall(r"\bR\.((?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*)\(", relax_script):
            relax_operators[operator] = relax_operators.get(operator, 0) + 1
        report["relaxOperators"] = relax_operators
        finish_stage("relaxImport", stage_started)

        current_stage = "llvm-build"
        print(f"stage {current_stage} start", flush=True)
        stage_started = time.perf_counter()
        executable = tvm.relax.build(relax_module, target="llvm")
        report["relaxLlvmBuild"] = "passed"
        finish_stage("llvmBuild", stage_started)

        current_stage = "cpu-vm-run"
        print(f"stage {current_stage} start", flush=True)
        stage_started = time.perf_counter()
        sequence_lengths = args.sequence_lengths or (
            [length for length in (1, 8, 9, 33, 64, 128) if length <= max_sequence_length]
            if args.sequence_sweep
            else [8]
        )
        if (
            len(set(sequence_lengths)) != len(sequence_lengths)
            or any(length < 1 or length > max_sequence_length for length in sequence_lengths)
        ):
            raise SystemExit(f"sequence lengths must be unique values from 1 to {max_sequence_length}")
        sequence_cases = run_cpu_parity(
            tvm,
            torch,
            np,
            executable,
            exported_parameters,
            main_parameter_names,
            wrapped,
            sequence_lengths,
            multimodal_example=multimodal_example,
            dynamic_sequence=args.dynamic_sequence,
            allow_parity_delta=args.continue_after_cpu_parity_delta,
        )
        baseline_case = next(case for case in sequence_cases if case["sequenceLength"] == 8)
        sequence_case_path = output_dir / "sequence-parity-cases.json"
        sequence_case_path.write_text(json.dumps({"cases": sequence_cases}, separators=(",", ":")) + "\n", encoding="utf-8")
        report["sequenceParityCasesPath"] = str(sequence_case_path)
        report["sequenceParityCasesSha256"] = sha256_file(sequence_case_path)
        report["sequenceCases"] = [
            {
                key: value
                for key, value in case.items()
                if key
                not in (
                    "inputIds",
                    "attentionMask",
                    "imageEmbeddings",
                    "imagePositions",
                    "positionIds",
                    "referenceValues",
                )
            }
            for case in sequence_cases
        ]
        report["relaxCpuRun"] = (
            "passed"
            if all(case["withinTolerance"] for case in sequence_cases)
            else "completed_with_parity_deltas"
        )
        report["tvmOutputShape"] = baseline_case["shape"]
        report["tvmOutputDtype"] = "float32"
        report["outputMaxAbsDelta"] = baseline_case["maxAbsDelta"]
        report["outputRmse"] = baseline_case["rmse"]
        finish_stage("cpuVmRun", stage_started)

        if args.target == "webgpu":
            current_stage = "webgpu-build"
            print(f"stage {current_stage} start", flush=True)
            stage_started = time.perf_counter()
            target = tvm.target.Target(
                "webgpu",
                host={"kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"},
            )
            executable = tvm.relax.build(relax_module, target=target)
            report["relaxWebgpuBuild"] = "passed"
            finish_stage("webgpuBuild", stage_started)

            current_stage = "webgpu-wasm-export"
            print(f"stage {current_stage} start", flush=True)
            stage_started = time.perf_counter()
            from tvm.contrib import tvmjs

            wasm_path = output_dir / "kanana-text-model.wasm"
            tvm_home = Path(__file__).resolve().parents[2] / ".toolchains" / "apache-tvm-src"
            wasm_runtime_directory = tvm_home / "web" / "dist" / "wasm"
            repo_root = Path(__file__).resolve().parents[2]
            runtime_assets = [
                wasm_runtime_directory / "wasm_runtime.bc",
                wasm_runtime_directory / "tvmjs_support.bc",
                wasm_runtime_directory / "webgpu_runtime.bc",
            ]
            if any(not asset.is_file() for asset in runtime_assets):
                raise RuntimeError("The pinned TVM WebGPU/wasm32 runtime assets are not built.")
            previous_tvm_home = os.environ.get("TVM_HOME")
            os.environ["TVM_HOME"] = str(tvm_home)
            try:
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
            report["webgpuWasmPath"] = str(wasm_path)
            report["webgpuWasmBytes"] = wasm_path.stat().st_size
            report["webgpuWasmSha256"] = sha256_file(wasm_path)
            report["tvmWasmRuntimeAssets"] = [
                {"path": str(asset), "bytes": asset.stat().st_size, "sha256": sha256_file(asset)}
                for asset in runtime_assets
            ]
            report["relaxWebgpuWasmExport"] = "passed"
            finish_stage("webgpuWasmExport", stage_started)
        if args.weight_source == "safetensors":
            current_stage = "safetensors-range-manifest"
            range_manifest_path = output_dir / "range-manifest.json"
            subprocess.run(
                [
                    "node",
                    "--import",
                    "tsx",
                    str(Path(__file__).resolve().parent / "hash_safetensors_ranges.mts"),
                    str(checkpoint_path),
                    str(range_manifest_path),
                    "--verify-fp16-rn",
                ],
                cwd=Path(__file__).resolve().parents[2],
                check=True,
            )
            report["rangeManifestPath"] = str(range_manifest_path)
            report["rangeManifestSha256"] = sha256_file(range_manifest_path)
            report["weightSourceConversionVerification"] = "passed"
    except Exception as error:
        report["failedStage"] = current_stage
        report["errorType"] = type(error).__name__
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()

    wasm_path = Path(report["webgpuWasmPath"]) if report.get("relaxWebgpuWasmExport") == "passed" else None
    if wasm_path is not None:
        report = add_tvm_build_evidence(
            report,
            model_id="kanana-2-v-decide-3b-2611",
            repository="kakaocorp/kanana-2-v-decide-3b-2611",
            revision=report["modelRevision"],
            checkpoint_sha256=report["expectedCheckpointSha256"],
            graph="text",
            wasm_path=wasm_path,
            build_root=output_dir,
            tvm_version=report["versions"]["tvm"],
        )
    save_report()
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("exportOperators", "externalWeightManifest", "traceback")
            },
            indent=2,
        )
    )
    if "error" in report:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
