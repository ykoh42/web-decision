#!/usr/bin/env python3
"""Prototype the pinned Kanana text branch through torch.export and TVM Relax."""

import argparse
import hashlib
import json
import re
import resource
import sys
import time
import traceback
from pathlib import Path

PINNED_CUSTOM_FILES = {
    "config.json": "64f4a103accb8946bb7ded7dc5d200f9d6041df3eaa891c04124bb41fa06a69b",
    "configuration_kanana_v_decide.py": "7dfa4a0480db133aec3a2da8411db2232aea58097deb1e626978b554c421ec93",
    "modeling_kanana_v_decide.py": "7e7abf64026f9c30e21fb663a93b5206b9b97795fea7dabc8769c967d5812be1",
    "decision_kanana_v_decide.py": "a4e33c66db53f50118005661dbc4354b0a259e01d243cc0f4afbe0e22106b3a2",
    "processing_kanana_v_decide.py": "a95089bae7012b471e349acb3755c98f74ebd6c324afe2d6217e6a97101beba9",
    "image_processing_kanana_v.py": "4794fd3dc937b85b3e2682680fb06765c06d9d7626d7f8250179cbd63bb2bbc9",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="cached pinned model directory; reads config and custom code only")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for export graph, Relax IR, and evidence")
    parser.add_argument(
        "--full-width-one-layer",
        action="store_true",
        help="keep pinned text tensor dimensions and rope config, reducing only the transformer layer count to one",
    )
    args = parser.parse_args()
    model_dir = args.model_dir.expanduser().resolve(strict=True)
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
    code_hashes = {name: sha256_file(model_dir / name) for name in PINNED_CUSTOM_FILES}
    hash_mismatches = {
        name: {"expected": PINNED_CUSTOM_FILES[name], "actual": actual}
        for name, actual in code_hashes.items()
        if actual != PINNED_CUSTOM_FILES[name]
    }
    if hash_mismatches:
        raise SystemExit(f"cached Kanana custom source hashes differ from the pinned files: {hash_mismatches}")
    text_config = config.text_config
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
    if args.full_width_one_layer:
        # Preserve every text tensor dimension and pinned RoPE setting. Only remove
        # the remaining 31 transformer blocks to bound CPU memory and compile cost.
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
        text_config._attn_implementation = "eager"
    full_width_dimensions_unchanged = args.full_width_one_layer and all(
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

    # The top-level custom model constructs its actual pinned language_model and
    # vision child. The tiny vision config is instantiated but never called.
    config.vision_config.depth = 1
    config.vision_config.hidden_size = 32
    config.vision_config.intermediate_size = 64
    config.vision_config.num_heads = 2
    config.vision_config.patch_size = 2
    config.vision_config.temporal_patch_size = 1
    config.vision_config.spatial_merge_size = 1
    config.vision_config.out_hidden_size = 32
    config.vision_config.num_position_embeddings = 16
    config.vision_config.in_channels = 3
    config.vision_config.gated_attention = True
    report = {
        "modelId": "kakaocorp/kanana-2-v-decide-3b-2611",
        "modelRevision": "1489062def4145ae3a0a9261fbac4f4d88a8c6d1",
        "expectedCheckpointSha256": "528930fa261b31ed1f47eecbe7d08efd2e226207508124ae186e1a65dbbbda3b",
        "modelDirectory": model_dir.name,
        "customSourceSha256": code_hashes,
        "mode": "full_width_one_layer" if args.full_width_one_layer else "reduced",
        "fullWidthDimensionsUnchanged": full_width_dimensions_unchanged,
        "onlyLayerCountReduced": args.full_width_one_layer
        and text_config.num_hidden_layers == 1
        and pinned_text_config["layers"] > 1
        and full_width_dimensions_unchanged,
        "pinnedTextConfig": pinned_text_config,
        "checkpointLoaded": False,
        "checkpointBytesRead": False,
        "onnxImportOrExportUsed": False,
        "device": "cpu",
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
        model = AutoModel.from_config(config, trust_remote_code=True)
        model.eval()
        branch = model.language_model
        report["modelClass"] = type(model).__name__
        report["textBranchClass"] = type(branch).__name__
        report["parameterCount"] = sum(parameter.numel() for parameter in branch.parameters())
        report["parameterBytes"] = sum(parameter.numel() * parameter.element_size() for parameter in branch.parameters())
        finish_stage("modelInit", stage_started)

        class TextBranch(torch.nn.Module):
            def __init__(self, language_model):
                super().__init__()
                self.language_model = language_model

            def forward(self, input_ids, attention_mask):
                return self.language_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).last_hidden_state

        wrapped = TextBranch(branch).eval()
        input_ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15]], dtype=torch.int64, device="cpu")
        attention_mask = torch.ones((1, 8), dtype=torch.int64, device="cpu")
        current_stage = "pytorch-eager"
        print("stage pytorch-eager start", flush=True)
        stage_started = time.perf_counter()
        with torch.inference_mode():
            torch_output = wrapped(input_ids, attention_mask)
        report["pytorchOutputShape"] = list(torch_output.shape)
        report["pytorchOutputDtype"] = str(torch_output.dtype)
        report["pytorchEagerCpu"] = "passed"
        finish_stage("pytorchEager", stage_started)

        current_stage = "torch-export"
        print("stage torch-export start", flush=True)
        stage_started = time.perf_counter()
        exported_program = torch.export.export(wrapped, (input_ids, attention_mask))
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
            run_ep_decomposition=True,
            unwrap_unit_return_tuple=True,
        )
        relax_script = relax_module.script()
        (output_dir / "relax-module.py").write_text(relax_script, encoding="utf-8")
        report["relaxImport"] = "passed"
        report["relaxFunctionNames"] = [global_var.name_hint for global_var in relax_module.functions]
        relax_operators = {}
        for operator in re.findall(r"\bR\.((?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*)\(", relax_script):
            relax_operators[operator] = relax_operators.get(operator, 0) + 1
        report["relaxOperators"] = relax_operators
        finish_stage("relaxImport", stage_started)

        current_stage = "llvm-build"
        print("stage llvm-build start", flush=True)
        stage_started = time.perf_counter()
        executable = tvm.relax.build(relax_module, target="llvm")
        report["relaxLlvmBuild"] = "passed"
        finish_stage("llvmBuild", stage_started)

        current_stage = "cpu-vm-run"
        print("stage cpu-vm-run start", flush=True)
        stage_started = time.perf_counter()
        vm = tvm.relax.VirtualMachine(executable, tvm.cpu())
        tvm_input_ids = tvm.runtime.tensor(input_ids.numpy(), device=tvm.cpu())
        tvm_attention_mask = tvm.runtime.tensor(attention_mask.numpy(), device=tvm.cpu())
        tvm_output = vm["main"](tvm_input_ids, tvm_attention_mask)
        report["relaxCpuRun"] = "passed"
        if not callable(getattr(tvm_output, "numpy", None)):
            tvm_output = tvm_output[0]
        tvm_array = np.asarray(tvm_output.numpy(), dtype=np.float32)
        expected = torch_output.detach().float().numpy()
        report["relaxCpuRun"] = "passed"
        report["tvmOutputShape"] = list(tvm_array.shape)
        report["tvmOutputDtype"] = str(tvm_output.dtype)
        report["outputMaxAbsDelta"] = float(np.max(np.abs(tvm_array - expected)))
        report["outputRmse"] = float(np.sqrt(np.mean(np.square(tvm_array - expected))))
        finish_stage("cpuVmRun", stage_started)
    except Exception as error:
        report["failedStage"] = current_stage
        report["errorType"] = type(error).__name__
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()

    save_report()
    print(json.dumps({key: value for key, value in report.items() if key not in ("exportOperators", "traceback")}, indent=2))
    if "error" in report:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
