#!/usr/bin/env python3
"""Import the official Kev checkpoint directly from torch.export into TVM Relax."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

from direct_wasm_packager import (
    KEV_DELIMITER_IDS,
    LICENSE_PROVENANCE,
    package_tvm_webgpu_artifacts,
)
from kev_attention_relax import rewrite_kev_attention

KEV_CODE_REVISION = "37cb2f6155afec1054de2cc78dfd7c7dd24a956d"
KEV_CHECKPOINT_REVISION = "abe7110dedfd68a185839550079bcf7c26f561ac"
BASE_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
INPUT_HASHES = {
    "kev/adapter_config.json": "5bfcf3dd85b1383c802146cfdef179e2e772685e310ac972fb672f22d7079b4f",
    "kev/adapter_model.safetensors": "deaab63b61f95d628503831e8b336e1c3e334b9a42d3530d979a5149b429c492",
    "kev/head.pt": "ce6cd9ffc54db41c179b65a33d60973dc8280c29e886218eb3ec07dc28d6f28b",
    "kev/provenance.json": "321bc9db7427b490ad943fa33576c7e01f5da4c20da1036fcbfee996f4b28f10",
    "kev/training_config.json": "ab20e55cbb832f409de9daf42f868b69be1eb59e51720302b166bc560fd8d240",
    "base/LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    "base/config.json": "504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59",
    "base/model.safetensors": "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
    "base/tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "base/tokenizer_config.json": "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5",
}
UPSTREAM_REFERENCE_FIXTURE_SHA256 = (
    "656e8403984e57f24dd38f08d461188bde7802288c369a88dac0b29319bada10"
)
EXPANDED_INFERENCE_FIXTURE_SHA256 = (
    "05e7f81d2ec5e8a306c0a8034253e3daa3983c088b8ccd9c29b5130b2eecbd41"
)
EXPANDED_DECISION_CASES = (
    ("baseline-double-charge", 22),
    ("account-security", 64),
    ("delayed-package", 57),
    ("long-tracking-context", 243),
)
EXPANDED_Q4_LOGIT_TOLERANCE = 0.15
EXPECTED_KEV_PARAMETER_COUNT = 511
PINNED_TVM_VERSION = "0.27.0.post1"
PINNED_TVM_WEB_REVISION = "7b5acb32bd271342a24130f4ef439ff6c2e52c6c"
PINNED_TVM_WEB_TAG = "v0.27.0.post1"
PINNED_TVM_WEB_LIBRARIES = {
    "web/dist/wasm/wasm_runtime.bc": "1974cd7846e3735735fe48828856b7113257a426c24a49e44f45c79c412b972f",
    "web/dist/wasm/tvmjs_support.bc": "24088d9f55002852d513af36b64770dacb1825e5dffebb95d23b43f104b19fc9",
    "web/dist/wasm/webgpu_runtime.bc": "c4baf79ed300de425db4d81743ab89d36cee323e40e18904c4ef7f196f2fe0aa",
}
PINNED_LIBATOMIC_PACKAGE_VERSION = "14.2.0-4ubuntu2~24.04.1"
PINNED_LIBATOMIC_SHA256 = (
    "9d8a34468e614a71a6427dbabb0ac24eb8cb1e324dc6aa3bf95eb6f59201f000"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_tvmjs_toolchain(tvm_home: Path) -> dict[str, object]:
    revision = subprocess.run(
        ["git", "-C", str(tvm_home), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tag = subprocess.run(
        ["git", "-C", str(tvm_home), "describe", "--tags", "--exact-match"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(tvm_home), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != PINNED_TVM_WEB_REVISION or tag != PINNED_TVM_WEB_TAG or status:
        raise ValueError(
            "TVM Web runtime source must be the clean pinned v0.27.0.post1 checkout."
        )
    libraries = {}
    mismatches = []
    for relative_path, expected_hash in PINNED_TVM_WEB_LIBRARIES.items():
        actual_hash = sha256_file(tvm_home / relative_path)
        if actual_hash != expected_hash:
            mismatches.append(
                f"{relative_path}: expected {expected_hash}, got {actual_hash}"
            )
        libraries[relative_path] = actual_hash
    if mismatches:
        raise ValueError(
            "Pinned TVM Web runtime library hash mismatch: "
            + "; ".join(mismatches)
            + "."
        )
    return {"sourceRevision": revision, "sourceTag": tag, "librarySha256": libraries}


def verify_libatomic(library_path: Path) -> dict[str, str]:
    actual_hash = sha256_file(library_path)
    if actual_hash != PINNED_LIBATOMIC_SHA256:
        raise ValueError("Pinned libatomic1 runtime library hash mismatch.")
    return {
        "package": "libatomic1",
        "packageVersion": PINNED_LIBATOMIC_PACKAGE_VERSION,
        "soname": "libatomic.so.1",
        "librarySha256": actual_hash,
    }


def extend_runtime_library_path(existing_path: str, runtime_directory: Path) -> str:
    """Expose libatomic after the active Python runtime's shared libraries."""
    return os.pathsep.join(
        value for value in (existing_path, str(runtime_directory)) if value
    )


def probabilities(logits, np):
    shifted = logits - np.max(logits)
    values = np.exp(shifted)
    return values / np.sum(values)


def load_upstream_reference(root: Path) -> tuple[dict, str]:
    path = root / "tests/fixtures/kev06-upstream-onnx-reference.json"
    digest = sha256_file(path)
    if digest != UPSTREAM_REFERENCE_FIXTURE_SHA256:
        raise ValueError("Pinned Kev Q4 reference fixture hash mismatch.")
    return json.loads(path.read_text(encoding="utf-8")), digest


def load_expanded_inference_fixture(
    root: Path, tokenizer_sha256: str
) -> tuple[dict, str]:
    path = root / "tests/fixtures/kev06-expanded-inference.json"
    digest = sha256_file(path)
    if digest != EXPANDED_INFERENCE_FIXTURE_SHA256:
        raise ValueError("Pinned expanded Kev inference fixture hash mismatch.")
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(fixture, dict):
        raise ValueError("Expanded Kev inference fixture must be a JSON object.")
    if (
        fixture.get("modelId") != "kev-0.6b-q4f16"
        or fixture.get("tokenizerSha256") != tokenizer_sha256
    ):
        raise ValueError("Expanded Kev fixture does not match the pinned model assets.")
    cases = fixture.get("decisionCases")
    if not isinstance(cases, list) or len(cases) != len(EXPANDED_DECISION_CASES):
        raise ValueError("Expanded Kev fixture has an unexpected decision case count.")
    for case, (expected_id, expected_length) in zip(
        cases, EXPANDED_DECISION_CASES, strict=True
    ):
        _validate_expanded_decision_case(case, expected_id, expected_length)
    return fixture, digest


def _validate_expanded_decision_case(
    case, expected_id: str, expected_length: int
) -> None:
    if not isinstance(case, dict) or case.get("id") != expected_id:
        raise ValueError(f"Expanded Kev case ID differs from {expected_id}.")
    token_ids = case.get("tokenIds")
    if (
        case.get("tokenCount") != expected_length
        or not isinstance(token_ids, list)
        or len(token_ids) != expected_length
        or any(type(token_id) is not int or token_id < 0 for token_id in token_ids)
    ):
        raise ValueError(f"Expanded Kev case has invalid token inputs: {expected_id}.")
    reference = case.get("reference")
    request = case.get("request")
    questions = request.get("questions") if isinstance(request, dict) else None
    indices = case.get("optionIndices")
    if (
        not isinstance(reference, dict)
        or not isinstance(questions, list)
        or not questions
    ):
        raise ValueError(f"Expanded Kev case is missing its reference: {expected_id}.")
    options = questions[0].get("options") if isinstance(questions[0], dict) else None
    if (
        not isinstance(options, list)
        or reference.get("options") != options
        or not isinstance(indices, list)
        or len(indices) != len(options)
        or any(type(index) is not int or index < 0 for index in indices)
    ):
        raise ValueError(f"Expanded Kev case has invalid option inputs: {expected_id}.")
    logits = reference.get("logits")
    reference_probabilities = reference.get("probabilities")
    if (
        not isinstance(logits, list)
        or not isinstance(reference_probabilities, list)
        or len(logits) != len(options)
        or len(reference_probabilities) != len(options)
        or reference.get("selected") not in options
    ):
        raise ValueError(f"Expanded Kev case has invalid Q4 references: {expected_id}.")


def run_cpu_vm(vm, tvm, np, token_ids: list[int], parameters) -> object:
    input_ids = np.asarray([token_ids], dtype=np.int32)
    attention_mask = np.ones_like(input_ids)
    result = vm["main"](
        tvm.runtime.tensor(input_ids, device=tvm.cpu()),
        tvm.runtime.tensor(attention_mask, device=tvm.cpu()),
        *parameters,
    )
    if not callable(getattr(result, "numpy", None)):
        result = result[0]
    output = np.asarray(result.numpy(), dtype=np.float32)
    if output.shape != input_ids.shape:
        raise ValueError("TVM CPU output shape differs from the packed input.")
    if not np.isfinite(output).all():
        raise ValueError("TVM CPU output contains non-finite values.")
    return output


def expanded_gpu_gate_contract(fixture: dict, fixture_sha256: str) -> dict:
    return {
        "status": "pending-real-chrome-webgpu",
        "fixtureSha256": fixture_sha256,
        "decisionCases": [
            {"id": case["id"], "tokenCount": case["tokenCount"]}
            for case in fixture["decisionCases"]
        ],
        "maxAbsoluteQ4LogitDifference": EXPANDED_Q4_LOGIT_TOLERANCE,
        "q4ProbabilityComparison": "descriptive-only",
        "allSelectionsMustMatchPinnedQ4": True,
        "outputsMustBeFinite": True,
        "probabilitiesMustSumToOne": True,
        "maxProbabilitySumError": 0.000001,
        "baselineCpuGpuParityRequired": True,
    }


def validate_upstream_inputs(encoded, reference) -> None:
    if encoded["ids"] != reference["tokenIds"]:
        raise ValueError(
            "Direct Kev input token IDs differ from the pinned Q4 reference."
        )
    if encoded["opt_idx"][0] != reference["optionIndices"]:
        raise ValueError(
            "Direct Kev option indices differ from the pinned Q4 reference."
        )


def compare_upstream_output(actual, reference, np) -> dict[str, float | bool]:
    actual = np.asarray(actual, dtype=np.float32)
    expected_logits = np.asarray(reference["logits"], dtype=np.float32)
    expected_probabilities = np.asarray(reference["probabilities"], dtype=np.float32)
    if (
        actual.shape != expected_logits.shape
        or expected_logits.shape != expected_probabilities.shape
    ):
        raise ValueError("TVM and pinned Q4 reference output shapes differ.")
    if (
        not np.isfinite(actual).all()
        or not np.isfinite(expected_logits).all()
        or not np.isfinite(expected_probabilities).all()
    ):
        raise ValueError(
            "TVM or pinned Q4 reference output contains non-finite values."
        )
    actual_probabilities = probabilities(actual, np)
    return {
        "maxAbsLogitDelta": float(np.max(np.abs(actual - expected_logits))),
        "maxAbsProbabilityDelta": float(
            np.max(np.abs(actual_probabilities - expected_probabilities))
        ),
        "selectedOptionMatches": int(np.argmax(actual))
        == int(np.argmax(expected_logits)),
    }


def verify_inputs(kev_dir: Path, base_dir: Path, source_dir: Path) -> dict[str, str]:
    revision = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != KEV_CODE_REVISION:
        raise ValueError(
            f"Kev reference code must be at {KEV_CODE_REVISION}, found {revision}."
        )
    changed = subprocess.run(
        ["git", "-C", str(source_dir), "diff-index", "--quiet", "HEAD", "--"],
        check=False,
    ).returncode
    if changed:
        raise ValueError("Kev reference code checkout has tracked modifications.")

    hashes = {}
    for name, expected in INPUT_HASHES.items():
        root = kev_dir if name.startswith("kev/") else base_dir
        relative = name.removeprefix("kev/").removeprefix("base/")
        actual = sha256_file(root / relative)
        if actual != expected:
            raise ValueError(f"Pinned source hash mismatch for {name}: {actual}")
        hashes[name] = actual
    return hashes


def load_reference(
    torch, transformers, peft, source_dir: Path, kev_dir: Path, base_dir: Path
):
    sys.path.insert(0, str(source_dir))
    from kev.model import DecisionModel

    metadata = torch.load(kev_dir / "head.pt", map_location="cpu", weights_only=True)
    if metadata.get("base_revision") != BASE_REVISION or metadata.get(
        "option_isolation"
    ):
        raise ValueError(
            "Kev head metadata does not match the pinned base and branch-mask contract."
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_dir, local_files_only=True
    )
    model = DecisionModel(
        str(base_dir),
        tokenizer,
        "cpu",
        lora=None,
        revision=None,
        attn="eager",
        dtype=torch.float32,
    )
    model.lm = peft.PeftModel.from_pretrained(
        model.lm, str(kev_dir), is_trainable=False
    )
    model.lm = model.lm.merge_and_unload()
    model.head.load_state_dict(metadata["head"])
    model.eval()
    return tokenizer, model


def source_parameter_tensors(tvm, exported_program, model):
    values = dict(exported_program.state_dict)
    values.update(model.state_dict())
    tensors = []
    names = []
    for spec in exported_program.graph_signature.input_specs:
        if spec.kind.name != "PARAMETER":
            continue
        name = str(spec.target)
        if name not in values:
            raise ValueError(f"Exported parameter {name} has no source tensor.")
        array = values[name].detach().cpu().contiguous().numpy()
        tensors.append(tvm.runtime.tensor(array, device=tvm.cpu()))
        names.append(name)
    expected = sum(
        spec.kind.name == "PARAMETER"
        for spec in exported_program.graph_signature.input_specs
    )
    if len(tensors) != expected:
        raise ValueError(
            "Exported graph parameter tensors do not match its parameter inputs."
        )
    return names, tensors


def export_dynamic_program(torch, model, input_ids, attention_mask):
    """Export an unbounded packed sequence dimension without adding an aggregate token cap."""
    sequence_length = torch.export.Dim("sequence_length", min=1)
    return torch.export.export(
        model,
        (input_ids, attention_mask),
        dynamic_shapes={
            "input_ids": {1: sequence_length},
            "attention_mask": {1: sequence_length},
        },
    )


def package_direct_model(
    executable, parameters, output_dir: Path, base_dir: Path, kev_source_dir: Path
):
    """Write the direct WebGPU executable and official base tokenizer assets."""
    # Hugging Face cache snapshots expose pinned files through symlinks to
    # content-addressed blobs. verify_inputs checks those files before export;
    # pass their canonical regular-file paths to the package writer.
    return package_tvm_webgpu_artifacts(
        executable,
        parameters,
        output_dir / "browser-package",
        (base_dir / "tokenizer.json").resolve(strict=True),
        (base_dir / "tokenizer_config.json").resolve(strict=True),
        (base_dir / "config.json").resolve(strict=True),
        (kev_source_dir / "LICENSE").resolve(strict=True),
        (base_dir / "LICENSE").resolve(strict=True),
    )


def build_and_package_direct_model(
    tvm,
    target,
    relax_module,
    parameters,
    output_dir: Path,
    base_dir: Path,
    kev_source_dir: Path,
):
    executable = tvm.relax.build(relax_module, target=target)
    return package_direct_model(
        executable, parameters, output_dir, base_dir, kev_source_dir
    )


def build_webgpu_package(
    tvm,
    integrate_q4,
    fp16_relax_module,
    parameter_names,
    source_parameters,
    expected_quantization,
    expected_parameter_count,
    output_dir: Path,
    base_dir: Path,
    kev_source_dir: Path,
):
    target = tvm.target.Target(
        "webgpu", host={"kind": "llvm", "mtriple": "wasm32-unknown-unknown"}
    )
    if (
        len(parameter_names) != len(source_parameters)
        or expected_parameter_count != EXPECTED_KEV_PARAMETER_COUNT
    ):
        raise ValueError(
            "Pinned Kev WebGPU export requires exactly 511 runtime parameters."
        )
    relax_module, parameters, quantization = integrate_q4(
        fp16_relax_module,
        parameter_names,
        source_parameters,
        tvm.cpu(),
        target=target,
    )
    if quantization != expected_quantization:
        raise ValueError("CPU and WebGPU Q4 parameter encodings differ.")
    if len(parameters) != expected_parameter_count:
        raise ValueError("WebGPU runtime parameter count differs from CPU.")

    relax_module, attention_report = rewrite_kev_attention(
        relax_module,
        question_id=KEV_DELIMITER_IDS["question"],
        decide_id=KEV_DELIMITER_IDS["decide"],
        expected_attention_blocks=28,
    )
    package = build_and_package_direct_model(
        tvm, target, relax_module, parameters, output_dir, base_dir, kev_source_dir
    )
    return {
        "target": "webgpu",
        "directory": package.output_dir.name,
        "parameterCount": package.parameter_count,
        "shardCount": package.shard_count,
        "wasmBytes": (package.output_dir / "model.wasm").stat().st_size,
        "manifestSha256": package.manifest_sha256,
        "license": "Apache-2.0",
        "licenses": [dict(record) for record in LICENSE_PROVENANCE],
        "attentionRewrite": {
            "attentionBlocks": attention_report.attention_blocks,
            "metadataBindings": list(attention_report.metadata_bindings),
            "preprocessCalls": attention_report.preprocess_calls,
            "fusedFunction": attention_report.fused_function,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kev-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--tvm-home", type=Path, required=True)
    parser.add_argument("--libatomic-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    kev_dir = args.kev_dir.resolve(strict=True)
    base_dir = args.base_dir.resolve(strict=True)
    source_dir = args.source_dir.resolve(strict=True)
    tvm_home = args.tvm_home.resolve(strict=True)
    libatomic_path = args.libatomic_path.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    web_toolchain = verify_tvmjs_toolchain(tvm_home)
    web_toolchain["emscriptenRuntimeDependency"] = verify_libatomic(libatomic_path)
    os.environ["TVM_HOME"] = str(tvm_home)
    existing_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = extend_runtime_library_path(
        existing_library_path, libatomic_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    import numpy as np
    import peft
    import torch
    import transformers
    import tvm
    from tvm.relax.frontend.torch import from_exported_program

    if tvm.__version__ != PINNED_TVM_VERSION:
        raise ValueError(f"TVM exporter must use apache-tvm=={PINNED_TVM_VERSION}.")

    torch.set_num_threads(1)
    torch.manual_seed(20261110)
    report = {
        "candidateModelId": "kev-0.6b-q4f16",
        "buildStage": "direct-tvm-q4f16-webgpu-candidate",
        "precision": {
            "backbone": "q4-uint4-offset8",
            "embedding": "q4-signed-int4",
            "scales": "float16",
            "pointerHead": "float32",
            "quantized": False,
        },
        "kevCheckpointRevision": KEV_CHECKPOINT_REVISION,
        "baseRevision": BASE_REVISION,
        "kevCodeRevision": KEV_CODE_REVISION,
        "onnxIntermediateUsed": False,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "tvm": tvm.__version__,
            "python": platform.python_version(),
        },
        "tvmWebToolchain": web_toolchain,
        "executionPlatform": platform.platform(),
        "converterHashes": {
            path.name: sha256_file(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("kev_wrapper.py"),
                Path(__file__).with_name("q4.py"),
                Path(__file__).with_name("q4_relax.py"),
                Path(__file__).with_name("direct_wasm_packager.py"),
                Path(__file__).with_name("kev_attention_relax.py"),
                Path(__file__).with_name("kev_attention_grid.py"),
                Path(__file__).with_name("kev_preprocess.py"),
                Path(__file__).with_name("pyproject.toml"),
                Path(__file__).with_name("uv.lock"),
            )
        },
        "stages": [],
    }
    current_stage = "source-verification"
    stage_started = time.perf_counter()

    def finish_stage(name: str) -> None:
        nonlocal stage_started
        report["stages"].append(
            {"name": name, "seconds": round(time.perf_counter() - stage_started, 3)}
        )
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        stage_started = time.perf_counter()

    report_path = output_dir / "direct-tvm-report.json"
    try:
        repository_root = Path(__file__).resolve().parents[2]
        upstream_reference, upstream_fixture_hash = load_upstream_reference(
            repository_root
        )
        expanded_fixture, expanded_fixture_hash = load_expanded_inference_fixture(
            repository_root, INPUT_HASHES["base/tokenizer.json"]
        )
        report["upstreamReference"] = {
            "fixtureSha256": upstream_fixture_hash,
            "modelId": upstream_reference["modelId"],
            "revision": upstream_reference["revision"],
            "graphSha256": upstream_reference["graphSha256"],
            "weightsSha256": upstream_reference["weightsSha256"],
        }
        report["expandedInferenceFixtureSha256"] = expanded_fixture_hash
        report["sourceHashes"] = verify_inputs(kev_dir, base_dir, source_dir)
        tokenizer, source_model = load_reference(
            torch, transformers, peft, source_dir, kev_dir, base_dir
        )
        fixture_path = (
            Path(__file__).resolve().parents[2]
            / "tests/fixtures/kev06-main-baseline.json"
        )
        report["inputFixtureSha256"] = sha256_file(fixture_path)
        fixture = json.loads(fixture_path.read_text())
        request = fixture["request"]
        record = {
            "state": request["state"],
            "questions": [
                {
                    "instr": request["questions"][0]["instructions"],
                    "options": request["questions"][0]["options"],
                    "label": 0,
                }
            ],
        }
        encoded = source_model.encode(tokenizer, record, strict=True)
        validate_upstream_inputs(encoded, upstream_reference)
        input_ids = torch.tensor([encoded["ids"]], dtype=torch.int32)
        attention_mask = torch.ones_like(input_ids)
        source_logits = source_model.forward(encoded)[0].detach().cpu().float().numpy()
        source_model.lm = source_model.lm.to(torch.float16)
        for parameter in source_model.parameters():
            parameter.requires_grad_(False)
        from kev_wrapper import KevDelimiters, KevGraph, prepare_matmul_weights

        prepare_matmul_weights(source_model.lm)
        prepare_matmul_weights(source_model.head)
        delimiters = KevDelimiters(
            *(
                tokenizer.convert_tokens_to_ids(token)
                for token in ("<|fim_middle|>", "<|box_end|>", "<|fim_suffix|>")
            )
        )
        wrapped = KevGraph(source_model.lm, source_model.head, delimiters).eval()
        option_indices = torch.tensor(encoded["opt_idx"][0], dtype=torch.int64)
        with torch.inference_mode():
            reference_logits = (
                wrapped(input_ids, attention_mask)[0, option_indices].float().numpy()
            )
        report["inputShape"] = list(input_ids.shape)
        report["inputDtype"] = str(input_ids.dtype)
        report["attentionMaskDtype"] = str(attention_mask.dtype)
        report["sourceLogits"] = source_logits.tolist()
        report["wrappedTorchLogits"] = reference_logits.tolist()
        report["sourceVsWrappedMaxAbsDelta"] = float(
            np.max(np.abs(source_logits - reference_logits))
        )
        finish_stage(current_stage)

        current_stage = "torch-export"
        exported_program = export_dynamic_program(
            torch, wrapped, input_ids, attention_mask
        )
        report["torchExportRangeConstraints"] = str(exported_program.range_constraints)
        report["torchExportNodeCount"] = len(
            list(exported_program.graph_module.graph.nodes)
        )
        report["exportedParameterCount"] = sum(
            spec.kind.name == "PARAMETER"
            for spec in exported_program.graph_signature.input_specs
        )
        report["exportedBufferCount"] = sum(
            spec.kind.name == "BUFFER"
            for spec in exported_program.graph_signature.input_specs
        )
        (output_dir / "exported-program.py").write_text(
            exported_program.graph_module.code, encoding="utf-8"
        )
        finish_stage(current_stage)

        current_stage = "relax-import"
        relax_module = from_exported_program(
            exported_program,
            run_ep_decomposition=True,
            unwrap_unit_return_tuple=True,
            keep_params_as_input=True,
        )
        fp16_relax_module = relax_module
        report["relaxParameterCount"] = len(relax_module["main"].params) - 2
        report["relaxOutputType"] = str(relax_module["main"].ret_ty)
        (output_dir / "relax-module-fp16.py").write_text(
            relax_module.script(), encoding="utf-8"
        )
        parameter_names, source_parameters = source_parameter_tensors(
            tvm, exported_program, wrapped
        )
        if len(source_parameters) != report["relaxParameterCount"]:
            raise ValueError("Exported graph and Relax source parameter counts differ.")
        from q4_relax import integrate_q4

        current_stage = "relax-q4-integration"
        relax_module, parameters, quantization = integrate_q4(
            relax_module, parameter_names, source_parameters, tvm.cpu()
        )
        report["sourceParameterCount"] = len(source_parameters)
        report["relaxParameterCount"] = len(relax_module["main"].params) - 2
        report["q4"] = quantization
        report["precision"]["quantized"] = True
        report["quantizedParameterInputCount"] = len(parameters)
        if len(parameters) != EXPECTED_KEV_PARAMETER_COUNT:
            raise ValueError("Pinned Kev Q4 runtime parameter count differs from 511.")
        if len(parameters) != report["relaxParameterCount"]:
            raise ValueError("Q4 Relax and runtime parameter counts differ.")
        (output_dir / "relax-module.py").write_text(
            relax_module.script(), encoding="utf-8"
        )
        finish_stage("relax-q4-integration")

        current_stage = "llvm-build"
        executable = tvm.relax.build(relax_module, target="llvm")
        finish_stage(current_stage)

        current_stage = "cpu-vm-run"
        vm = tvm.relax.VirtualMachine(executable, tvm.cpu())
        output = run_cpu_vm(vm, tvm, np, encoded["ids"], parameters)
        expected = reference_logits
        actual = output[0, encoded["opt_idx"][0]]
        report["tvmLogits"] = actual.tolist()
        report["maxAbsDeltaFromWrappedTorch"] = float(np.max(np.abs(actual - expected)))
        report["maxAbsDeltaFromOfficialFp32"] = float(
            np.max(np.abs(actual - source_logits))
        )
        source_probabilities = probabilities(source_logits, np)
        wrapped_probabilities = probabilities(expected, np)
        tvm_probabilities = probabilities(actual, np)
        report["maxAbsProbabilityDeltaFromOfficialFp32"] = float(
            np.max(np.abs(tvm_probabilities - source_probabilities))
        )
        report["maxAbsProbabilityDeltaFromWrappedTorch"] = float(
            np.max(np.abs(tvm_probabilities - wrapped_probabilities))
        )
        reference_comparison = compare_upstream_output(actual, upstream_reference, np)
        report["upstreamQ4ReferenceComparison"] = reference_comparison
        report["selectedOptionIndexMatchesOfficial"] = int(np.argmax(actual)) == int(
            np.argmax(source_logits)
        )
        report["selectedOptionIndexMatchesUpstreamQ4"] = reference_comparison[
            "selectedOptionMatches"
        ]
        report["acceptanceThresholds"] = {
            "maxCpuLogitDeltaFromPinnedUpstreamQ4": upstream_reference[
                "maxAbsoluteLogitDifference"
            ],
            "maxCpuProbabilityDeltaFromPinnedUpstreamQ4": upstream_reference[
                "maxAbsoluteProbabilityDifference"
            ],
            "selectedOptionMustMatchPinnedUpstreamQ4": True,
            "selectedOptionMustMatchOfficial": True,
            "outputMustBeFinite": True,
            "outputShapeMustMatchInput": True,
        }
        if (
            reference_comparison["maxAbsLogitDelta"]
            > upstream_reference["maxAbsoluteLogitDifference"]
        ):
            raise ValueError("TVM CPU logits exceed the pinned Q4 reference threshold.")
        if (
            reference_comparison["maxAbsProbabilityDelta"]
            > upstream_reference["maxAbsoluteProbabilityDifference"]
        ):
            raise ValueError(
                "TVM CPU probabilities exceed the pinned Q4 reference threshold."
            )
        if not report["selectedOptionIndexMatchesUpstreamQ4"]:
            raise ValueError(
                "TVM CPU selected a different option than the pinned Q4 reference."
            )
        if not report["selectedOptionIndexMatchesOfficial"]:
            raise ValueError("TVM Q4 changed the official model's selected option.")
        report["qualityGates"] = {
            "cpuBaseline": {
                "status": "passed",
                "inputTokenCount": len(encoded["ids"]),
                "reference": "pinned-upstream-q4",
                **reference_comparison,
            },
            "webgpuExpandedMatrix": expanded_gpu_gate_contract(
                expanded_fixture, expanded_fixture_hash
            ),
        }
        report["expandedInference"] = report["qualityGates"]["webgpuExpandedMatrix"]
        report["parameterInputCount"] = len(parameters)

        finish_stage(current_stage)
        current_stage = "webgpu-build"
        del executable, vm, parameters, relax_module
        del wrapped, source_model, exported_program, tokenizer
        report["browserPackage"] = build_webgpu_package(
            tvm,
            integrate_q4,
            fp16_relax_module,
            parameter_names,
            source_parameters,
            report["q4"],
            report["quantizedParameterInputCount"],
            output_dir,
            base_dir,
            source_dir,
        )
        report["status"] = "passed"
        finish_stage(current_stage)
    except Exception as error:
        report["failedStage"] = current_stage
        report["errorType"] = type(error).__name__
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
