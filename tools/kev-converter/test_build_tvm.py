"""Checks for the direct Kev converter's pinned compatibility gate."""

import hashlib
import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from build_tvm import (
    EXPANDED_DECISION_CASES,
    EXPANDED_INFERENCE_FIXTURE_SHA256,
    PINNED_LIBATOMIC_PACKAGE_VERSION,
    PINNED_LIBATOMIC_SHA256,
    PINNED_TVM_WEB_LIBRARIES,
    PINNED_TVM_WEB_REVISION,
    PINNED_TVM_WEB_TAG,
    UPSTREAM_REFERENCE_FIXTURE_SHA256,
    compare_upstream_output,
    expanded_gpu_gate_contract,
    export_dynamic_program,
    extend_runtime_library_path,
    load_expanded_inference_fixture,
    load_upstream_reference,
    validate_upstream_inputs,
    verify_libatomic,
    verify_tvmjs_toolchain,
)

HISTORICAL_CONVERTER_SOURCE_HASHES = {
    "build_tvm.py": "cd4d11a735026628c0f15f7d6598d095a82540fbf946e141ea03153abe078ccc",
    "kev_wrapper.py": "b00bf287a336fed6d3777944704929621339cda5a2b2c9b6ac88c18101a9bcbe",
    "q4.py": "72f716ee3c32dda77da195f83d6206a687cf344e2da917921b33c4d2f5f1dcd3",
    "q4_relax.py": "17c079431d28c4540aafc9fd78ae5cdc0bd058ce596e7f6e66fbd9c606131e0d",
}

HISTORICAL_WEBGPU_R1_SOURCE_HASHES = {
    "build_tvm.py": "0e48f555e7492dd3e12d237c1ea0e4866692275a5b1bab7b6d78de312a3a2ec5",
    "kev_attention_relax.py": "e11fdb0df8b620198a2ecd2a6580035f68dcfe1b3469ae8e7034a0d93fd26472",
    "kev_preprocess.py": "f25fbe7acffc7fe75aa9b382a855754a8050e50e4bace9f797a37bc4811decc0",
}


class UpstreamReferenceTests(unittest.TestCase):
    def setUp(self):
        self.reference, self.fixture_hash = load_upstream_reference(
            Path(__file__).resolve().parents[2]
        )

    def test_export_does_not_impose_an_aggregate_sequence_upper_bound(self):
        import torch

        class AddInputs(torch.nn.Module):
            def forward(self, input_ids, attention_mask):
                return input_ids + attention_mask

        inputs = torch.zeros((1, 22), dtype=torch.int32)
        exported = export_dynamic_program(torch, AddInputs(), inputs, inputs)

        self.assertIn("int_oo", str(exported.range_constraints))

    def test_reference_fixture_is_hash_pinned(self):
        self.assertEqual(self.fixture_hash, UPSTREAM_REFERENCE_FIXTURE_SHA256)

    def test_reference_inputs_are_exact(self):
        encoded = {
            "ids": self.reference["tokenIds"],
            "opt_idx": [self.reference["optionIndices"]],
        }

        validate_upstream_inputs(encoded, self.reference)

    def test_wrong_tokens_or_option_indices_fail_closed(self):
        encoded = {
            "ids": [*self.reference["tokenIds"], 0],
            "opt_idx": [self.reference["optionIndices"]],
        }
        with self.assertRaises(ValueError):
            validate_upstream_inputs(encoded, self.reference)

        encoded = {
            "ids": self.reference["tokenIds"],
            "opt_idx": [[*self.reference["optionIndices"], 21]],
        }
        with self.assertRaises(ValueError):
            validate_upstream_inputs(encoded, self.reference)

    def test_q4_candidate_uses_the_existing_reference_thresholds(self):
        actual = np.array([4.551493167877197, -0.8641834855079651], dtype=np.float32)

        comparison = compare_upstream_output(actual, self.reference, np)

        self.assertLessEqual(
            comparison["maxAbsLogitDelta"],
            self.reference["maxAbsoluteLogitDifference"],
        )
        self.assertLessEqual(
            comparison["maxAbsProbabilityDelta"],
            self.reference["maxAbsoluteProbabilityDifference"],
        )
        self.assertTrue(comparison["selectedOptionMatches"])

    def test_non_finite_or_mismatched_reference_outputs_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            compare_upstream_output(np.array([1.0]), self.reference, np)

        with self.assertRaisesRegex(ValueError, "non-finite"):
            compare_upstream_output(
                np.array([np.nan, 0.0], dtype=np.float32), self.reference, np
            )

    def test_expanded_q4_fixture_is_hash_and_length_pinned(self):
        root = Path(__file__).resolve().parents[2]
        fixture, digest = load_expanded_inference_fixture(
            root, "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
        )

        self.assertEqual(digest, EXPANDED_INFERENCE_FIXTURE_SHA256)
        self.assertEqual(
            [(case["id"], case["tokenCount"]) for case in fixture["decisionCases"]],
            list(EXPANDED_DECISION_CASES),
        )

    def test_expanded_gpu_gate_records_all_lengths_and_runtime_criteria(self):
        root = Path(__file__).resolve().parents[2]
        fixture, _ = load_expanded_inference_fixture(
            root, "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
        )
        contract = expanded_gpu_gate_contract(
            fixture, EXPANDED_INFERENCE_FIXTURE_SHA256
        )

        self.assertEqual(contract["status"], "pending-real-chrome-webgpu")
        self.assertEqual(
            [(case["id"], case["tokenCount"]) for case in contract["decisionCases"]],
            list(EXPANDED_DECISION_CASES),
        )
        self.assertEqual(contract["maxAbsoluteQ4LogitDifference"], 0.15)
        self.assertEqual(contract["q4ProbabilityComparison"], "descriptive-only")
        self.assertTrue(contract["allSelectionsMustMatchPinnedQ4"])
        self.assertTrue(contract["outputsMustBeFinite"])
        self.assertTrue(contract["probabilitiesMustSumToOne"])
        self.assertEqual(contract["maxProbabilitySumError"], 0.000001)
        self.assertTrue(contract["baselineCpuGpuParityRequired"])

    def test_recorded_q4_cpu_run_preserves_historical_converter_provenance(self):
        report_path = (
            Path(__file__).resolve().with_name("reports") / "direct-tvm-q4-cpu-r11.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["onnxIntermediateUsed"])
        self.assertEqual(report["q4"]["quantizedLinearCount"], 196)
        self.assertEqual(report["q4"]["quantizedEmbeddingCount"], 1)
        self.assertEqual(report["quantizedParameterInputCount"], 511)
        self.assertEqual(report["inputDtype"], "torch.int32")
        self.assertEqual(report["attentionMaskDtype"], "torch.int32")
        self.assertTrue(report["selectedOptionIndexMatchesOfficial"])
        self.assertTrue(report["selectedOptionIndexMatchesUpstreamQ4"])
        for name, expected_hash in report["converterHashes"].items():
            if name in HISTORICAL_CONVERTER_SOURCE_HASHES:
                self.assertEqual(
                    expected_hash, HISTORICAL_CONVERTER_SOURCE_HASHES[name], name
                )
                continue
            source_path = Path(__file__).resolve().with_name(name)
            actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
            self.assertEqual(actual_hash, expected_hash, name)

    def test_direct_q4_cpu_r12_records_dynamic_export_and_cpu_gate(self):
        report_path = (
            Path(__file__).resolve().with_name("reports") / "direct-tvm-q4-cpu-r12.json"
        )
        report_bytes = report_path.read_bytes()
        report = json.loads(report_bytes)

        self.assertEqual(
            hashlib.sha256(report_bytes).hexdigest(),
            "29dd122974c8220be889acaa3cb05a2be39facafc83b3cdd8e12c0c831bee8e6",
        )
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["onnxIntermediateUsed"])
        self.assertIn("int_oo", report["torchExportRangeConstraints"])
        self.assertIn("s53", report["relaxOutputType"])
        self.assertEqual(report["inputShape"], [1, 22])
        self.assertTrue(report["selectedOptionIndexMatchesOfficial"])
        self.assertLessEqual(
            report["upstreamQ4ReferenceComparison"]["maxAbsLogitDelta"],
            report["acceptanceThresholds"]["maxCpuLogitDeltaFromPinnedUpstreamQ4"],
        )
        self.assertLessEqual(
            report["upstreamQ4ReferenceComparison"]["maxAbsProbabilityDelta"],
            report["acceptanceThresholds"][
                "maxCpuProbabilityDeltaFromPinnedUpstreamQ4"
            ],
        )

    def test_direct_webgpu_candidate_report_keeps_hardware_gate_pending(self):
        reports = Path(__file__).resolve().with_name("reports")
        report_path = reports / "direct-tvm-q4-webgpu-build-r1.json"
        report_bytes = report_path.read_bytes()
        report = json.loads(report_bytes)
        manifest_path = reports / "direct-tvm-q4-webgpu-manifest-r1.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        model_config_path = reports / "direct-tvm-q4-webgpu-model-config-r1.json"
        model_config_bytes = model_config_path.read_bytes()
        model_config = json.loads(model_config_bytes)

        self.assertEqual(
            hashlib.sha256(report_bytes).hexdigest(),
            "1913aeb513e9eff5b4fdc13b6c5bfb411673391c46fb794b8d9e1fc1c7c2755f",
        )
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["onnxIntermediateUsed"])
        self.assertEqual(report["candidateModelId"], "kev-0.6b-q4f16")
        self.assertEqual(report["parameterInputCount"], 511)
        canonical_manifest = (
            json.dumps(
                manifest,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        self.assertEqual(
            hashlib.sha256(manifest_bytes).hexdigest(),
            "d7ebc0e596e73c57d5ca7ff8fc80c632a8b4256b044f8a599012843bfefb9876",
        )
        self.assertEqual(
            hashlib.sha256(canonical_manifest).hexdigest(),
            report["browserPackage"]["manifestSha256"],
        )
        self.assertEqual(manifest["backend"], "tvm-webgpu")
        self.assertEqual(manifest["parameterCount"], 511)
        self.assertEqual(
            manifest["checkpoint"]["revision"], report["kevCheckpointRevision"]
        )
        self.assertEqual(manifest["base"]["revision"], report["baseRevision"])
        self.assertEqual(manifest["maxStateTokens"], 8192)
        self.assertEqual(manifest["maxBranchTokens"], 8192)
        self.assertEqual(manifest["maxOptions"], 255)
        self.assertEqual(
            hashlib.sha256(model_config_bytes).hexdigest(),
            manifest["files"]["model_config.json"]["sha256"],
        )
        self.assertEqual(model_config["max_position_embeddings"], 32768)
        self.assertEqual(model_config["kev"]["max_state_tokens"], 8192)
        self.assertEqual(model_config["kev"]["max_branch_tokens"], 8192)
        self.assertEqual(model_config["kev"]["max_options"], 255)

        cpu_gate = report["qualityGates"]["cpuBaseline"]
        self.assertEqual(cpu_gate["status"], "passed")
        self.assertEqual(cpu_gate["inputTokenCount"], 22)
        self.assertTrue(cpu_gate["selectedOptionMatches"])
        self.assertLessEqual(
            cpu_gate["maxAbsLogitDelta"],
            report["acceptanceThresholds"]["maxCpuLogitDeltaFromPinnedUpstreamQ4"],
        )
        self.assertLessEqual(
            cpu_gate["maxAbsProbabilityDelta"],
            report["acceptanceThresholds"][
                "maxCpuProbabilityDeltaFromPinnedUpstreamQ4"
            ],
        )
        self.assertEqual(
            report["qualityGates"]["webgpuExpandedMatrix"]["status"],
            "pending-real-chrome-webgpu",
        )
        self.assertEqual(report["browserPackage"]["parameterCount"], 511)

        for name, expected_hash in report["converterHashes"].items():
            if name in HISTORICAL_WEBGPU_R1_SOURCE_HASHES:
                self.assertEqual(
                    expected_hash, HISTORICAL_WEBGPU_R1_SOURCE_HASHES[name], name
                )
                continue
            source_path = Path(__file__).resolve().with_name(name)
            actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
            self.assertEqual(actual_hash, expected_hash, name)

    def test_direct_webgpu_candidate_r2_pins_current_sources_and_package_manifest(self):
        reports = Path(__file__).resolve().with_name("reports")
        report_bytes = (reports / "direct-tvm-q4-webgpu-build-r2.json").read_bytes()
        report = json.loads(report_bytes)
        manifest_bytes = (
            reports / "direct-tvm-q4-webgpu-manifest-r2.json"
        ).read_bytes()
        manifest = json.loads(manifest_bytes)
        model_config_bytes = (
            reports / "direct-tvm-q4-webgpu-model-config-r2.json"
        ).read_bytes()
        model_config = json.loads(model_config_bytes)

        self.assertEqual(
            hashlib.sha256(report_bytes).hexdigest(),
            "59009ed215dde8dfcffac87e38b8a2ad5c1c6396cc1b85b23e87394dc19fb67f",
        )
        self.assertEqual(
            hashlib.sha256(manifest_bytes).hexdigest(),
            "984bbf82059d8c6b261999b598c383bb98746595642c779e1540bd00923ca6e9",
        )
        self.assertEqual(
            hashlib.sha256(model_config_bytes).hexdigest(),
            "7143a4974bc3ba3c2eea3bfb0b69d8b3284d7501ab7beea3312ba58e76b87251",
        )
        canonical_manifest = (
            json.dumps(
                manifest,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        self.assertEqual(
            hashlib.sha256(canonical_manifest).hexdigest(),
            report["browserPackage"]["manifestSha256"],
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["candidateModelId"], "kev-0.6b-q4f16")
        self.assertEqual(report["qualityGates"]["cpuBaseline"]["status"], "passed")
        self.assertEqual(
            report["qualityGates"]["webgpuExpandedMatrix"]["status"],
            "pending-real-chrome-webgpu",
        )
        self.assertEqual(manifest["backend"], "tvm-webgpu")
        self.assertEqual(manifest["parameterCount"], 511)
        self.assertEqual(manifest["maxStateTokens"], 8192)
        self.assertEqual(manifest["maxBranchTokens"], 8192)
        self.assertEqual(model_config["max_position_embeddings"], 32768)

        for name, expected_hash in report["converterHashes"].items():
            source_path = Path(__file__).resolve().with_name(name)
            actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
            self.assertEqual(actual_hash, expected_hash, name)

    def test_converter_uses_the_locked_upstream_tvm_wheel_without_source_patches(self):
        converter_root = Path(__file__).resolve().parent
        project = tomllib.loads(
            (converter_root / "pyproject.toml").read_text(encoding="utf-8")
        )
        lock = tomllib.loads((converter_root / "uv.lock").read_text(encoding="utf-8"))
        dependency = next(
            item
            for item in project["project"]["dependencies"]
            if item.startswith("apache-tvm==")
        )
        locked_tvm = next(
            package for package in lock["package"] if package["name"] == "apache-tvm"
        )
        build_script = (converter_root / "build_tvm.py").read_text(encoding="utf-8")

        self.assertEqual(dependency, "apache-tvm==0.27.0.post1")
        self.assertEqual(locked_tvm["version"], "0.27.0.post1")
        self.assertEqual(locked_tvm["source"]["registry"], "https://pypi.org/simple")
        self.assertTrue(locked_tvm["wheels"])
        for marker in (
            ".toolchains/apache-tvm-src",
            "apply_tvm_webgpu",
            "git apply",
            "patches/tvm",
            "onnx_frontend.",
            "_parse_graph_initializers",
            "_impl_v",
        ):
            with self.subTest(marker=marker):
                self.assertNotIn(marker, build_script)

    def test_tvmjs_export_toolchain_requires_pinned_clean_source_and_libraries(self):
        converter_root = Path(__file__).resolve().parent
        tvm_home = converter_root / "fake-pinned-tvm"
        command_results = [
            SimpleNamespace(stdout=f"{PINNED_TVM_WEB_REVISION}\n"),
            SimpleNamespace(stdout=f"{PINNED_TVM_WEB_TAG}\n"),
            SimpleNamespace(stdout=""),
        ]
        hashes = list(PINNED_TVM_WEB_LIBRARIES.values())

        with (
            patch("build_tvm.subprocess.run", side_effect=command_results) as git_run,
            patch("build_tvm.sha256_file", side_effect=hashes),
        ):
            result = verify_tvmjs_toolchain(tvm_home)

        self.assertEqual(
            result,
            {
                "sourceRevision": PINNED_TVM_WEB_REVISION,
                "sourceTag": PINNED_TVM_WEB_TAG,
                "librarySha256": PINNED_TVM_WEB_LIBRARIES,
            },
        )
        self.assertEqual(git_run.call_count, 3)

    def test_tvmjs_export_toolchain_reports_all_runtime_hash_mismatches(self):
        converter_root = Path(__file__).resolve().parent
        tvm_home = converter_root / "fake-pinned-tvm"
        command_results = [
            SimpleNamespace(stdout=f"{PINNED_TVM_WEB_REVISION}\n"),
            SimpleNamespace(stdout=f"{PINNED_TVM_WEB_TAG}\n"),
            SimpleNamespace(stdout=""),
        ]
        actual_hashes = ["0" * 64, "1" * 64, "2" * 64]

        with (
            patch("build_tvm.subprocess.run", side_effect=command_results),
            patch("build_tvm.sha256_file", side_effect=actual_hashes),
        ):
            with self.assertRaises(ValueError) as error:
                verify_tvmjs_toolchain(tvm_home)

        for (path, _), actual_hash in zip(
            PINNED_TVM_WEB_LIBRARIES.items(), actual_hashes, strict=True
        ):
            with self.subTest(path=path):
                self.assertIn(path, str(error.exception))
                self.assertIn(actual_hash, str(error.exception))

    def test_emcc_libatomic_dependency_is_version_and_hash_pinned(self):
        with tempfile.TemporaryDirectory() as temporary:
            library = Path(temporary) / "libatomic.so.1"
            library.write_bytes(b"pinned libatomic test fixture")
            with patch("build_tvm.sha256_file", return_value=PINNED_LIBATOMIC_SHA256):
                result = verify_libatomic(library)

        self.assertEqual(
            result,
            {
                "package": "libatomic1",
                "packageVersion": PINNED_LIBATOMIC_PACKAGE_VERSION,
                "soname": "libatomic.so.1",
                "librarySha256": PINNED_LIBATOMIC_SHA256,
            },
        )

    def test_libatomic_path_keeps_the_active_python_runtime_first(self):
        self.assertEqual(
            extend_runtime_library_path(
                "/opt/python/lib:/opt/other-runtime/lib", Path("/usr/lib")
            ),
            "/opt/python/lib:/opt/other-runtime/lib:/usr/lib",
        )


if __name__ == "__main__":
    unittest.main()
