"""Tests for the TVM build report emitted by Kanana exporters."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import build_evidence
except ModuleNotFoundError:
    from scripts.kanana import build_evidence

add_tvm_build_evidence = build_evidence.add_tvm_build_evidence


class TvmBuildEvidenceTests(unittest.TestCase):
    def test_accepts_only_the_locked_python_packages(self):
        expected = {"apache-tvm": "0.27.0.post1", "torch": "2.8.0"}

        packages = build_evidence.verify_locked_python_packages(
            expected,
            {"apache-tvm": "0.27.0.post1", "torch": "2.8.0"},
        )

        self.assertEqual(packages, expected)

    def test_rejects_python_packages_that_differ_from_the_lock(self):
        with self.assertRaisesRegex(RuntimeError, "missingOrMismatched=.*extra=.*"):
            build_evidence.verify_locked_python_packages(
                {"torch": "2.8.0"},
                {"torch": "2.9.0", "onnx": "1.23.2"},
            )

    def test_records_the_completed_wasm_and_pinned_build_inputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            wasm = root / "kanana-text-model.wasm"
            wasm_bytes = b"compiled wasm"
            wasm.write_bytes(wasm_bytes)

            environment = {"python": {"version": "3.11.11"}, "host": {"build": "test-build"}}
            with patch.object(build_evidence, "collect_build_environment", return_value=environment):
                report = add_tvm_build_evidence(
                    {"mode": "checkpoint_full"},
                    model_id="kanana-2-v-decide-3b-2611",
                    repository="kakaocorp/kanana-2-v-decide-3b-2611",
                    revision="a" * 40,
                    checkpoint_sha256="b" * 64,
                    graph="text",
                    wasm_path=wasm,
                    build_root=root,
                    tvm_version="0.27.0.post1",
                )

        self.assertEqual(report["evidenceSchema"], "web-decision-tvm-build-v1")
        self.assertEqual(report["modelId"], "kanana-2-v-decide-3b-2611")
        self.assertEqual(report["repository"], "kakaocorp/kanana-2-v-decide-3b-2611")
        self.assertEqual(report["wasm"], "kanana-text-model.wasm")
        self.assertEqual(report["wasmBytes"], len(wasm_bytes))
        self.assertEqual(report["wasmSha256"], hashlib.sha256(wasm_bytes).hexdigest())
        self.assertEqual(report["tvmVersion"], "0.27.0.post1")
        self.assertEqual(report["buildEnvironment"], environment)
        self.assertFalse(report["embeddedBuildRoot"])
        self.assertFalse(report["onnxImportOrExportUsed"])

    def test_rejects_a_wasm_that_embeds_the_build_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            wasm = root / "kanana-vision-model.wasm"
            wasm.write_bytes(str(root.resolve()).encode())

            with self.assertRaisesRegex(RuntimeError, "embeds its build root"):
                add_tvm_build_evidence(
                    {},
                    model_id="kanana-2-v-decide-3b-2611",
                    repository="kakaocorp/kanana-2-v-decide-3b-2611",
                    revision="a" * 40,
                    checkpoint_sha256="b" * 64,
                    graph="vision",
                    wasm_path=wasm,
                    build_root=root,
                    tvm_version="0.27.0.post1",
                )

    def test_rejects_a_wasm_outside_its_build_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            build_root = root / "build"
            build_root.mkdir()
            wasm = root / "kanana-text-model.wasm"
            wasm.write_bytes(b"compiled wasm")

            with self.assertRaises(ValueError):
                add_tvm_build_evidence(
                    {},
                    model_id="kanana-2-v-decide-3b-2611",
                    repository="kakaocorp/kanana-2-v-decide-3b-2611",
                    revision="a" * 40,
                    checkpoint_sha256="b" * 64,
                    graph="text",
                    wasm_path=wasm,
                    build_root=build_root,
                    tvm_version="0.27.0.post1",
                )


if __name__ == "__main__":
    unittest.main()
