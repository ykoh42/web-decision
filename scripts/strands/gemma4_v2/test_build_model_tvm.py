import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from build_model_tvm import (
    source_checkout,
    verify_build_environment,
    verify_file,
    write_build_evidence,
)

SCRIPT = Path(__file__).with_name("build_model_tvm.py")


def imported_names(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


class BuildModelTest(unittest.TestCase):
    def test_formal_build_does_not_import_probe_modules(self):
        pending = [SCRIPT]
        visited = set()
        while pending:
            path = pending.pop()
            if path in visited:
                continue
            visited.add(path)
            for name in imported_names(path):
                self.assertFalse(
                    name.startswith("probe_"), f"{path.name} imports {name}"
                )
                local = SCRIPT.parent / f"{name}.py"
                if local.is_file():
                    pending.append(local)

    def test_build_environment_rejects_wrong_dependencies(self):
        with patch("build_model_tvm.sys.version_info", (3, 11, 11)):
            with patch("build_model_tvm.version", return_value="not-pinned"):
                with self.assertRaisesRegex(ValueError, "apache-tvm"):
                    verify_build_environment()

    def test_checkpoint_size_and_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint"
            path.write_bytes(b"test")
            digest = hashlib.sha256(b"test").hexdigest()
            verify_file(path, 4, digest)
            with self.assertRaisesRegex(ValueError, "size"):
                verify_file(path, 5, digest)
            with self.assertRaisesRegex(ValueError, "hash"):
                verify_file(path, 4, "0" * 64)
            with self.assertRaisesRegex(ValueError, "missing"):
                verify_file(path.parent / "missing", 4, digest)

    def test_source_checkout_records_uncommitted_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git = ["git", "-C", str(root)]
            subprocess.run([*git, "init", "--quiet"], check=True)
            subprocess.run(
                [
                    *git,
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.invalid",
                    "commit",
                    "--allow-empty",
                    "--no-gpg-sign",
                    "--quiet",
                    "-m",
                    "Initial commit",
                ],
                check=True,
            )
            with patch.dict(os.environ, {"GIT_DIR": str(root / "wrong.git")}):
                clean = source_checkout(root)
            self.assertRegex(clean["head"], r"^[0-9a-f]{40,64}$")
            self.assertFalse(clean["dirty"])
            (root / "working-source.py").write_text("pass\n", encoding="utf-8")
            subprocess.run([*git, "add", "--force", "working-source.py"], check=True)
            self.assertEqual(
                source_checkout(root), {"head": clean["head"], "dirty": True}
            )

    def test_build_evidence_names_and_hashes_the_exported_wasm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wasm = b"\0asm\x01\0\0\0"
            environment = {
                "nativeTvm": {"pythonPackage": {"version": "0.27.0.post1"}},
                "repository": {"before": {"head": "a" * 40, "dirty": True}},
            }
            (root / "layer.wasm").write_bytes(wasm)
            write_build_evidence(root, root, environment)
            report = json.loads((root / "release-build.json").read_text())
            evidence = json.loads((root / "build-evidence.json").read_text())

        self.assertEqual(report["evidenceSchema"], "web-decision-tvm-build-v1")
        self.assertEqual(report["status"], "success")
        self.assertEqual(report["target"], "webgpu / wasm32")
        self.assertEqual(report["modelId"], "strands-decider-e2b-gemma4-v2-2611")
        self.assertEqual(report["artifact"], "graph.wasm")
        self.assertEqual(report["wasm"], "layer.wasm")
        self.assertEqual(report["wasmBytes"], len(wasm))
        self.assertEqual(report["wasmSha256"], hashlib.sha256(wasm).hexdigest())
        self.assertEqual(report["tvmVersion"], "0.27.0.post1")
        self.assertEqual(report["buildBackend"], "native-tvm")
        self.assertFalse(report["embeddedBuildRoot"])
        self.assertFalse(report["onnxImportOrExportUsed"])
        self.assertEqual(report["buildEnvironment"], environment)
        self.assertEqual(
            evidence,
            {
                "schemaVersion": 1,
                "builds": [
                    {
                        "modelId": report["modelId"],
                        "artifact": "graph.wasm",
                        "buildRoot": ".",
                        "wasm": "layer.wasm",
                        "report": "release-build.json",
                    }
                ],
            },
        )

    def test_build_evidence_rejects_an_embedded_build_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "layer.wasm").write_bytes(b"WASM location: " + str(root).encode())
            with self.assertRaisesRegex(ValueError, "embeds its build root"):
                write_build_evidence(root, root, {})
            self.assertFalse((root / "release-build.json").exists())
            self.assertFalse((root / "build-evidence.json").exists())

    def test_cli_requires_all_paths(self):
        missing = subprocess.run(
            [sys.executable, str(SCRIPT)], capture_output=True, text=True
        )
        self.assertEqual(missing.returncode, 2)
        self.assertIn("--checkpoint", missing.stderr)
        help_result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True
        )
        self.assertEqual(help_result.returncode, 0)
        for option in ("--checkpoint", "--tvm-source", "--emsdk", "--output"):
            self.assertIn(option, help_result.stdout)
        with tempfile.TemporaryDirectory() as directory:
            bad_path = Path(directory) / "absent"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--checkpoint",
                    str(bad_path),
                    "--tvm-source",
                    directory,
                    "--emsdk",
                    directory,
                    "--output",
                    directory,
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("checkpoint must be an existing directory", result.stderr)


if __name__ == "__main__":
    unittest.main()
