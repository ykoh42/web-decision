"""Check release evidence refuses bytes different from the packaged manifest."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import build_release_tvm


class BuildEvidenceTest(unittest.TestCase):
    def test_records_only_the_expected_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = root / "assets"
            output = root / "build"
            assets.mkdir()
            (output / "text").mkdir(parents=True)
            wasm = output / "text/text.wasm"
            wasm.write_bytes(b"independent wasm")
            pin = {
                "bytes": wasm.stat().st_size,
                "sha256": hashlib.sha256(wasm.read_bytes()).hexdigest(),
            }
            (assets / "manifest.json").write_text(
                json.dumps({"files": {"text.wasm": pin}})
            )
            environment = {"pythonPackage": {"version": "0.27.0.post1"}}
            with patch.object(build_release_tvm, "ASSETS", assets):
                entry = build_release_tvm.record_build(output, "text.wasm", environment)
                report = json.loads(
                    (output / entry["buildRoot"] / entry["report"]).read_text()
                )
                self.assertEqual(report["wasmSha256"], pin["sha256"])
                self.assertEqual(report["wasm"], "text.wasm")
                wasm.write_bytes(b"different wasm")
                with self.assertRaisesRegex(ValueError, "differs from the manifest"):
                    build_release_tvm.record_build(output, "text.wasm", environment)

    def test_rejects_a_nonempty_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "existing").touch()
            with self.assertRaisesRegex(ValueError, "must be empty"):
                build_release_tvm.build(output)


if __name__ == "__main__":
    unittest.main()
