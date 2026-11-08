"""Exercise q4 byte verification and safe report handling."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import verify_q4_cache as verifier


def tensor(name, offset, length):
    entries = {"location": "source.bin", "offset": str(offset), "length": str(length)}
    return SimpleNamespace(
        name=name,
        external_data=[
            SimpleNamespace(key=key, value=value) for key, value in entries.items()
        ],
    )


class VerifyQ4CacheTests(unittest.TestCase):
    def test_tensor_ranges_match_and_report_mismatches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            cache = root / "cache"
            source.mkdir()
            cache.mkdir()
            (source / "source.bin").write_bytes(b"abcd")
            (cache / "weights.bin").write_bytes(b"abXd")
            tensors = [tensor("q4_packed_0", 0, 2), tensor("q4_scale_1", 2, 2)]
            records = {
                "param_0": (
                    "weights.bin",
                    {"nbytes": 2, "format": "raw", "byteOffset": 0},
                ),
                "param_1": (
                    "weights.bin",
                    {"nbytes": 2, "format": "raw", "byteOffset": 2},
                ),
            }
            total, mismatches = verifier.verify_tensors(tensors, records, source, cache)
            self.assertEqual(total, 4)
            self.assertEqual(mismatches, [{"param": 1, "source": "q4_scale_1"}])
            (cache / "weights.bin").write_bytes(b"abcd")
            self.assertEqual(
                verifier.verify_tensors(tensors, records, source, cache), (4, [])
            )
            records["param_1"][1]["format"] = "wrong"
            self.assertEqual(
                len(verifier.verify_tensors(tensors, records, source, cache)[1]), 1
            )

    def test_truncated_tensor_range_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tensor.bin"
            path.write_bytes(b"ab")
            with self.assertRaisesRegex(ValueError, "unexpected end"):
                verifier.digest_part(path, 0, 3)

    def test_check_only_does_not_create_report_and_rejects_mismatched_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "models/kev-0.6b-onnx/onnx"
            cache = root / "models/tvm-kev-q4-seq69"
            source.mkdir(parents=True)
            cache.mkdir(parents=True)
            (source / "source.bin").write_bytes(b"abcd")
            (cache / "weights.bin").write_bytes(b"abcd")
            records = [
                {
                    "name": f"param_{index}",
                    "nbytes": 2,
                    "format": "raw",
                    "byteOffset": index * 2,
                }
                for index in range(2)
            ]
            manifest = {
                "metadata": {"numParams": 2},
                "records": [{"dataPath": "weights.bin", "records": records}],
            }
            (cache / "tensor-cache.json").write_text(json.dumps(manifest))
            graph = SimpleNamespace(
                initializer=[tensor("q4_packed_0", 0, 2), tensor("q4_scale_1", 2, 2)]
            )
            with (
                patch.object(verifier, "ROOT", root),
                patch.object(verifier, "EXPECTED_PARAMETERS", 2),
                patch.object(
                    verifier.onnx, "load", return_value=SimpleNamespace(graph=graph)
                ),
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                self.assertEqual(verifier.main(["--check-only"]), 0)
                self.assertEqual(json.loads(output.getvalue())["totalBytesChecked"], 4)
                self.assertFalse((root / "reports").exists())
                (cache / "weights.bin").write_bytes(b"abXd")
                self.assertEqual(verifier.main(["--check-only"]), 1)

    def test_existing_report_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "reports").mkdir()
            report = root / "reports/q4-cache-verification.json"
            report.write_text("keep")
            with (
                patch.object(verifier, "ROOT", root),
                patch.object(verifier.onnx, "load") as load,
                self.assertRaisesRegex(SystemExit, "Refusing to overwrite"),
            ):
                verifier.main([])
            self.assertEqual(report.read_text(), "keep")
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
