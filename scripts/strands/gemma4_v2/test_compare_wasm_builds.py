import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("compare_wasm_builds.py")
WASM = b"\0asm\x01\0\0\0"


class CompareWasmBuildsTest(unittest.TestCase):
    def test_same_different_expected_and_same_file(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.wasm"
            second = Path(directory) / "second.wasm"
            first.write_bytes(WASM)
            second.write_bytes(WASM)

            def run(*args):
                return subprocess.run(
                    [sys.executable, str(SCRIPT), *map(str, args)],
                    capture_output=True,
                    text=True,
                )

            result = run(first, second)
            self.assertEqual(result.returncode, 0)
            report = json.loads(result.stdout)
            self.assertTrue(report["sameBytes"])
            self.assertIsNone(report["expectedMatch"])
            self.assertEqual(
                run(
                    first, second, "--expected-sha256", report["first"]["sha256"]
                ).returncode,
                0,
            )
            self.assertEqual(
                run(first, second, "--expected-sha256", "0" * 64).returncode, 1
            )
            self.assertEqual(
                run(first, second, "--expected-sha256", "invalid").returncode, 2
            )
            self.assertEqual(run(first, first).returncode, 2)
            second.write_bytes(WASM + b"\x00")
            self.assertEqual(run(first, second).returncode, 1)
            second.write_bytes(b"not wasm")
            self.assertEqual(run(first, second).returncode, 2)


if __name__ == "__main__":
    unittest.main()
