import unittest

from build_report import raise_for_failed_build, record_wasm_export_failure


class BuildReportTest(unittest.TestCase):
    def test_wasm_export_failure_marks_build_failed_and_returns_nonzero(self):
        report = {"status": "compiled"}

        record_wasm_export_failure(report, RuntimeError("emcc failed"))

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["wasmExportStatus"], "unavailable")
        self.assertEqual(report["wasmExportErrorType"], "RuntimeError")
        self.assertEqual(report["wasmExportError"], "emcc failed")
        with self.assertRaises(SystemExit) as error:
            raise_for_failed_build(report)
        self.assertEqual(error.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
