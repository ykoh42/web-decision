"""Patch-free WebGPU code generation checks for tiled Kev attention."""

import hashlib
import json
import unittest
from pathlib import Path

import tvm
from kev_attention_tiled import _online_attention_row


class KevTiledAttentionTests(unittest.TestCase):
    def test_synthetic_gpu_report_matches_the_prototype_source(self):
        report_path = (
            Path(__file__).resolve().with_name("reports")
            / "kev-tiled-row-webgpu-synthetic-r1.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        source_path = Path(__file__).resolve().with_name("kev_attention_tiled.py")
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()

        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["runtimeIntegrated"])
        self.assertFalse(report["modelWeightsUsed"])
        self.assertEqual(report["source"]["sha256"], source_hash)
        self.assertEqual(
            [case["sequenceLength"] for case in report["cases"]],
            [7, 17, 33, 8192],
        )
        self.assertTrue(
            all(case["maxAbsoluteError"] < 0.001 for case in report["cases"])
        )

    def test_macro_stages_expand_into_one_linear_storage_webgpu_kernel(self):
        schedule = tvm.s_tir.Schedule(_online_attention_row)
        self.assertEqual(len(schedule.mod.functions), 1)

        executable = tvm.tirx.build(schedule.mod, target="webgpu")
        webgpu_module = next(
            module for module in executable.imports if module.kind == "webgpu"
        )
        wgsl = webgpu_module.inspect_source("wgsl")

        self.assertIn("@workgroup_size(128, 1, 1)", wgsl)
        self.assertIn("workgroupBarrier();", wgsl)
        self.assertIn("var<workgroup> state : array<f32, 384>", wgsl)
        self.assertIn("var<workgroup> partial : array<f32, 128>", wgsl)
        self.assertIn("var<workgroup> scores : array<f32, 64>", wgsl)
        self.assertLess(len(wgsl.encode("utf-8")), 10_000)


if __name__ == "__main__":
    unittest.main()
