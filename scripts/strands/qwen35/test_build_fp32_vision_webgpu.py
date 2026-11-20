"""Check the FP32 source-precision boundary against the original scheduled Qwen vision graph."""

import unittest

import tvm
from build_fp32_vision_webgpu import ROUND_KERNEL, fp32_boundary, scheduled_vision
from qwen35_vm_shape_order import order_vm_shape_stores


class VisionPrecisionTests(unittest.TestCase):
    def test_fp32_boundary_preserves_the_original_launch_and_guards(self):
        target = tvm.target.Target(
            "webgpu", host={"kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"}
        )
        module = scheduled_vision(target)
        original = module[ROUND_KERNEL]
        candidate = fp32_boundary(original)
        guard = "if value_index < sequence:"
        original_prefix, _, rounding = original.script().partition(guard)
        candidate_prefix, _, copy = candidate.script().partition(guard)
        self.assertEqual(candidate_prefix, original_prefix)
        self.assertIn('T.reinterpret("uint32", input_buffer[value_index])', rounding)
        self.assertEqual(
            copy.strip(), "output_buffer[value_index] = input_buffer[value_index]"
        )
        self.assertTrue(module[ROUND_KERNEL].same_as(original))
        self.assertEqual(module["main"].script().count(f"{ROUND_KERNEL}("), 296)
        self.assertEqual(module["main"].script().count("attention_single_segment("), 24)
        original_stores = {
            int(store.indices[0]): store for store in module["shape_func"].body.seq
        }
        order_vm_shape_stores(module)
        ordered_stores = module["shape_func"].body.seq
        self.assertEqual(
            [int(store.indices[0]) for store in ordered_stores],
            [0, 2, 3, 4, 5, 6, 7, 8, 9],
        )
        for store in ordered_stores:
            self.assertTrue(store.same_as(original_stores[int(store.indices[0])]))


if __name__ == "__main__":
    unittest.main()
