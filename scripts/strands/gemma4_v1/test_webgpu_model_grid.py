"""Check model-local launch bounds with independent and shared WebGPU workgroups."""

import sys
import unittest
from pathlib import Path

import tvm
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T
from webgpu_model_grid import needs_guard, verify_and_guard_grids

sys.path.append(str(Path(__file__).resolve().parents[1] / "gemma4_v2"))


@T.prim_func(s_tir=True)
def copy_rows(source: T.handle, result: T.handle):
    count = T.int32()
    left = T.match_buffer(source, (count, 256), "float32")
    right = T.match_buffer(result, (count, 256), "float32")
    for row in T.thread_binding(count, thread="blockIdx.x"):
        for lane in T.thread_binding(256, thread="threadIdx.x"):
            right[row, lane] = left[row, lane]


@I.ir_module
class DecoderAttention:
    @R.function
    def global_attention_main(
        query: R.Tensor((1, 8, "q", 512), "float32"),
        key: R.Tensor((1, 1, "k", 512), "float32"),
        values: R.Tensor((1, 8, "q", "k"), "float32"),
    ) -> R.Tensor((1, 8, "q", "k"), "float32"):
        return R.nn.softmax(values, axis=-1)


class WebGPUGridTest(unittest.TestCase):
    def test_only_guard_when_an_independent_grid_can_exceed_the_device_axis(self):
        self.assertFalse(needs_guard(copy_rows, {"count": 65535}))
        self.assertTrue(needs_guard(copy_rows, {"count": 65536}))
        with self.assertRaisesRegex(ValueError, "Missing Gemma"):
            needs_guard(copy_rows, {})

    def test_exact_attention_dimensions_preserve_workgroup_synchronization(self):
        target = tvm.target.Target("webgpu", host="llvm")
        with (
            target,
            tvm.transform.PassContext(
                disabled_pass=["StaticPlanBlockMemory", "FuseOps"]
            ),
        ):
            lowered = relax.get_default_pipeline(target)(DecoderAttention)
            transformed = verify_and_guard_grids(lowered, 131072)
            code = relax.build(transformed, target=target, relax_pipeline=None)
        shader = next(
            part
            for host in code.mod.imports
            for part in host.imports
            if part.kind == "webgpu"
        )
        self.assertIn("workgroupBarrier();", shader.inspect_source())


if __name__ == "__main__":
    unittest.main()
