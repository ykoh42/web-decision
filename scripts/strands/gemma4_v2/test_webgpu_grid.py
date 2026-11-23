"""Verify bounded workgroup coordinates in a Gemma-shaped dynamic kernel."""

import unittest

import tvm
import tvm_ffi
from tvm import relax, tirx
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T
from webgpu_grid import GRID_LIMIT, grid_axes, guard_model_grids


@T.prim_func(s_tir=True)
def copy_rows(source: T.handle, result: T.handle):
    count = T.int32()
    left = T.match_buffer(source, (count, 256), "float32")
    right = T.match_buffer(result, (count, 256), "float32")
    T.func_attr({"tirx.is_scheduled": True, "global_symbol": "copy_rows"})
    for row in T.thread_binding(count, thread="blockIdx.x"):
        for lane in T.thread_binding(256, thread="threadIdx.x"):
            right[row, lane] = left[row, lane]


@I.ir_module
class MeanRows:
    @R.function
    def main(data: R.Tensor((1, "rows", 1536), "float32")) -> R.Tensor(
        (1, "rows"), "float32"
    ):
        result = R.mean(data, axis=[2])
        return result


class WebGPUGridTest(unittest.TestCase):
    def test_dynamic_grid_keeps_tail_guard_in_compiled_shader(self):
        module = guard_model_grids(tvm.IRModule({"copy_rows": copy_rows}))
        body = module["copy_rows"].body
        self.assertEqual(grid_axes(body), {"blockIdx.x"})
        guards = []
        tvm_ffi.structural_walk(body, (tirx.IfThenElse, guards.append))
        self.assertEqual(len(guards), 1)
        self.assertIn("count", str(guards[0].condition))
        self.assertIn(str(GRID_LIMIT), body.script())

        target = tvm.target.Target("webgpu", host="llvm")
        executable = tvm.compile(module, target=target)
        shader = next(part for part in executable.mod.imports if part.kind == "webgpu")
        source = shader.inspect_source()
        self.assertIn("< podArgs.count", source)

    def test_dynamic_grid_preserves_cross_thread_mean(self):
        target = tvm.target.Target("webgpu", host="llvm")
        pipeline = tvm.transform.Sequential(
            [relax.get_default_pipeline(target), guard_model_grids]
        )
        with tvm.transform.PassContext(
            disabled_pass=["StaticPlanBlockMemory", "FuseOps"]
        ):
            executable = relax.build(MeanRows, target=target, relax_pipeline=pipeline)
        shader = next(
            part
            for host in executable.mod.imports
            for part in host.imports
            if part.kind == "webgpu"
        )
        source = shader.inspect_source()
        self.assertIn("< podArgs.rows", source)
        self.assertIn("workgroupBarrier();", source)
        self.assertEqual(source.count("T_divide_ptr["), 1)
        self.assertRegex(
            source, r"if \(i32\(threadIdx\.x\) == 0i\) \{\s+T_divide_ptr\["
        )


if __name__ == "__main__":
    unittest.main()
