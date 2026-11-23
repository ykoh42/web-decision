"""CPU checks for model-specific Relax operation partitioning."""

import unittest

import numpy as np
import tvm
import tvm_ffi
from model_partition import partition_model
from tvm import relax

SOURCE = """
s = TypeVar("s")
@I.ir_module
class Module:
    @R.function
    def float_main(x: R.Tensor((s, 2), dtype="float32"), y: R.Tensor((s, 2), dtype="float32")) -> R.Tensor((s, 2), dtype="float32"):
        with R.dataflow():
            a: R.Tensor((s, 2), dtype="float32") = R.add(x, y)
            b: R.Tensor((s, 2), dtype="float32") = R.multiply(a, y)
            gv: R.Tensor((s, 2), dtype="float32") = b
            R.output(gv)
        return gv

    @R.function
    def packed_main(x: R.Tensor((s, 2), dtype="float32"), y: R.Tensor((s, 2), dtype="float32")) -> R.Tensor((s, 2), dtype="float32"):
        cls = Module
        return cls.float_main(x, y)
"""


TIR_SOURCE = """
@I.ir_module
class Module:
    @T.prim_func
    def twice(x: T.Buffer((4,), "float32"), output: T.Buffer((4,), "float32")):
        for i in range(4):
            output[i] = x[i] * T.float32(2)

    @R.function
    def float_main(x: R.Tensor((4,), dtype="float32")) -> R.Tensor((4,), dtype="float32"):
        cls = Module
        with R.dataflow():
            a = R.call_tir(cls.twice, (x,), out_ty=R.Tensor((4,), dtype="float32"))
            gv: R.Tensor((4,), dtype="float32") = R.add(a, x)
            R.output(gv)
        return gv
"""


class ModelPartitionTests(unittest.TestCase):
    def test_float_graph_keeps_entry_and_cpu_results(self):
        original = tvm.script.from_source(SOURCE)
        partitioned = partition_model(original, "float_main")
        self.assertTrue(
            tvm_ffi.structural_equal(
                original["packed_main"], partitioned["packed_main"]
            )
        )
        self.assertTrue(
            tvm_ffi.structural_equal(
                original["float_main"].ty, partitioned["float_main"].ty
            )
        )
        with tvm.transform.PassContext(
            disabled_pass=["StaticPlanBlockMemory", "FuseOps"]
        ):
            vm = relax.VirtualMachine(
                relax.build(partitioned, target="llvm"), tvm.cpu()
            )
        for length in (1, 2, 7):
            x = np.arange(length * 2, dtype=np.float32).reshape(length, 2)
            y = np.full_like(x, 0.5)
            actual = vm["packed_main"](
                tvm.runtime.tensor(x), tvm.runtime.tensor(y)
            ).numpy()
            np.testing.assert_array_equal(actual, (x + y) * y)

    def test_original_tir_is_reattached_before_validation(self):
        original = tvm.script.from_source(TIR_SOURCE)
        partitioned = partition_model(original, "float_main")
        self.assertTrue(
            tvm_ffi.structural_equal(original["twice"], partitioned["twice"])
        )
        with tvm.transform.PassContext(
            disabled_pass=["StaticPlanBlockMemory", "FuseOps"]
        ):
            vm = relax.VirtualMachine(
                relax.build(partitioned, target="llvm"), tvm.cpu()
            )
        data = np.arange(4, dtype=np.float32)
        actual = vm["float_main"](tvm.runtime.tensor(data)).numpy()
        np.testing.assert_array_equal(actual, data * 3)

    def test_missing_target_fails(self):
        with self.assertRaisesRegex(ValueError, "Missing model function"):
            partition_model(tvm.script.from_source(SOURCE), "missing")


if __name__ == "__main__":
    unittest.main()
