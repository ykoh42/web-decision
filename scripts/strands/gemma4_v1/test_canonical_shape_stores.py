"""Only proven independent shape stores may be reordered."""

import unittest

import numpy as np
import tvm
import tvm_ffi
from canonical_shape_stores import canonical_stores
from tvm import relax
from tvm.script import tirx as T


@T.prim_func
def independent(slots: T.Buffer((4,), "int64")):
    T.func_attr({"tirx.is_host_func": True, "global_symbol": "shape_func"})
    slots[2] = slots[0] * 4
    slots[1] = slots[0] * 2


@T.prim_func
def dependent(slots: T.Buffer((4,), "int64")):
    T.func_attr({"tirx.is_host_func": True, "global_symbol": "shape_func"})
    slots[2] = slots[0] * 4
    slots[1] = slots[2] * 2


class CanonicalShapeTests(unittest.TestCase):
    def changed(self, func):
        transformed = canonical_stores(tvm.IRModule({"shape_func": func}))["shape_func"]
        return not tvm_ffi.structural_equal(func, transformed)

    def test_independent_stores_reorder_but_dependencies_do_not(self):
        self.assertTrue(self.changed(independent))
        self.assertFalse(self.changed(dependent))

    def test_non_host_and_other_functions_remain_untouched(self):
        self.assertFalse(
            self.changed(independent.with_attr("tirx.is_host_func", False))
        )
        self.assertFalse(self.changed(independent.with_attr("global_symbol", "other")))

    def test_dynamic_model_without_gpu_preserves_results(self):
        tokens = tvm.tirx.Var("tokens", "int64")
        data = relax.Var("data", relax.TensorType((tokens, 8), "float32"))
        weights = [
            relax.Var(f"weight_{i}", relax.TensorType((8, n), "float32"))
            for i, n in enumerate((13, 17))
        ]
        builder = relax.BlockBuilder()
        with builder.function("main", [data, *weights]):
            with builder.dataflow():
                output = builder.emit_output(
                    relax.Tuple(
                        [
                            builder.emit(relax.op.matmul(data, weight))
                            for weight in weights
                        ]
                    )
                )
            builder.emit_func_output(output)
        default = relax.get_default_pipeline(tvm.target.Target("llvm"))

        @tvm.transform.module_pass(opt_level=0)
        def pipeline(module, _context):
            return canonical_stores(default(module))

        executable = relax.build(builder.get(), target="llvm", relax_pipeline=pipeline)
        vm = relax.VirtualMachine(executable, tvm.cpu())
        arrays = [np.ones((8, width), dtype=np.float32) for width in (13, 17)]
        for length in (1, 2, 31, 255):
            value = np.arange(length * 8, dtype=np.float32).reshape(length, 8)
            actual = vm["main"](
                tvm.runtime.tensor(value),
                *(tvm.runtime.tensor(weight) for weight in arrays),
            )
            for result, weight in zip(actual, arrays, strict=True):
                np.testing.assert_array_equal(result.numpy(), value @ weight)


if __name__ == "__main__":
    unittest.main()
