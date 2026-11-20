import unittest

import tvm
from kanana_shape_heap import (
    PUBLISHED_PRODUCTS,
    TIR_STORE_ORDER,
    canonicalize_shape_heap,
    slot,
)
from tvm import relax, tirx


def make_host_function(graph, *, invalid_slot=None):
    expected = PUBLISHED_PRODUCTS[graph]
    extent = max(product.destination for product in expected) + 1
    source = [
        f'@T.prim_func(s_tir=True)\ndef shape_func(H: T.Buffer(({extent},), "int64")):',
        '    T.func_attr({"tirx.is_host_func": True})',
    ]
    for item in reversed(expected):
        read_slot = (
            item.source if item.destination != invalid_slot else item.destination
        )
        value = " * ".join([f"H[{read_slot}]"] * item.degree)
        source.append(
            f"    H[{item.destination}] = {value} * T.int64({item.coefficient})"
        )
    return tvm.script.from_source("\n".join(source), extra_vars={"T": tvm.script.tirx})


class KananaShapeHeapTests(unittest.TestCase):
    def test_reorders_only_the_original_stores(self):
        for graph, expected in TIR_STORE_ORDER.items():
            with self.subTest(graph=graph):
                func = make_host_function(graph)
                original = {slot(store.indices): store for store in func.body.seq}
                module = tvm.IRModule({"shape_func": func})
                result = canonicalize_shape_heap(graph)(module)["shape_func"]
                self.assertEqual(
                    [slot(store.indices) for store in result.body.seq],
                    list(expected),
                )
                for store in result.body.seq:
                    self.assertTrue(store.same_as(original[slot(store.indices)]))

    def test_rejects_a_store_that_reads_a_written_slot(self):
        func = make_host_function("vision", invalid_slot=20)
        with self.assertRaisesRegex(ValueError, "product at slot 20"):
            canonicalize_shape_heap("vision")(tvm.IRModule({"shape_func": func}))

    def test_webgpu_default_pipeline_runs_before_our_pass(self):
        source = """
@I.ir_module
class Toy:
    @R.function
    def main(x: R.Tensor(["n"], "float32")) -> R.Shape(ndim=2):
        n = T.int64()
        return R.shape([n * 2, n * 3])
"""
        toy = tvm.script.from_source(
            source,
            extra_vars={
                "I": tvm.script.ir,
                "R": tvm.script.relax,
                "T": tvm.script.tirx,
            },
        )
        observed = []

        @tvm.transform.module_pass(opt_level=0)
        def after_lowering(mod, _ctx):
            shape_func = mod["shape_func"]
            self.assertIsInstance(shape_func, tirx.PrimFunc)
            self.assertTrue(shape_func.attrs.get("tirx.is_host_func"))
            self.assertTrue(shape_func.attrs.get("global_symbol"))
            observed.append(shape_func)
            return mod

        target = tvm.target.Target(
            "webgpu", host={"kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"}
        )
        pipeline = tvm.transform.Sequential(
            [relax.get_default_pipeline(target), after_lowering]
        )
        with target:
            pipeline(toy)
        self.assertEqual(len(observed), 1)


if __name__ == "__main__":
    unittest.main()
