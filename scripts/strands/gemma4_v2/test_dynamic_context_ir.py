"""Check symbolic context lowering without allocating or running model tensors."""

import unittest

import tvm
from image_embedding_ir import build_image_embedding_ir
from rotary_ir import build as build_rotary
from tvm import relax
from word_embedding_ir import build as build_word


def webgpu_source(module, target):
    with tvm.transform.PassContext(disabled_pass=["StaticPlanBlockMemory", "FuseOps"]):
        executable = relax.build(module, target=target)
    pending = [executable.mod]
    shaders = []
    while pending:
        current = pending.pop()
        if current.kind == "webgpu":
            shaders.append(current.inspect_source())
        pending.extend(current.imports)
    return "\n".join(shaders)


class DynamicContextIrTest(unittest.TestCase):
    def test_dynamic_context_does_not_specialize_embedding_shaders(self):
        target = tvm.target.Target(
            "webgpu", host={"kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"}
        )
        for build, dimension, entries in (
            (build_word, "length", ("packed_main",)),
            (
                build_rotary,
                "length",
                ("local_cos", "local_sin", "global_cos", "global_sin"),
            ),
            (build_image_embedding_ir, "n", ("main",)),
        ):
            with self.subTest(module=build.__module__):
                module = build()
                pinned = webgpu_source(module, target)
                for name in entries:
                    func = module[name]
                    sequence = func.ret_ty.shape.values[1]
                    self.assertIsInstance(sequence, tvm.tirx.Var)
                    self.assertEqual(sequence.name, dimension)
                    bounds = dict(func.attrs.get("tir_var_upper_bound", {}))
                    self.assertNotIn(dimension, bounds)
                    bounds[dimension] = 131073
                    module.update_func(
                        module.get_global_var(name),
                        func.with_attr("tir_var_upper_bound", bounds),
                    )
                self.assertEqual(pinned.count("@compute"), len(entries))
                self.assertIn("podArgs.n", pinned)
                self.assertEqual(webgpu_source(module, target), pinned)


if __name__ == "__main__":
    unittest.main()
