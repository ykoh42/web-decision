"""Check prefix admission, GDN chunk boundaries and FP32 cumulative decay."""

import hashlib
import re
import sys
import unittest
from pathlib import Path
from typing import TypeVar

import numpy as np
import tvm
import tvm_ffi
from tvm import relax
from tvm.script import tirx as T
from tvm.tirx.layout import Axis

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_gdn_prefix_candidate import (  # noqa: E402
    candidate_module,
    candidate_script,
    fp32_activations,
)
from qwen35_gdn_cumulative_decay import qwen35_gdn_cumsum64  # noqa: E402
from qwen35_tvm_build import load_compiler_input  # noqa: E402
from qwen35_vm_shape_order import order_vm_shape_stores  # noqa: E402


def serial_kernel(func):
    # Each GPU invocation has private state and writes an independent result.
    # Use the same body on LLVM to exercise arithmetic and layouts in CPU CI.
    script, count = re.subn(
        r'T\.thread_binding\(([^\n]+?), thread="(?:blockIdx\.[xy]|threadIdx\.x)"\)',
        r"range(\1)",
        func.script(),
    )
    if count == 0:
        raise ValueError("Expected the candidate WebGPU thread bindings")
    cpu = tvm.script.from_source(
        script, extra_vars={"T": T, "TypeVar": TypeVar, "Axis": Axis}
    )
    return tvm.compile(cpu.with_attr("global_symbol", "main"), target="llvm")["main"]


class PrefixCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod, cls.program_digest = candidate_module()
        cls.fp32, cls.fp32_program_digest = candidate_module("fp32")

    def test_request_and_weight_arguments(self):
        main = self.mod["main"]
        self.assertEqual(
            self.program_digest,
            "36956618199e24078c7b11ca33c42953b09e1ed410d02bfb0d609b17eb116730",
        )
        original_order = "\n".join(
            p.name for p in main.params if p.name != "source_prefix"
        )
        self.assertEqual(
            hashlib.sha256(original_order.encode()).hexdigest(),
            "33002edb71b198a7a3e336a6daf7db952b5d04f97ae015174a705b1ae3943f36",
        )
        self.assertEqual(len(main.params), 704)
        self.assertEqual(int(main.attrs["num_input"]), 7)
        self.assertEqual(main.params[6].name, "source_prefix")
        self.assertEqual(int(main.attrs["tir_var_upper_bound"]["s22"]), 4096)
        self.assertEqual(
            str(main.ret_ty).count('R.Tensor((s0, s86), dtype="float32")'), 2
        )
        self.assertIn("match_cast", main.script())
        with self.assertRaises(ValueError):
            candidate_script("unexpected graph", "")

    def test_graph_rejects_prefix_beyond_sequence(self):
        main = self.mod["main"]
        data, prefix = main.params[0], main.params[6]
        # Execute the guard copied from the actual decoder without its 697 weights.
        body = relax.SeqExpr([main.body.blocks[0]], data)
        probe = relax.Function([data, prefix], body, data.ty, is_pure=False)
        exe = relax.build(tvm.IRModule({"main": probe}), target="llvm")
        vm = relax.VirtualMachine(exe, tvm.cpu())
        data_array = np.zeros((1, 2, 2048), dtype="f4")
        np.testing.assert_array_equal(
            vm["main"](tvm.runtime.tensor(data_array), tvm_ffi.Shape([2])).numpy(),
            data_array,
        )
        with self.assertRaisesRegex(Exception, "match_cast"):
            vm["main"](tvm.runtime.tensor(data_array), tvm_ffi.Shape([3]))

    def test_fp32_activations_preserve_abi_and_gdn_values(self):
        fp32 = self.fp32
        self.assertEqual(
            self.fp32_program_digest,
            "855f56c709297231a76446413b9a0a218f16b0d234fc1b72d75661f90943dbb1",
        )
        self.assertEqual(
            [p.name for p in fp32["main"].params],
            [p.name for p in self.mod["main"].params],
        )
        tvm.ir.assert_structural_equal(
            fp32["main"].ty, self.mod["main"].ty, map_free_vars=True
        )

        # With a one-token prefix the remaining two rows start in the next chunk.
        core = np.random.default_rng(19).normal(size=(1, 16, 2, 64, 128)).astype("f4")
        expected = np.stack(
            (core[:, :, 0, 0], core[:, :, 1, 0], core[:, :, 1, 1]), axis=1
        )
        bits = expected.view("u4")
        bf16 = ((bits + 32767 + ((bits >> 16) & 1)) & 0xFFFF0000).view("f4")
        self.assertTrue(np.any(expected != bf16))
        query = tvm.runtime.tensor(np.zeros_like(expected))
        packed = tvm.runtime.tensor(core)
        for module, target in ((self.mod, bf16), (fp32, expected)):
            unpack = serial_kernel(module["qwen35_gdn_unpack"])
            output = tvm.runtime.tensor(np.empty_like(expected))
            unpack(1, packed, query, output)
            np.testing.assert_array_equal(output.numpy(), target)

    def test_fp32_chunk_alias_uses_existing_prefix_guard_and_bounds(self):
        main = self.fp32["main"]
        self.assertEqual(int(main.attrs["tir_var_lower_bound"]["s_chunks"]), 1)
        self.assertEqual(int(main.attrs["tir_var_upper_bound"]["s_chunks"]), 65)
        shape_block = main.body.blocks[0]
        chunk_shape = shape_block.bindings[-1].var
        body = relax.SeqExpr([shape_block], chunk_shape)
        probe = relax.Function(
            [main.params[0], main.params[6]],
            body,
            relax.ShapeType(ndim=1),
            is_pure=False,
        )
        vm = relax.VirtualMachine(
            relax.build(tvm.IRModule({"main": probe}), target="llvm"), tvm.cpu()
        )
        tensor = tvm.runtime.tensor(np.zeros((1, 4096, 2048), dtype="f4"))
        for prefix in (0, 1, 63, 64, 4095, 4096):
            with self.subTest(prefix=prefix):
                expected = (4096 + (-prefix % 64) + 63) // 64
                actual = vm["main"](tensor, tvm_ffi.Shape([prefix]))
                self.assertEqual(tuple(actual), (expected,))
        with self.assertRaisesRegex(Exception, "match_cast"):
            vm["main"](tensor, tvm_ffi.Shape([4097]))

    def test_fp32_rejects_changed_rounding_sites(self):
        source, _ = load_compiler_input("text", tvm, relax)
        call = "R.call_tir(cls.qwen35_text_bf16_rne_fp32,"
        with self.assertRaisesRegex(ValueError, "replaced 627"):
            fp32_activations(source.replace(call, "R.call_tir(cls.other,", 1))
        with self.assertRaisesRegex(ValueError, "occurrence count"):
            fp32_activations(source.replace("= rounded[0]", "= value[0]", 1))

    def test_beta_layout_around_prefix_and_at_full_context(self):
        pack = serial_kernel(self.mod["qwen35_gdn_chunkify_beta"])
        for length, prefix in (
            (2, 0),
            (63, 1),
            (64, 63),
            (65, 64),
            (397, 285),
            (4096, 4095),
        ):
            with self.subTest(length=length, prefix=prefix):
                gap = -prefix % 64
                chunks = (length + gap + 63) // 64
                v = np.arange(length * 16, dtype="f4").reshape(1, length, 16)
                expect = np.zeros((1, chunks * 64, 16), dtype="f4")
                expect[:, :prefix] = v[:, :prefix]
                expect[:, prefix + gap : length + gap] = v[:, prefix:]
                gpu_layout = expect.transpose(0, 2, 1).reshape(1, 16, chunks, 64)
                out = tvm.runtime.tensor(np.empty(gpu_layout.shape, dtype="f4"))
                pack(prefix, chunks, tvm.runtime.tensor(v), out)
                np.testing.assert_array_equal(out.numpy(), gpu_layout)

    def test_cumulative_rounding(self):
        run = serial_kernel(qwen35_gdn_cumsum64)
        rng = np.random.default_rng(41)
        values = -np.exp2(rng.uniform(-12, 4, (1, 16, 2, 64))).astype("f4")
        values[0, 0, 0] = 0
        values[0, 0, 1] = -1
        values[0, 0, 1, 0] = -(2**24)
        expected = values.astype("f8").cumsum(-1).astype("f4")
        output = tvm.runtime.tensor(np.empty_like(values))
        run(1, 2, tvm.runtime.tensor(values), output)
        np.testing.assert_array_equal(output.numpy().view("u4"), expected.view("u4"))


class ShapeOrderTests(unittest.TestCase):
    def test_independent_shape_stores_preserve_values(self):
        @T.prim_func
        def shape_func(heap: T.Buffer((5,), "int64")):
            T.func_attr({"tirx.is_host_func": True, "global_symbol": "shape_func"})
            heap[4] = T.max(heap[0], heap[1])
            heap[2] = heap[1] + T.int64(1)
            heap[3] = heap[0] * T.int64(64)

        module = tvm.IRModule({"shape_func": shape_func})
        order_vm_shape_stores(module)
        self.assertEqual(
            [int(store.indices[0]) for store in module["shape_func"].body.seq],
            [2, 3, 4],
        )
        run = tvm.tirx.build(module, target="llvm")["shape_func"]
        values = tvm.runtime.tensor(np.array([63, 64, 0, 0, 0], dtype="int64"))
        run(values)
        np.testing.assert_array_equal(values.numpy(), [63, 64, 65, 4032, 64])

    def test_dependent_shape_stores_are_rejected(self):
        @T.prim_func
        def shape_func(heap: T.Buffer((5,), "int64")):
            T.func_attr({"tirx.is_host_func": True})
            heap[2] = heap[1] + T.int64(1)
            heap[4] = heap[2] * T.int64(64)

        with self.assertRaisesRegex(ValueError, "must be independent"):
            order_vm_shape_stores(tvm.IRModule({"shape_func": shape_func}))
