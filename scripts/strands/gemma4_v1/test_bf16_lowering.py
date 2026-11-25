"""Check the portable BF16 bit operations without checkpoint weights."""

import unittest

import numpy as np
import torch
import tvm
from bf16_lowering import norm_mean_compute, round_compute, unpack_compute
from tvm import relax


def evaluate(compute, source, *arguments):
    value = relax.Var("value", relax.TensorType(source.shape, source.dtype.name))
    builder = relax.BlockBuilder()
    with builder.function("main", [value]):
        with builder.dataflow():
            call = builder.emit_te(compute, value, *arguments)
            output = builder.emit_output(call)
        builder.emit_func_output(output)
    vm = relax.VirtualMachine(relax.build(builder.get(), target="llvm"), tvm.cpu())
    return vm["main"](tvm.runtime.tensor(source)).numpy()


class BF16LoweringTests(unittest.TestCase):
    def test_norm_mean_retains_small_values_after_a_large_element(self):
        source = np.zeros((2, 512), dtype=np.float32)
        source[0, 0] = source[1, -1] = 1
        source[0, 32:] = source[1, :-32] = 2**-24
        expected = source.mean(axis=-1, keepdims=True, dtype=np.float64).astype(
            np.float32
        )
        np.testing.assert_array_equal(evaluate(norm_mean_compute, source), expected)

    def test_rounds_finite_midpoints_to_even(self):
        source = np.array(
            [
                0x3F807FFF,
                0x3F808000,
                0x3F808001,
                0x3F818000,
                0xBF818000,
                0x00008000,
                0x00018000,
                0x7F800000,
                0xFF800000,
                0x80000000,
            ],
            dtype=np.uint32,
        ).view(np.float32)
        expected = torch.from_numpy(source).bfloat16().float().numpy()
        actual = evaluate(round_compute, source)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_unpacks_both_halves_and_discards_padding(self):
        source = np.array([0xBF813F80, 0x00004000], dtype=np.uint32)
        expected = np.array([[0x3F800000, 0xBF810000, 0x40000000]], dtype=np.uint32)
        np.testing.assert_array_equal(
            evaluate(unpack_compute, source, [1, 3]).view(np.uint32), expected
        )


if __name__ == "__main__":
    unittest.main()
