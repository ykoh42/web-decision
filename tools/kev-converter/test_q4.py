"""Small deterministic checks for Kev's direct TVM Q4 operators."""

import unittest

import numpy as np
import tvm
from q4 import (
    dequantize_matmul_rows,
    dequantize_rows,
    gather_te,
    matmul_te,
    pack_nibbles,
    pack_offset_nibbles,
    quantize_matmul_rows,
    quantize_rows,
    unpack_nibbles,
    unpack_offset_nibbles,
)
from tvm import te


class Q4OperatorTests(unittest.TestCase):
    def test_signed_nibble_round_trip(self):
        values = np.tile(np.array([-8, -1, 0, 7], dtype=np.int8), (2, 8))
        np.testing.assert_array_equal(unpack_nibbles(pack_nibbles(values)), values)

    def test_offset_nibble_round_trip(self):
        values = np.tile(np.array([0, 7, 8, 15], dtype=np.uint8), (2, 8))
        np.testing.assert_array_equal(
            unpack_offset_nibbles(pack_offset_nibbles(values)), values
        )

    def test_quantization_error_is_bounded_by_half_a_scale(self):
        rng = np.random.default_rng(20261110)
        values = rng.normal(0, 0.4, (5, 32)).astype(np.float16)
        packed, scales = quantize_rows(values)
        reconstructed = dequantize_rows(packed, scales)
        error = np.max(np.abs(values.astype(np.float32) - reconstructed))
        self.assertLessEqual(error, float(np.max(np.abs(scales))) + 1e-3)

    def test_matmul_quantization_uses_offset_eight_and_divide_by_eight(self):
        values = np.zeros((1, 32), dtype=np.float16)
        values[0, :4] = np.array([-8, -1, 0, 7], dtype=np.float16)
        packed, scales = quantize_matmul_rows(values)
        codes = unpack_offset_nibbles(packed)

        self.assertEqual(float(scales[0, 0]), 1.0)
        np.testing.assert_array_equal(
            codes[0, :4], np.array([0, 7, 8, 15], dtype=np.uint8)
        )
        np.testing.assert_array_equal(
            dequantize_matmul_rows(packed, scales)[0, :4],
            np.array([-8, -1, 0, 7], dtype=np.float32),
        )

    def test_matmul_scale_keeps_the_dominant_extreme_sign(self):
        values = np.full((1, 32), 8, dtype=np.float16)
        packed, scales = quantize_matmul_rows(values)

        self.assertEqual(float(scales[0, 0]), -1.0)
        np.testing.assert_array_equal(
            dequantize_matmul_rows(packed, scales),
            np.full((1, 32), 8, dtype=np.float32),
        )

    def test_matmul_rounding_matches_roundf_at_half_steps(self):
        values = np.zeros((1, 32), dtype=np.float16)
        values[0, :4] = np.array([-8, 0.5, 0.25, -0.5], dtype=np.float16)

        packed, _ = quantize_matmul_rows(values)

        np.testing.assert_array_equal(
            unpack_offset_nibbles(packed)[0, :4],
            np.array([0, 9, 8, 8], dtype=np.uint8),
        )

    def test_embedding_quantization_uses_signed_scale_orientation(self):
        values = np.full((1, 32), 8, dtype=np.float16)
        packed, scales = quantize_rows(values)

        self.assertEqual(float(scales[0, 0]), -1.0)
        np.testing.assert_array_equal(
            unpack_nibbles(packed), np.full((1, 32), -8, dtype=np.int8)
        )

    def test_cpu_matmul_and_embedding_gather(self):
        rng = np.random.default_rng(20261111)
        inputs = rng.normal(0, 0.5, (3, 32)).astype(np.float16)
        weights = rng.normal(0, 0.4, (5, 32)).astype(np.float16)
        packed, scales = quantize_matmul_rows(weights)
        decoded = dequantize_matmul_rows(packed, scales)
        x = te.placeholder(inputs.shape, dtype="float16", name="x")
        q = te.placeholder(packed.shape, dtype="uint32", name="packed")
        s = te.placeholder(scales.shape, dtype="float16", name="scales")
        output = matmul_te(x, q, s)
        module = tvm.build(te.create_prim_func([x, q, s, output]), target="llvm")
        device = tvm.cpu()
        result = tvm.runtime.empty((3, 5), dtype="float32", device=device)
        module(
            tvm.runtime.tensor(inputs, device=device),
            tvm.runtime.tensor(packed, device=device),
            tvm.runtime.tensor(scales, device=device),
            result,
        )
        np.testing.assert_allclose(
            result.numpy(), inputs.astype(np.float32) @ decoded.T, atol=1e-4, rtol=1e-4
        )

        embedding = rng.normal(0, 0.4, (6, 32)).astype(np.float16)
        packed_embedding, embedding_scales = quantize_rows(embedding)
        token_ids = np.array([1, 4, 2], dtype=np.int64)
        p = te.placeholder(
            packed_embedding.shape, dtype="uint32", name="embedding_packed"
        )
        ids = te.placeholder(token_ids.shape, dtype="int64", name="token_ids")
        es = te.placeholder(
            embedding_scales.shape, dtype="float16", name="embedding_scales"
        )
        gathered = gather_te(p, ids, es)
        gather_module = tvm.build(
            te.create_prim_func([p, ids, es, gathered]), target="llvm"
        )
        output = tvm.runtime.empty((3, 32), dtype="float16", device=device)
        gather_module(
            tvm.runtime.tensor(packed_embedding, device=device),
            tvm.runtime.tensor(token_ids, device=device),
            tvm.runtime.tensor(embedding_scales, device=device),
            output,
        )
        expected = dequantize_rows(packed_embedding, embedding_scales)[
            token_ids
        ].astype(np.float16)
        np.testing.assert_allclose(output.numpy(), expected, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
