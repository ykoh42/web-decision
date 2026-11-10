"""Numerical and codegen checks for the isolated compact attention experiment."""

import re
import unittest

import numpy as np
import tvm
from kev_attention import (
    FINITE_FP16_MASK,
    build_kev_attention_prim_funcs,
    schedule_kev_attention_for_webgpu,
)

SCALE = np.float32(1.0 / np.sqrt(8.0))


def dense_attention_reference(query, key, value, segments, valid, scale):
    """Reproduce dense finite-mask attention with the current dtype boundaries."""
    sequence_length, num_heads, head_dim = query.shape
    mask_bias = np.float32(np.float16(FINITE_FP16_MASK))
    output = np.zeros((sequence_length, num_heads, head_dim), dtype=np.float32)
    positions = np.arange(sequence_length)
    for query_index in range(sequence_length):
        query_segment = segments[query_index]
        allowed = (
            (positions <= query_index)
            & (valid != 0)
            & ((segments == query_segment) | (segments == 0))
        )
        for head_index in range(num_heads):
            query_vector = query[query_index, head_index].astype(np.float32)
            key_vectors = key[:, head_index].astype(np.float32)
            dots = np.sum(query_vector[None, :] * key_vectors, axis=1, dtype=np.float32)
            scores = np.asarray(dots * scale, dtype=np.float32)
            logits = np.where(allowed, scores, scores + mask_bias).astype(np.float32)
            shifted = logits - np.max(logits)
            exponentials = np.exp(shifted, dtype=np.float32)
            probabilities = exponentials / np.sum(exponentials, dtype=np.float32)
            probabilities = probabilities.astype(np.float16).astype(np.float32)
            output[query_index, head_index] = probabilities @ value[
                :, head_index
            ].astype(np.float32)
    return output


def packed_metadata(sequence_length, padding=3):
    """Build a valid prefix, two contiguous branches, and right padding."""
    prefix_end = min(5, sequence_length - 2)
    active_end = sequence_length - min(padding, sequence_length - prefix_end)
    segments = np.full(sequence_length, -1, dtype=np.int32)
    valid = np.zeros(sequence_length, dtype=np.int32)
    starts = np.zeros(sequence_length, dtype=np.int32)
    segments[:prefix_end] = 0
    valid[:prefix_end] = 1
    first_end = prefix_end + max(1, (active_end - prefix_end) // 2)
    for start, end, segment in (
        (prefix_end, first_end, 1),
        (first_end, active_end, 2),
    ):
        segments[start:end] = segment
        valid[start:end] = 1
        starts[start:end] = start
    return prefix_end, segments, valid, starts


class KevAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = tvm.cpu()
        cls.plan_module, cls.stats_module, cls.output_module = tuple(
            tvm.tirx.build(prim_func, target="llvm")
            for prim_func in build_kev_attention_prim_funcs()
        )

    def run_kernel(
        self, query, key, value, prefix_end, segments, valid, starts, scale=SCALE
    ):
        sequence_length, num_heads, head_dim = query.shape
        output = tvm.runtime.empty(query.shape, dtype="float32", device=self.device)
        query_tensor = tvm.runtime.tensor(query, device=self.device)
        key_tensor = tvm.runtime.tensor(key, device=self.device)
        value_tensor = tvm.runtime.tensor(value, device=self.device)
        segment_tensor = tvm.runtime.tensor(segments, device=self.device)
        valid_tensor = tvm.runtime.tensor(valid, device=self.device)
        starts_tensor = tvm.runtime.tensor(starts, device=self.device)
        plan = tvm.runtime.empty(
            (sequence_length, 2), dtype="int32", device=self.device
        )
        stats = tvm.runtime.empty(
            (sequence_length, num_heads, 2), dtype="float32", device=self.device
        )
        self.plan_module(
            segment_tensor,
            valid_tensor,
            starts_tensor,
            np.int32(prefix_end),
            np.int64(sequence_length),
            plan,
        )
        self.stats_module(
            query_tensor,
            key_tensor,
            segment_tensor,
            valid_tensor,
            starts_tensor,
            plan,
            np.int32(prefix_end),
            scale,
            np.int64(sequence_length),
            np.int64(num_heads),
            np.int64(head_dim),
            stats,
        )
        self.output_module(
            query_tensor,
            key_tensor,
            value_tensor,
            segment_tensor,
            valid_tensor,
            starts_tensor,
            plan,
            stats,
            np.int32(prefix_end),
            scale,
            np.int64(sequence_length),
            np.int64(num_heads),
            np.int64(head_dim),
            output,
        )
        return output.numpy()

    def assert_matches_dense(
        self, query, key, value, prefix_end, segments, valid, starts
    ):
        expected = dense_attention_reference(query, key, value, segments, valid, SCALE)
        actual = self.run_kernel(query, key, value, prefix_end, segments, valid, starts)
        np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)

    def test_dynamic_lengths_match_dense_across_tail_tiles(self):
        rng = np.random.default_rng(20261110)
        for sequence_length in (63, 64, 65, 127, 128, 129, 257):
            with self.subTest(sequence_length=sequence_length):
                prefix_end, segments, valid, starts = packed_metadata(sequence_length)
                shape = (sequence_length, 2, 8)
                query = rng.normal(0, 0.4, shape).astype(np.float16)
                key = rng.normal(0, 0.4, shape).astype(np.float16)
                value = rng.normal(0, 0.4, shape).astype(np.float16)
                self.assert_matches_dense(
                    query, key, value, prefix_end, segments, valid, starts
                )

    def test_internal_padding_and_invalid_queries_keep_prefix_visibility(self):
        rng = np.random.default_rng(20261111)
        sequence_length = 13
        prefix_end = 2
        segments = np.array(
            [0, 0, 1, 1, -1, 1, 2, 2, 2, -1, -1, -1, -1], dtype=np.int32
        )
        valid = (segments >= 0).astype(np.int32)
        starts = np.array([0, 0, 2, 2, 0, 2, 6, 6, 6, 0, 0, 0, 0], np.int32)
        shape = (sequence_length, 2, 8)
        query = rng.normal(0, 0.3, shape).astype(np.float16)
        key = rng.normal(0, 0.3, shape).astype(np.float16)
        value = rng.normal(0, 0.3, shape).astype(np.float16)
        self.assert_matches_dense(
            query, key, value, prefix_end, segments, valid, starts
        )

    def test_all_masked_rows_use_finite_fp16_bias_fallback(self):
        sequence_length = 4
        query = np.ones((sequence_length, 1, 1), dtype=np.float16)
        key = np.arange(sequence_length, dtype=np.float16).reshape(
            sequence_length, 1, 1
        )
        value = (10 * np.arange(sequence_length, dtype=np.float16)).reshape(
            sequence_length, 1, 1
        )
        segments = np.full(sequence_length, -1, dtype=np.int32)
        valid = np.zeros(sequence_length, dtype=np.int32)
        starts = np.zeros(sequence_length, dtype=np.int32)
        scale = np.float32(1.0)
        expected = dense_attention_reference(query, key, value, segments, valid, scale)
        actual = self.run_kernel(query, key, value, 0, segments, valid, starts, scale)
        np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)
        self.assertGreater(float(actual[0, 0, 0]), 20.0)

    def test_webgpu_codegen_has_only_linear_buffers_and_small_workgroups(self):
        symbols = ("kev_attention_plan", "kev_attention_stats", "kev_attention_output")
        for symbol, prim_func in zip(
            symbols, schedule_kev_attention_for_webgpu(), strict=True
        ):
            with self.subTest(function=symbol):
                script = prim_func.script()
                self.assertIn("sequence_length", script)
                self.assertNotRegex(
                    script,
                    r"(?:Buffer|Tensor)\(\([^)]*sequence_length\s*,\s*sequence_length",
                )
                self.assertNotIn("attention_bias", script)
                executable = tvm.tirx.build(prim_func, target="webgpu")
                webgpu_module = next(
                    module for module in executable.imports if module.kind == "webgpu"
                )
                wgsl = webgpu_module.inspect_source("wgsl")
                workgroups = re.findall(
                    r"@workgroup_size\((\d+),\s*(\d+),\s*(\d+)\)", wgsl
                )
                self.assertTrue(workgroups)
                for dimensions in workgroups:
                    self.assertLessEqual(
                        np.prod([int(value) for value in dimensions]), 256
                    )
                self.assertNotIn("score_buffer", wgsl)
                self.assertNotIn("bias_buffer", wgsl)
                self.assertIn("branch_start_by_row", wgsl)
                if symbol == "kev_attention_output":
                    self.assertIn("@workgroup_size(1, 1, 1)", wgsl)


if __name__ == "__main__":
    unittest.main()
