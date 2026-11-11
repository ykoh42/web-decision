"""Shape, dense-reference, and WebGPU codegen checks for attention_grid."""

import unittest

import numpy as np
from kev_attention_grid import (
    ATTENTION_SCALE,
    FINITE_FP16_MASK,
    HEAD_DIM,
    KEY_TILE,
    NEGATIVE_INFINITY,
    NUM_QUERY_HEADS,
    attention_grid,
    build_attention_grid,
)


def _synthetic_inputs(sequence_length, seed, all_invalid=False):
    """Build FP16 Q/K and FP32 V with repeated eight-head GQA inputs."""
    rng = np.random.default_rng(seed)
    query = rng.normal(0.0, 0.12, (NUM_QUERY_HEADS, sequence_length, HEAD_DIM))
    key_heads = rng.normal(0.0, 0.12, (NUM_QUERY_HEADS // 2, HEAD_DIM, sequence_length))
    value_heads = rng.normal(
        0.0, 0.5, (NUM_QUERY_HEADS // 2, sequence_length, HEAD_DIM)
    )
    query = query.astype(np.float16)
    key = np.repeat(key_heads.astype(np.float16), 2, axis=0)
    value = np.repeat(value_heads.astype(np.float32), 2, axis=0)
    segments = np.minimum(
        np.arange(sequence_length, dtype=np.int32) * 3 // max(sequence_length, 1),
        2,
    )
    valid = (np.arange(sequence_length) % 11 != 5).astype(np.int32)
    if sequence_length > 4:
        valid[-1] = 0
    if all_invalid:
        valid.fill(0)
        segments.fill(-1)
    return query, key, value, segments, valid


def _masked_scores(query, key, segments, valid):
    """Compute the dense FP32 score tensor with Kev's finite mask."""
    sequence_length = query.shape[1]
    positions = np.arange(sequence_length)
    visible = (
        (positions[None, :] <= positions[:, None])
        & (valid[None, :] != 0)
        & ((segments[None, :] == segments[:, None]) | (segments[None, :] == 0))
    )
    dots = np.matmul(query.astype(np.float32), key.astype(np.float32))
    bias = np.where(visible, np.float32(0.0), np.float32(FINITE_FP16_MASK))
    return np.asarray(dots * np.float32(ATTENTION_SCALE) + bias[None, :, :], np.float32)


def _probabilities(scores, maximum, denominator):
    """Apply the dense softmax result's explicit FP16 boundary."""
    values = np.exp(scores - maximum, dtype=np.float32) / np.float32(denominator)
    return values.astype(np.float16).astype(np.float32)


def _dense_reference(query, key, value, segments, valid):
    """Compute dense softmax/PV with FP32 accumulation and FP16 probabilities."""
    scores = _masked_scores(query, key, segments, valid)
    maximum = np.max(scores, axis=-1, keepdims=True)
    exponentials = np.exp(scores - maximum, dtype=np.float32)
    denominator = np.sum(exponentials, axis=-1, keepdims=True, dtype=np.float32)
    probabilities = _probabilities(scores, maximum, denominator)
    return np.matmul(probabilities, value.astype(np.float32))


def _tiled_reference(query, key, value, segments, valid):
    """Simulate two-pass tilewise stats and the emitted FP16 P rounding."""
    scores = _masked_scores(query, key, segments, valid)
    output = np.zeros((NUM_QUERY_HEADS, query.shape[1], HEAD_DIM), np.float32)
    for head in range(NUM_QUERY_HEADS):
        for query_index in range(query.shape[1]):
            maximum = np.float32(NEGATIVE_INFINITY)
            denominator = np.float32(0.0)
            for base in range(0, query.shape[1], KEY_TILE):
                tile_scores = scores[head, query_index, base : base + KEY_TILE]
                tile_maximum = max(maximum, np.max(tile_scores))
                tile_denominator = np.sum(
                    np.exp(tile_scores - tile_maximum, dtype=np.float32),
                    dtype=np.float32,
                )
                previous_scale = np.exp(
                    np.float32(maximum - tile_maximum), dtype=np.float32
                )
                denominator = np.float32(
                    denominator * previous_scale + tile_denominator
                )
                maximum = np.float32(tile_maximum)
            probabilities = _probabilities(
                scores[head, query_index], maximum, denominator
            )
            output[head, query_index] = np.matmul(
                probabilities,
                value[head].astype(np.float32),
            )
    return output


class KevAttentionGridTests(unittest.TestCase):
    def test_dynamic_prim_func_has_only_linear_scratch(self):
        script = attention_grid.script()
        self.assertIn("sequence_length", script)
        self.assertNotRegex(
            script,
            r"\([^)]*sequence_length\s*,\s*sequence_length",
        )
        self.assertIn('(16, "sequence_length", 128), "float16"', script)
        self.assertIn('(16, 128, sequence_length), "float16"', script)
        self.assertRegex(
            script,
            r'value: T.Buffer\(\(16, sequence_length, 128\), "float32"\)',
        )
        self.assertRegex(
            script,
            r'output: T.Buffer\(\(16, sequence_length, 128\), "float32"\)',
        )

    def test_tiled_stats_match_dense_reference_across_tail_tiles(self):
        for sequence_length in (1, 7, 63, 65, 129):
            with self.subTest(sequence_length=sequence_length):
                inputs = _synthetic_inputs(sequence_length, 20261111 + sequence_length)
                expected = _dense_reference(*inputs[:3], *inputs[3:])
                actual = _tiled_reference(*inputs[:3], *inputs[3:])
                np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=0)

    def test_all_invalid_rows_preserve_the_finite_mask_fallback(self):
        inputs = _synthetic_inputs(7, 20261119, all_invalid=True)
        actual = _tiled_reference(*inputs[:3], *inputs[3:])
        expected = _dense_reference(*inputs[:3], *inputs[3:])
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
        self.assertGreater(float(np.linalg.norm(actual)), 0.0)

    def test_probability_fp16_boundary_changes_the_pv_result(self):
        inputs = _synthetic_inputs(17, 20261120)
        query, key, value, segments, valid = inputs
        scores = _masked_scores(query, key, segments, valid)
        maximum = np.max(scores, axis=-1, keepdims=True)
        exponentials = np.exp(scores - maximum, dtype=np.float32)
        denominator = np.sum(exponentials, axis=-1, keepdims=True, dtype=np.float32)
        unrounded = exponentials / denominator
        rounded = unrounded.astype(np.float16).astype(np.float32)
        self.assertGreater(
            float(
                np.max(np.abs(np.matmul(rounded, value) - np.matmul(unrounded, value)))
            ),
            0.0,
        )

    def test_webgpu_codegen_uses_two_pass_workgroup_kernel(self):
        executable = build_attention_grid("webgpu")
        webgpu_module = next(
            module for module in executable.imports if module.kind == "webgpu"
        )
        wgsl = webgpu_module.inspect_source("wgsl")
        self.assertIn("@workgroup_size(128, 1, 1)", wgsl)
        self.assertIn("var<workgroup> state : array<f32, 384>", wgsl)
        self.assertIn("var<workgroup> partial : array<f32, 128>", wgsl)
        self.assertIn("var<workgroup> scores : array<f32, 64>", wgsl)
        self.assertIn("var<storage, read> value_ptr : array<f32>", wgsl)
        self.assertRegex(wgsl, r"f16\(probability\[0i\]\)")
        self.assertRegex(wgsl, r"f32\(probability_f16\[0i\]\)")
        self.assertIn("loop_length: i32", wgsl)
        self.assertIn("skip_future_tiles: i32", wgsl)
        self.assertIn("query_start: i32", wgsl)
        self.assertIn("query_chunk_length: i32", wgsl)
        self.assertIn("podArgs.loop_length", wgsl)
        self.assertNotIn("length_i32_ptr", wgsl)
        self.assertIn("blockIdx.y", attention_grid.script())
        self.assertLess(len(wgsl.encode("utf-8")), 10 * 1024)
        self.assertNotRegex(wgsl, r"array<[^>]+,\s*\w*sequence_length")


if __name__ == "__main__":
    unittest.main()
