"""Parity and shape checks for the compact Kev preprocessing kernel."""

import unittest

import numpy as np
import tvm
from kev_preprocess import (
    PREPROCESS_BRANCH_STARTS,
    PREPROCESS_DECISION_SUMS,
    PREPROCESS_FIELDS,
    PREPROCESS_NEXT_DECISION,
    PREPROCESS_POSITION_IDS,
    PREPROCESS_SEGMENTS,
    PREPROCESS_VALID,
    build_kev_preprocessor_module,
)
from tvm import relax

QUESTION = 9
DECIDE = 8
STATE = 151659
OPTION_START = 151648
OPTION_END = 151649


def imported_modules(module):
    """Yield imported runtime modules recursively."""
    for imported in module.imports:
        yield imported
        yield from imported_modules(imported)


def dense_reference(input_ids, attention_mask):
    """Reproduce the existing S-by-S Kev wrapper for short rows."""
    batch_size, sequence_length = input_ids.shape
    if sequence_length > 64:
        raise ValueError("The dense oracle is limited to short test rows.")
    positions = np.arange(sequence_length, dtype=np.int32)
    outputs = np.zeros(
        (PREPROCESS_FIELDS, batch_size, sequence_length + 1), dtype=np.int32
    )
    dense_attention = np.zeros(
        (batch_size, sequence_length, sequence_length), dtype=np.bool_
    )
    for batch in range(batch_size):
        valid = (attention_mask[batch] != 0).astype(np.int32)
        questions = (input_ids[batch] == QUESTION).astype(np.int32) * valid
        segments = np.cumsum(questions, dtype=np.int32) * valid + valid - 1
        same_branch = np.maximum(
            (segments[:, None] == segments[None, :]).astype(np.int32),
            (segments[None, :] == 0).astype(np.int32),
        )
        branch_starts = (same_branch * questions[None, :] * positions[None, :]).sum(
            axis=1, dtype=np.int32
        )
        first_question = (questions * (segments == 1).astype(np.int32) * positions).sum(
            dtype=np.int32
        )
        position_ids = (
            np.where(
                segments == 0,
                positions,
                first_question + positions - branch_starts,
            )
            * valid
        )
        decisions = (input_ids[batch] == DECIDE).astype(np.int32) * valid
        next_decision = (same_branch * decisions[None, :] * positions[None, :]).sum(
            axis=1, dtype=np.int32
        )
        decision_sums = np.zeros(sequence_length + 1, dtype=np.int32)
        for index in positions:
            if decisions[index] != 0:
                decision_sums[segments[index]] += index
        causal = positions[None, :] <= positions[:, None]
        allowed = causal & (same_branch != 0) & (valid[None, :] != 0)

        outputs[PREPROCESS_VALID, batch, :sequence_length] = valid
        outputs[PREPROCESS_SEGMENTS, batch, :sequence_length] = segments
        outputs[PREPROCESS_BRANCH_STARTS, batch, :sequence_length] = branch_starts
        outputs[PREPROCESS_POSITION_IDS, batch, :sequence_length] = position_ids
        outputs[PREPROCESS_DECISION_SUMS, batch] = decision_sums
        outputs[PREPROCESS_NEXT_DECISION, batch, :sequence_length] = next_decision
        dense_attention[batch] = allowed
    return outputs, dense_attention


def packed_aggregate_fixture():
    counts = (8187, 8187, 8187, 4092, 4096)
    ids = [STATE]
    for count in counts:
        ids.extend((QUESTION, *([100] * count), OPTION_START, OPTION_END, DECIDE))
    return np.asarray([ids], dtype=np.int32)


def aggregate_reference(input_ids):
    """Linear reference for the 32,770-token packed integration fixture."""
    batch_size, sequence_length = input_ids.shape
    positions = np.arange(sequence_length, dtype=np.int32)
    output = np.zeros((PREPROCESS_FIELDS, batch_size, sequence_length + 1), np.int32)
    valid = np.ones(sequence_length, dtype=np.int32)
    questions = (input_ids[0] == QUESTION).astype(np.int32)
    segments = np.cumsum(questions, dtype=np.int32)
    question_positions = np.flatnonzero(questions)
    first_question = int(question_positions[0])
    starts = np.maximum.accumulate(np.where(questions != 0, positions, 0))
    position_ids = np.where(
        segments == 0, positions, first_question + positions - starts
    ).astype(np.int32)
    decisions = (input_ids[0] == DECIDE).astype(np.int32)
    decision_sums = np.zeros(sequence_length + 1, dtype=np.int32)
    decision_positions = np.flatnonzero(decisions)
    np.add.at(decision_sums, segments[decision_positions], decision_positions)
    next_decision = np.where(
        segments > 0,
        decision_sums[segments] + decision_sums[0],
        decision_sums[0],
    ).astype(np.int32)
    output[PREPROCESS_VALID, 0, :sequence_length] = valid
    output[PREPROCESS_SEGMENTS, 0, :sequence_length] = segments
    output[PREPROCESS_BRANCH_STARTS, 0, :sequence_length] = starts
    output[PREPROCESS_POSITION_IDS, 0, :sequence_length] = position_ids
    output[PREPROCESS_DECISION_SUMS, 0] = decision_sums
    output[PREPROCESS_NEXT_DECISION, 0, :sequence_length] = next_decision
    return output


class KevPreprocessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = build_kev_preprocessor_module(QUESTION, DECIDE)
        executable = relax.build(cls.module, target="llvm")
        cls.vm = relax.VirtualMachine(executable, tvm.cpu())

    def run_preprocessor(self, input_ids, attention_mask):
        result = self.vm["main"](
            tvm.runtime.tensor(np.asarray(input_ids, dtype=np.int32)),
            tvm.runtime.tensor(np.asarray(attention_mask, dtype=np.int32)),
        )
        return np.asarray(result.numpy())

    def assert_dense_parity(self, input_ids, attention_mask):
        ids = np.asarray(input_ids, dtype=np.int32)
        mask = np.asarray(attention_mask, dtype=np.int32)
        expected, dense_attention = dense_reference(ids, mask)
        actual = self.run_preprocessor(ids, mask)
        np.testing.assert_array_equal(actual, expected)

        positions = np.arange(ids.shape[1], dtype=np.int32)
        valid = actual[PREPROCESS_VALID, :, : ids.shape[1]]
        segments = actual[PREPROCESS_SEGMENTS, :, : ids.shape[1]]
        for batch in range(ids.shape[0]):
            compact_attention = (
                (positions[None, :] <= positions[:, None])
                & (valid[batch, None, :] != 0)
                & (
                    (segments[batch, :, None] == segments[batch, None, :])
                    | (segments[batch, None, :] == 0)
                )
            )
            np.testing.assert_array_equal(compact_attention, dense_attention[batch])

    def test_matches_dense_wrapper_with_two_right_padded_rows(self):
        input_ids = [
            [100, 101, 9, 11, 20, 7, 21, 7, 8, 9, 12, 22, 7, 23, 7, 8],
            [100, 9, 13, 30, 7, 8, *([0] * 10)],
        ]
        attention_mask = [[1] * 16, [1, 1, 1, 1, 1, 1, *([0] * 10)]]
        self.assert_dense_parity(input_ids, attention_mask)
        actual = self.run_preprocessor(input_ids, attention_mask)
        self.assertEqual(actual[PREPROCESS_VALID, 1, 6:16].tolist(), [0] * 10)
        self.assertEqual(actual[PREPROCESS_NEXT_DECISION, 1, 6:16].tolist(), [0] * 10)

    def test_matches_dense_wrapper_with_left_and_internal_padding(self):
        input_ids = [
            [9, 100, 9, 11, 8, 9, 22, 8, 7, 0, 8],
            [9, 100, 11, 7, 8, 0, 0, 0, 0, 0, 0],
        ]
        attention_mask = [
            [0, 3, 1, 1, 0, 1, 3, 1, 1, 0, 0],
            [0, 3, 2, 2, 2, 0, 0, 0, 0, 0, 0],
        ]
        self.assert_dense_parity(input_ids, attention_mask)

    def test_matches_dense_wrapper_without_questions_and_with_duplicate_deciders(self):
        cases = [
            ([[8, 8, 100, 7]], [[1, 1, 1, 1]]),
            ([[9, 8, 7, 9]], [[0, 0, 0, 0]]),
            ([[8, 8, 9, 8, 8, 7]], [[1, 1, 1, 1, 1, 1]]),
            (
                [[100, 9, OPTION_START, OPTION_END, 8, 9, OPTION_START, OPTION_END, 8]],
                [[1] * 9],
            ),
        ]
        for input_ids, attention_mask in cases:
            with self.subTest(input_ids=input_ids):
                self.assert_dense_parity(input_ids, attention_mask)

    def test_segment_bucket_handles_question_marker_at_every_position(self):
        self.assert_dense_parity([[QUESTION] * 4], [[1] * 4])

    def test_dynamic_kernel_accepts_32770_packed_tokens_and_reuses_positions(self):
        input_ids = packed_aggregate_fixture()
        self.assertEqual(input_ids.shape, (1, 32770))
        actual = self.run_preprocessor(input_ids, np.ones_like(input_ids))
        np.testing.assert_array_equal(actual, aggregate_reference(input_ids))
        self.assertEqual(
            int(actual[PREPROCESS_POSITION_IDS, 0, : input_ids.shape[1]].max()),
            8191,
        )

    def test_relax_shapes_are_linear_and_sequence_dimension_is_unbounded(self):
        script = self.module.script()
        self.assertRegex(
            script,
            r"Tensor\(\(batch_size, sequence_length\), dtype=\"int32\"\)",
        )
        self.assertRegex(
            script,
            r"Tensor\(\(6, batch_size, sequence_length \+ 1\), dtype=\"int32\"\)",
        )
        self.assertNotRegex(
            script,
            r"(?:Tensor|Buffer)\(\([^)]*sequence_length\s*,\s*sequence_length",
        )
        self.assertNotIn("32770", script)
        self.assertNotIn("32768", script)

    def test_dynamic_relax_module_builds_for_webgpu(self):
        target = tvm.target.Target(
            "webgpu",
            host={"kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"},
        )
        module = build_kev_preprocessor_module(QUESTION, DECIDE, target=target)
        executable = relax.build(module, target=target)
        webgpu_modules = [
            module
            for module in imported_modules(executable.mod)
            if module.kind == "webgpu"
        ]
        self.assertTrue(webgpu_modules)
        wgsl = webgpu_modules[0].inspect_source("wgsl")
        self.assertIn("fn _kev_preprocess_kernel", wgsl)
        self.assertIn("attention_mask_ptr", wgsl)
        self.assertIn("@workgroup_size(1, 1, 1)", wgsl)


if __name__ == "__main__":
    unittest.main()
