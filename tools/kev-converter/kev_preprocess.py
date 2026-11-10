"""Linear-memory Kev branch metadata preprocessing for TVM Relax."""

import tvm
from tvm import relax
from tvm.script import tirx as T

PREPROCESS_VALID = 0
PREPROCESS_SEGMENTS = 1
PREPROCESS_BRANCH_STARTS = 2
PREPROCESS_POSITION_IDS = 3
PREPROCESS_DECISION_SUMS = 4
PREPROCESS_NEXT_DECISION = 5
PREPROCESS_FIELDS = 6


@T.prim_func(s_tir=True)
def _kev_preprocess(
    ids_handle: T.handle,
    mask_handle: T.handle,
    batch_size: T.int64,
    sequence_length: T.int64,
    question_id: T.int32,
    decide_id: T.int32,
    output_handle: T.handle,
):
    """Write six O(B*S) vectors in one `(6, B, S+1)` slab."""
    input_ids = T.match_buffer(ids_handle, (batch_size, sequence_length), "int32")
    attention_mask = T.match_buffer(mask_handle, (batch_size, sequence_length), "int32")
    output = T.match_buffer(
        output_handle, (PREPROCESS_FIELDS, batch_size, sequence_length + 1), "int32"
    )
    T.func_attr({"tirx.noalias": True})
    with T.sblock("kev_preprocess_workgroup"):
        T.reads()
        T.writes()
        for batch in T.serial(batch_size):
            for index in T.serial(sequence_length + 1):
                output[PREPROCESS_VALID, batch, index] = 0
                output[PREPROCESS_SEGMENTS, batch, index] = 0
                output[PREPROCESS_BRANCH_STARTS, batch, index] = 0
                output[PREPROCESS_POSITION_IDS, batch, index] = 0
                output[PREPROCESS_DECISION_SUMS, batch, index] = 0
                output[PREPROCESS_NEXT_DECISION, batch, index] = 0

            segment_count = T.int32(0)
            first_question = sequence_length
            for index in T.serial(sequence_length):
                valid = T.Cast("int32", attention_mask[batch, index] != 0)
                question = T.Cast(
                    "int32", (input_ids[batch, index] == question_id) & (valid != 0)
                )
                decision = T.Cast(
                    "int32", (input_ids[batch, index] == decide_id) & (valid != 0)
                )
                segment_count = segment_count + question
                segment = T.if_then_else(valid != 0, segment_count, T.int32(-1))
                output[PREPROCESS_VALID, batch, index] = valid
                output[PREPROCESS_SEGMENTS, batch, index] = segment
                first_question = T.min(
                    first_question,
                    T.if_then_else(question != 0, index, sequence_length),
                )
                bucket = T.max(segment, T.int32(0))
                output[PREPROCESS_DECISION_SUMS, batch, bucket] = output[
                    PREPROCESS_DECISION_SUMS, batch, bucket
                ] + decision * T.Cast("int32", index)
            first_question = T.if_then_else(
                first_question == sequence_length,
                T.int64(0),
                first_question,
            )

            last_question = T.int32(-1)
            prefix_decisions = output[PREPROCESS_DECISION_SUMS, batch, 0]
            for index in T.serial(sequence_length):
                valid = output[PREPROCESS_VALID, batch, index]
                segment = output[PREPROCESS_SEGMENTS, batch, index]
                question = T.Cast(
                    "int32", (input_ids[batch, index] == question_id) & (valid != 0)
                )
                last_question = T.max(
                    last_question,
                    T.if_then_else(
                        question != 0,
                        T.Cast("int32", index),
                        T.int32(-1),
                    ),
                )
                branch_start = T.if_then_else(segment > 0, last_question, T.int32(0))
                position = T.if_then_else(
                    valid == 0,
                    T.int32(0),
                    T.if_then_else(
                        segment == 0,
                        T.Cast("int32", index),
                        T.Cast(
                            "int32",
                            first_question + index - T.Cast("int64", last_question),
                        ),
                    ),
                )
                bucket = T.max(segment, T.int32(0))
                branch_decisions = output[PREPROCESS_DECISION_SUMS, batch, bucket]
                output[PREPROCESS_BRANCH_STARTS, batch, index] = branch_start
                output[PREPROCESS_POSITION_IDS, batch, index] = position
                output[PREPROCESS_NEXT_DECISION, batch, index] = T.if_then_else(
                    segment > 0,
                    branch_decisions + prefix_decisions,
                    prefix_decisions,
                )


def build_kev_preprocessor_module(question_id: int, decide_id: int) -> tvm.IRModule:
    """Build a symbolic-shape Relax module for Kev's O(S) metadata vectors.

    The output is `(6, B, S+1)` int32: valid, segment, branch start,
    position ID, decision-index sum by segment, and per-token decision index.
    The extra bucket preserves the dense wrapper's behavior even for raw rows
    with a question marker at every position. No aggregate sequence bound is
    attached to the symbolic sequence dimension.
    """
    builder = relax.BlockBuilder()
    preprocess = builder.add_func(_kev_preprocess, "kev_preprocess")
    batch_size = tvm.tirx.Var("batch_size", "int64")
    sequence_length = tvm.tirx.Var("sequence_length", "int64")
    input_ids = relax.Var(
        "input_ids", relax.TensorType((batch_size, sequence_length), "int32")
    )
    attention_mask = relax.Var(
        "attention_mask", relax.TensorType((batch_size, sequence_length), "int32")
    )
    output_type = relax.TensorType(
        (PREPROCESS_FIELDS, batch_size, sequence_length + 1), "int32"
    )
    question_value = relax.prim_value(question_id, dtype="int32")
    decide_value = relax.prim_value(decide_id, dtype="int32")

    with builder.function("main", [input_ids, attention_mask]):
        shape = relax.get_shape_of(input_ids)
        actual_batch = shape[0]
        actual_length = shape[1]
        output = builder.emit(
            relax.call_tir(
                preprocess,
                [
                    input_ids,
                    attention_mask,
                    actual_batch,
                    actual_length,
                    question_value,
                    decide_value,
                ],
                output_type,
            ),
            "kev_preprocess",
        )
        builder.emit_func_output(output)
    return builder.get()
