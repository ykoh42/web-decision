"""Two-pass, full-sequence/head-grid attention for the pinned Kev graph."""

import tvm
from tvm import s_tir, tirx
from tvm.script import tirx as T

NUM_QUERY_HEADS = 16
HEAD_DIM = 128
KEY_TILE = 64
LANES_PER_KEY = 2
ATTENTION_SCALE = 0.088388346135616302
FINITE_FP16_MASK = -65504.0
NEGATIVE_INFINITY = -3.402823466e38


@T.macro
def _initialize_state(state):
    """Initialize each output lane's online-softmax state."""
    for lane_loop in T.thread_binding(HEAD_DIM, thread="threadIdx.x"):
        with T.sblock("initialize_attention_state"):
            lane = T.axis.spatial(HEAD_DIM, lane_loop)
            T.reads()
            T.writes(state[lane, 0:3])
            state[lane, 0] = T.float32(NEGATIVE_INFINITY)
            state[lane, 1] = T.float32(0.0)
            state[lane, 2] = T.float32(0.0)
    T.tvm_storage_sync("shared")


@T.macro
def _compute_dot_partials(
    query, key, partial, head, query_index, key_base, length_i32, sequence_length
):
    """Compute the two interleaved FP32 dot-product partials per key."""
    for partial_loop in T.thread_binding(HEAD_DIM, thread="threadIdx.x"):
        with T.sblock("attention_dot_partials"):
            lane = T.axis.spatial(HEAD_DIM, partial_loop)
            key_slot = lane // LANES_PER_KEY
            feature_lane = lane % LANES_PER_KEY
            key_index = key_base + key_slot
            safe_key_index = T.min(key_index, length_i32 - 1)
            T.reads(
                query[head, query_index, 0:HEAD_DIM],
                key[head, 0:HEAD_DIM, 0:sequence_length],
            )
            T.writes(partial[lane])
            partial_sum = T.float32(0.0)
            for feature_group in T.serial(HEAD_DIM // LANES_PER_KEY):
                channel = feature_lane + feature_group * LANES_PER_KEY
                partial_sum = partial_sum + T.Cast(
                    "float32",
                    query[head, query_index, channel],
                ) * T.Cast(
                    "float32",
                    key[head, channel, T.Cast("int64", safe_key_index)],
                )
            partial[lane] = partial_sum
    T.tvm_storage_sync("shared")

    for reduce_loop in T.serial(LANES_PER_KEY.bit_length() - 1):
        for reduce_lane_loop in T.thread_binding(HEAD_DIM, thread="threadIdx.x"):
            with T.sblock("reduce_attention_dot_partials"):
                step = T.axis.spatial(LANES_PER_KEY.bit_length() - 1, reduce_loop)
                lane = T.axis.spatial(HEAD_DIM, reduce_lane_loop)
                feature_lane = lane % LANES_PER_KEY
                offset = LANES_PER_KEY >> (step + 1)
                T.reads(partial[0:HEAD_DIM])
                T.writes(partial[0:HEAD_DIM])
                if feature_lane < offset:
                    partial[lane] = partial[lane] + partial[lane + offset]
            T.tvm_storage_sync("shared")


@T.macro
def _write_masked_scores(
    partial,
    scores,
    segments,
    valid,
    head,
    query_index,
    key_base,
    query_segment,
    length_i32,
    sequence_length,
):
    """Apply causal, segment, and validity conditions with the finite FP16 bias."""
    for score_loop in T.thread_binding(HEAD_DIM, thread="threadIdx.x"):
        with T.sblock("write_attention_scores"):
            lane = T.axis.spatial(HEAD_DIM, score_loop)
            key_slot = lane // LANES_PER_KEY
            feature_lane = lane % LANES_PER_KEY
            raw_key_index = key_base + key_slot
            in_range = raw_key_index < length_i32
            safe_key_index = T.min(raw_key_index, length_i32 - 1)
            visible = (
                in_range
                & (raw_key_index <= query_index)
                & (valid[0, T.Cast("int64", safe_key_index)] != 0)
                & (
                    (segments[0, T.Cast("int64", safe_key_index)] == query_segment)
                    | (segments[0, T.Cast("int64", safe_key_index)] == 0)
                )
            )
            bias = T.if_then_else(
                visible,
                T.float32(0.0),
                T.float32(FINITE_FP16_MASK),
            )
            score = partial[lane] * T.float32(ATTENTION_SCALE) + bias
            T.reads(
                partial[0:HEAD_DIM],
                valid[0:1, 0:sequence_length],
                segments[0:1, 0:sequence_length],
            )
            T.writes(scores[0:KEY_TILE])
            if feature_lane == 0:
                scores[key_slot] = T.if_then_else(
                    in_range,
                    score,
                    T.float32(NEGATIVE_INFINITY),
                )
    T.tvm_storage_sync("shared")


@T.macro
def _update_softmax_stats(state, scores):
    """Update the row maximum and denominator for one key tile."""
    for lane_loop in T.thread_binding(HEAD_DIM, thread="threadIdx.x"):
        with T.sblock("update_attention_stats"):
            lane = T.axis.spatial(HEAD_DIM, lane_loop)
            tile_maximum = T.float32(NEGATIVE_INFINITY)
            for slot in T.serial(KEY_TILE):
                tile_maximum = T.max(tile_maximum, scores[slot])
            previous_maximum = state[lane, 0]
            next_maximum = T.max(previous_maximum, tile_maximum)
            tile_denominator = T.float32(0.0)
            for slot in T.serial(KEY_TILE):
                tile_denominator = tile_denominator + T.exp(scores[slot] - next_maximum)
            previous_scale = T.exp(previous_maximum - next_maximum)
            T.reads(scores[0:KEY_TILE], state[lane, 0:2])
            T.writes(state[lane, 0:2])
            state[lane, 0] = next_maximum
            state[lane, 1] = state[lane, 1] * previous_scale + tile_denominator
    T.tvm_storage_sync("shared")


@T.macro
def _accumulate_rounded_values(
    state, scores, value, head, key_base, length_i32, sequence_length
):
    """Recompute probabilities, round P through FP16, and accumulate FP32 V."""
    for lane_loop in T.thread_binding(HEAD_DIM, thread="threadIdx.x"):
        with T.sblock("accumulate_attention_values"):
            lane = T.axis.spatial(HEAD_DIM, lane_loop)
            accumulator = state[lane, 2]
            maximum = state[lane, 0]
            denominator = state[lane, 1]
            T.reads(
                scores[0:KEY_TILE],
                state[lane, 0:3],
                value[head, 0:sequence_length, lane],
            )
            T.writes(state[lane, 2])
            for slot in T.serial(KEY_TILE):
                raw_key_index = key_base + slot
                if raw_key_index < length_i32:
                    probability = T.exp(scores[slot] - maximum) / denominator
                    probability_f16 = T.Cast("float16", probability)
                    probability_f32 = T.Cast("float32", probability_f16)
                    accumulator = (
                        accumulator
                        + probability_f32
                        * value[
                            head,
                            T.Cast("int64", raw_key_index),
                            lane,
                        ]
                    )
            state[lane, 2] = accumulator
    T.tvm_storage_sync("shared")


@T.prim_func(s_tir=True)
def attention_grid(
    query_handle: T.handle,
    key_handle: T.handle,
    value_handle: T.handle,
    segment_handle: T.handle,
    valid_handle: T.handle,
    sequence_length: T.int64,
    loop_length: T.int32,
    output_handle: T.handle,
):
    """Fuse dense QK, finite masking, softmax, and PV on a 2D query/head grid."""
    query = T.match_buffer(
        query_handle,
        (NUM_QUERY_HEADS, sequence_length, HEAD_DIM),
        "float16",
    )
    key = T.match_buffer(
        key_handle,
        (NUM_QUERY_HEADS, HEAD_DIM, sequence_length),
        "float16",
    )
    value = T.match_buffer(
        value_handle,
        (NUM_QUERY_HEADS, sequence_length, HEAD_DIM),
        "float32",
    )
    segments = T.match_buffer(segment_handle, (1, sequence_length), "int32")
    valid = T.match_buffer(valid_handle, (1, sequence_length), "int32")
    output = T.match_buffer(
        output_handle,
        (NUM_QUERY_HEADS, sequence_length, HEAD_DIM),
        "float32",
    )
    T.func_attr({"tirx.noalias": True})

    for query_loop in T.thread_binding(sequence_length, thread="blockIdx.x"):
        for head_loop in T.thread_binding(
            NUM_QUERY_HEADS,
            thread="blockIdx.y",
        ):
            query_index = T.Cast("int32", query_loop)
            head = T.Cast("int32", head_loop)
            query_segment = segments[0, T.Cast("int64", query_index)]
            partial = T.alloc_buffer((HEAD_DIM,), "float32", scope="shared")
            scores = T.alloc_buffer((KEY_TILE,), "float32", scope="shared")
            state = T.alloc_buffer((HEAD_DIM, 3), "float32", scope="shared")
            _initialize_state(state)

            tile_count = (loop_length + KEY_TILE - 1) // KEY_TILE
            for tile_loop in T.serial(tile_count):
                key_base = tile_loop * KEY_TILE
                _compute_dot_partials(
                    query,
                    key,
                    partial,
                    head,
                    query_index,
                    key_base,
                    loop_length,
                    sequence_length,
                )
                _write_masked_scores(
                    partial,
                    scores,
                    segments,
                    valid,
                    head,
                    query_index,
                    key_base,
                    query_segment,
                    loop_length,
                    sequence_length,
                )
                _update_softmax_stats(state, scores)

            for tile_loop in T.serial(tile_count):
                key_base = tile_loop * KEY_TILE
                _compute_dot_partials(
                    query,
                    key,
                    partial,
                    head,
                    query_index,
                    key_base,
                    loop_length,
                    sequence_length,
                )
                _write_masked_scores(
                    partial,
                    scores,
                    segments,
                    valid,
                    head,
                    query_index,
                    key_base,
                    query_segment,
                    loop_length,
                    sequence_length,
                )
                _accumulate_rounded_values(
                    state,
                    scores,
                    value,
                    head,
                    key_base,
                    loop_length,
                    sequence_length,
                )

            for output_loop in T.thread_binding(HEAD_DIM, thread="threadIdx.x"):
                with T.sblock("write_attention_output"):
                    lane = T.axis.spatial(HEAD_DIM, output_loop)
                    T.reads(state[lane, 2])
                    T.writes(output[head, T.Cast("int64", query_index), lane])
                    output[head, T.Cast("int64", query_index), lane] = state[lane, 2]


def build_attention_grid(target="webgpu"):
    """Build the dynamic full-grid kernel for a target such as WebGPU."""
    schedule = s_tir.Schedule(attention_grid)
    return tirx.build(schedule.mod, target=tvm.target.Target(target))
