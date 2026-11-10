"""Isolated cooperative TIRX prototype for Kev branch-aware attention."""

import tvm
from tvm import s_tir, tirx
from tvm.script import tirx as T

HEAD_DIM = 128
KEY_TILE = 64
LANES_PER_KEY = 2
REDUCTION_STEPS = LANES_PER_KEY.bit_length() - 1


@T.prim_func(s_tir=True)
def _reduction_probe(
    query_handle: T.handle,
    key_handle: T.handle,
    value_handle: T.handle,
    output_handle: T.handle,
    sequence_length: T.int32,
    num_heads: T.int64,
):
    """Probe reduction result visibility across workgroup lanes."""
    query = T.match_buffer(
        query_handle, (sequence_length, num_heads, HEAD_DIM), "float32"
    )
    key = T.match_buffer(key_handle, (sequence_length, num_heads, HEAD_DIM), "float32")
    value = T.match_buffer(
        value_handle, (sequence_length, num_heads, HEAD_DIM), "float32"
    )
    output = T.match_buffer(
        output_handle, (sequence_length, num_heads, HEAD_DIM), "float32"
    )
    T.func_attr({"tirx.noalias": True})

    with T.sblock("root"):
        T.reads()
        T.writes()
        dot = T.alloc_buffer((1,), "float32", scope="shared")
        for query_loop in T.serial(sequence_length):
            for head_loop in T.serial(num_heads):
                for channel_loop in T.serial(HEAD_DIM):
                    with T.sblock("kev_attention_tiled_dot"):
                        query_index = T.axis.spatial(sequence_length, query_loop)
                        head_index = T.axis.spatial(num_heads, head_loop)
                        channel = T.axis.reduce(HEAD_DIM, channel_loop)
                        T.reads(
                            query[query_index, head_index, 0:HEAD_DIM],
                            key[query_index, head_index, 0:HEAD_DIM],
                        )
                        T.writes(dot[0])
                        with T.init():
                            dot[0] = T.float32(0.0)
                        dot[0] = (
                            dot[0]
                            + query[query_index, head_index, channel]
                            * key[query_index, head_index, channel]
                        )
                for lane_loop in T.serial(HEAD_DIM):
                    with T.sblock("kev_attention_tiled_consume"):
                        query_index = T.axis.spatial(sequence_length, query_loop)
                        head_index = T.axis.spatial(num_heads, head_loop)
                        lane = T.axis.spatial(HEAD_DIM, lane_loop)
                        T.reads(dot[0], value[query_index, head_index, lane])
                        T.writes(output[query_index, head_index, lane])
                        output[query_index, head_index, lane] = (
                            dot[0] * value[query_index, head_index, lane]
                        )


def schedule_reduction_probe(prim_func=None):
    """Bind one query/head per workgroup and 128 reduction/output lanes."""
    prim_func = _reduction_probe if prim_func is None else prim_func
    schedule = s_tir.Schedule(prim_func)
    for block_name in ("kev_attention_tiled_dot", "kev_attention_tiled_consume"):
        loops = schedule.get_loops(schedule.get_sblock(block_name))
        for loop, tag in zip(loops, ("blockIdx.x", "blockIdx.y", "threadIdx.x")):
            schedule.bind(loop, tag)
    return schedule.mod["main"]


@T.macro
def _initialize_attention_state(state):
    """Initialize the shared online-softmax state for every output lane."""
    for lane_loop in T.thread_binding(HEAD_DIM, thread="threadIdx.x"):
        with T.sblock("initialize_row"):
            lane = T.axis.spatial(HEAD_DIM, lane_loop)
            T.reads()
            T.writes(state[lane, 0:3])
            state[lane, 0] = T.float32(-3.402823466e38)
            state[lane, 1] = T.float32(0.0)
            state[lane, 2] = T.float32(0.0)
    T.tvm_storage_sync("shared")


@T.macro
def _compute_key_dot_partials(
    query, key, partial, key_base, sequence_length_i32, sequence_length
):
    """Compute two feature partials for each key in the current tile."""
    for partial_loop in T.thread_binding(HEAD_DIM, thread="threadIdx.x"):
        with T.sblock("key_dot_partials"):
            lane = T.axis.spatial(HEAD_DIM, partial_loop)
            key_slot = lane // LANES_PER_KEY
            feature_lane = lane % LANES_PER_KEY
            key_index = key_base + key_slot
            safe_key_index = T.min(key_index, sequence_length_i32 - 1)
            T.reads(query[0:HEAD_DIM], key[0:sequence_length, 0:HEAD_DIM])
            T.writes(partial[lane])
            partial_sum = T.float32(0.0)
            for feature_group in T.serial(HEAD_DIM // LANES_PER_KEY):
                channel = feature_lane + feature_group * LANES_PER_KEY
                partial_sum = partial_sum + T.Cast("float32", query[channel]) * T.Cast(
                    "float32", key[T.Cast("int64", safe_key_index), channel]
                )
            partial[lane] = partial_sum
    T.tvm_storage_sync("shared")


@T.macro
def _reduce_key_dot_partials(partial):
    """Finish the feature reduction for every key in the current tile."""
    for reduce_loop in T.serial(REDUCTION_STEPS):
        for reduce_lane_loop in T.thread_binding(HEAD_DIM, thread="threadIdx.x"):
            with T.sblock("reduce_key_dot"):
                step = T.axis.spatial(REDUCTION_STEPS, reduce_loop)
                lane = T.axis.spatial(HEAD_DIM, reduce_lane_loop)
                feature_lane = lane % LANES_PER_KEY
                offset = LANES_PER_KEY >> (step + 1)
                T.reads(partial[0:HEAD_DIM])
                T.writes(partial[0:HEAD_DIM])
                if feature_lane < offset:
                    partial[lane] = partial[lane] + partial[lane + offset]
        T.tvm_storage_sync("shared")


@T.macro
def _write_key_scores(
    partial,
    scores,
    segments,
    valid,
    key_base,
    query_position,
    query_segment,
    scale,
    sequence_length_i32,
    sequence_length,
):
    """Apply causal, segment, and validity masks to the tile scores."""
    for score_loop in T.thread_binding(HEAD_DIM, thread="threadIdx.x"):
        with T.sblock("write_key_scores"):
            lane = T.axis.spatial(HEAD_DIM, score_loop)
            key_slot = lane // LANES_PER_KEY
            feature_lane = lane % LANES_PER_KEY
            key_index = key_base + key_slot
            safe_key_index = T.min(key_index, sequence_length_i32 - 1)
            in_range = key_index < sequence_length_i32
            visible = (
                in_range
                & (key_index <= query_position)
                & (valid[T.Cast("int64", safe_key_index)] != 0)
                & (
                    (segments[T.Cast("int64", safe_key_index)] == query_segment)
                    | (segments[T.Cast("int64", safe_key_index)] == 0)
                )
            )
            bias = T.if_then_else(
                visible,
                T.float32(0.0),
                T.float32(-65504.0),
            )
            score = partial[lane] * scale + bias
            T.reads(
                partial[0:HEAD_DIM],
                valid[0:sequence_length],
                segments[0:sequence_length],
            )
            T.writes(scores[0:KEY_TILE])
            if feature_lane == 0:
                scores[key_slot] = T.if_then_else(
                    in_range,
                    score,
                    T.float32(-3.402823466e38),
                )
    T.tvm_storage_sync("shared")


@T.macro
def _update_online_attention(
    state, scores, value, key_base, sequence_length_i32, sequence_length
):
    """Merge the current tile into the running softmax and value sums."""
    for update_loop in T.thread_binding(HEAD_DIM, thread="threadIdx.x"):
        with T.sblock("online_tile_update"):
            lane = T.axis.spatial(HEAD_DIM, update_loop)
            tile_maximum = T.float32(-3.402823466e38)
            tile_denominator = T.float32(0.0)
            tile_accumulator = T.float32(0.0)
            for slot in T.serial(KEY_TILE):
                tile_maximum = T.max(tile_maximum, scores[slot])
            previous_maximum = state[lane, 0]
            next_maximum = T.max(previous_maximum, tile_maximum)
            previous_weight = T.exp(previous_maximum - next_maximum)
            for slot in T.serial(KEY_TILE):
                key_index = key_base + slot
                safe_key_index = T.min(key_index, sequence_length_i32 - 1)
                weight = T.exp(scores[slot] - next_maximum)
                tile_denominator = tile_denominator + weight
                if key_index < sequence_length_i32:
                    tile_accumulator = tile_accumulator + weight * T.Cast(
                        "float32",
                        value[T.Cast("int64", safe_key_index), lane],
                    )
            T.reads(
                scores[0:KEY_TILE],
                state[lane, 0:3],
                value[0:sequence_length, lane],
            )
            T.writes(state[lane, 0:3])
            state[lane, 0] = next_maximum
            state[lane, 1] = state[lane, 1] * previous_weight + tile_denominator
            state[lane, 2] = state[lane, 2] * previous_weight + tile_accumulator
    T.tvm_storage_sync("shared")


@T.macro
def _write_attention_output(state, output):
    """Normalize the running value sums into the output row."""
    for output_loop in T.thread_binding(HEAD_DIM, thread="threadIdx.x"):
        with T.sblock("write_attention_output"):
            lane = T.axis.spatial(HEAD_DIM, output_loop)
            denominator = state[lane, 1]
            accumulator = state[lane, 2]
            T.reads(state[lane, 1:3])
            T.writes(output[lane])
            output[lane] = T.if_then_else(
                denominator > T.float32(0.0),
                accumulator / denominator,
                T.float32(0.0),
            )


@T.prim_func(s_tir=True)
def _online_attention_row(
    query_handle: T.handle,
    key_handle: T.handle,
    value_handle: T.handle,
    segment_handle: T.handle,
    valid_handle: T.handle,
    query_position: T.int32,
    query_segment: T.int32,
    scale: T.float32,
    sequence_length: T.int64,
    output_handle: T.handle,
):
    """Stream one query/head row through 64-key tiles with online softmax."""
    query = T.match_buffer(query_handle, (HEAD_DIM,), "float16")
    sequence_length_i32 = T.Cast("int32", sequence_length)
    key = T.match_buffer(key_handle, (sequence_length, HEAD_DIM), "float16")
    value = T.match_buffer(value_handle, (sequence_length, HEAD_DIM), "float16")
    segments = T.match_buffer(segment_handle, (sequence_length,), "int32")
    valid = T.match_buffer(valid_handle, (sequence_length,), "int32")
    output = T.match_buffer(output_handle, (HEAD_DIM,), "float32")
    T.func_attr({"tirx.noalias": True})

    for workgroup_loop in T.thread_binding(1, thread="blockIdx.x"):
        partial = T.alloc_buffer((HEAD_DIM,), "float32", scope="shared")
        scores = T.alloc_buffer((KEY_TILE,), "float32", scope="shared")
        state = T.alloc_buffer((HEAD_DIM, 3), "float32", scope="shared")

        _initialize_attention_state(state)

        for tile_loop in T.serial((sequence_length_i32 + KEY_TILE - 1) // KEY_TILE):
            key_base = tile_loop * KEY_TILE
            _compute_key_dot_partials(
                query, key, partial, key_base, sequence_length_i32, sequence_length
            )
            _reduce_key_dot_partials(partial)
            _write_key_scores(
                partial,
                scores,
                segments,
                valid,
                key_base,
                query_position,
                query_segment,
                scale,
                sequence_length_i32,
                sequence_length,
            )
            _update_online_attention(
                state, scores, value, key_base, sequence_length_i32, sequence_length
            )

        _write_attention_output(state, output)


def build_online_attention_row(target="webgpu"):
    """Build the patch-free TVM kernel for one attention query/head row."""
    target = tvm.target.Target(target)
    schedule = s_tir.Schedule(_online_attention_row)
    return tirx.build(schedule.mod, target=target)
