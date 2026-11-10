"""Experimental compact branch-aware attention kernels for Kev.

This module is separate from ``build_tvm.py`` and the shipped package. It uses
only O(S*H) scratch and no S-by-S score or bias buffer. The output kernel is
correctness-first: each output lane recomputes its row's QK dot products, so it
is not a performance-ready GPU implementation.

Metadata must describe a prefix followed by contiguous branch ranges, with
invalid positions marked by valid=0 and segment=-1. Rows with no visible key
scan all S keys and add the finite FP16 mask. Rows with visible keys omit
non-visible cross-branch keys; this relies on their -65504 masked logits
underflowing for normal Qwen activation ranges, and is not an adversarial-logit
equivalence guarantee.
"""

import tvm
from tvm import s_tir
from tvm.script import tirx as T

FINITE_FP16_MASK = -65504.0


@T.prim_func(s_tir=True)
def _kev_attention_plan(
    segment_handle: T.handle,
    valid_handle: T.handle,
    branch_start_handle: T.handle,
    prefix_end: T.int32,
    sequence_length: T.int64,
    plan_handle: T.handle,
):
    """Select visible branch ranges and detect rows needing finite-mask fallback."""
    segments = T.match_buffer(segment_handle, (sequence_length,), "int32")
    valid = T.match_buffer(valid_handle, (sequence_length,), "int32")
    branch_start_by_row = T.match_buffer(
        branch_start_handle, (sequence_length,), "int32"
    )
    plan = T.match_buffer(plan_handle, (sequence_length, 2), "int32")
    T.func_attr({"tirx.noalias": True})

    for query_loop in T.serial(sequence_length):
        with T.sblock("kev_attention_plan"):
            query_index = T.axis.spatial(sequence_length, query_loop)
            T.reads(
                segments[0:sequence_length],
                valid[0:sequence_length],
                branch_start_by_row[0:sequence_length],
            )
            T.writes(plan[query_index, 0:2])
            query_segment = segments[query_index]
            branch_start = branch_start_by_row[query_index]
            prefix_limit = T.min(prefix_end, T.Cast("int32", query_index + T.int64(1)))
            branch_length = T.if_then_else(
                query_segment > 0,
                T.Cast("int32", query_index) - branch_start + T.int32(1),
                T.int32(0),
            )
            candidate_count = T.if_then_else(
                query_segment == 0,
                T.Cast("int32", query_index) + T.int32(1),
                prefix_limit + branch_length,
            )
            visible_count = T.int32(0)
            for candidate in T.serial(candidate_count):
                candidate_key = T.if_then_else(
                    (query_segment > 0) & (candidate >= prefix_limit),
                    branch_start + candidate - prefix_limit,
                    candidate,
                )
                candidate_segment = segments[candidate_key]
                visible = (valid[candidate_key] != 0) & (
                    (candidate_segment == query_segment) | (candidate_segment == 0)
                )
                visible_count = visible_count + T.Cast("int32", visible)
            all_masked = visible_count == 0
            selected_count = T.if_then_else(
                all_masked,
                T.Cast("int32", sequence_length),
                candidate_count,
            )
            plan[query_index, 0] = selected_count
            plan[query_index, 1] = T.Cast("int32", all_masked)


@T.prim_func(s_tir=True)
def _kev_attention_stats(
    query_handle: T.handle,
    key_handle: T.handle,
    segment_handle: T.handle,
    valid_handle: T.handle,
    branch_start_handle: T.handle,
    plan_handle: T.handle,
    prefix_end: T.int32,
    scale: T.float32,
    sequence_length: T.int64,
    num_heads: T.int64,
    head_dim: T.int64,
    stats_handle: T.handle,
):
    """Compute each row/head's FP32 online-softmax maximum and denominator."""
    query = T.match_buffer(
        query_handle, (sequence_length, num_heads, head_dim), "float16"
    )
    key = T.match_buffer(key_handle, (sequence_length, num_heads, head_dim), "float16")
    segments = T.match_buffer(segment_handle, (sequence_length,), "int32")
    valid = T.match_buffer(valid_handle, (sequence_length,), "int32")
    branch_start_by_row = T.match_buffer(
        branch_start_handle, (sequence_length,), "int32"
    )
    plan = T.match_buffer(plan_handle, (sequence_length, 2), "int32")
    stats = T.match_buffer(stats_handle, (sequence_length, num_heads, 2), "float32")
    T.func_attr({"tirx.noalias": True})

    for query_loop in T.serial(sequence_length):
        for head_loop in T.serial(num_heads):
            with T.sblock("kev_attention_stats"):
                query_index = T.axis.spatial(sequence_length, query_loop)
                head_index = T.axis.spatial(num_heads, head_loop)
                T.reads(
                    query[0:sequence_length, 0:num_heads, 0:head_dim],
                    key[0:sequence_length, 0:num_heads, 0:head_dim],
                    segments[0:sequence_length],
                    valid[0:sequence_length],
                    branch_start_by_row[0:sequence_length],
                    plan[0:sequence_length, 0:2],
                )
                T.writes(stats[query_index, head_index, 0:2])
                query_segment = segments[query_index]
                branch_start = branch_start_by_row[query_index]
                prefix_limit = T.min(
                    prefix_end, T.Cast("int32", query_index + T.int64(1))
                )
                selected_count = plan[query_index, 0]
                all_masked = plan[query_index, 1] != 0
                maximum = T.float32(-3.402823466e38)
                denominator = T.float32(0.0)
                for tile in T.serial(selected_count):
                    candidate_key = T.if_then_else(
                        (query_segment > 0) & (tile >= prefix_limit),
                        branch_start + tile - prefix_limit,
                        tile,
                    )
                    key_index = T.if_then_else(all_masked, tile, candidate_key)
                    key_segment = segments[key_index]
                    visible = (valid[key_index] != 0) & (
                        (key_segment == query_segment) | (key_segment == 0)
                    )
                    bias = T.if_then_else(
                        all_masked,
                        T.float32(FINITE_FP16_MASK),
                        T.if_then_else(
                            visible,
                            T.float32(0.0),
                            T.float32(FINITE_FP16_MASK),
                        ),
                    )
                    dot = T.float32(0.0)
                    for channel in T.serial(head_dim):
                        dot = dot + T.Cast(
                            "float32", query[query_index, head_index, channel]
                        ) * T.Cast("float32", key[key_index, head_index, channel])
                    score = dot * scale + bias
                    next_maximum = T.max(maximum, score)
                    denominator = denominator * T.exp(maximum - next_maximum) + T.exp(
                        score - next_maximum
                    )
                    maximum = next_maximum
                stats[query_index, head_index, 0] = maximum
                stats[query_index, head_index, 1] = denominator


@T.prim_func(s_tir=True)
def _kev_attention_output(
    query_handle: T.handle,
    key_handle: T.handle,
    value_handle: T.handle,
    segment_handle: T.handle,
    valid_handle: T.handle,
    branch_start_handle: T.handle,
    plan_handle: T.handle,
    stats_handle: T.handle,
    prefix_end: T.int32,
    scale: T.float32,
    sequence_length: T.int64,
    num_heads: T.int64,
    head_dim: T.int64,
    output_handle: T.handle,
):
    """Recompute logits, round probabilities to FP16, then accumulate PV in FP32."""
    query = T.match_buffer(
        query_handle, (sequence_length, num_heads, head_dim), "float16"
    )
    key = T.match_buffer(key_handle, (sequence_length, num_heads, head_dim), "float16")
    value = T.match_buffer(
        value_handle, (sequence_length, num_heads, head_dim), "float16"
    )
    segments = T.match_buffer(segment_handle, (sequence_length,), "int32")
    valid = T.match_buffer(valid_handle, (sequence_length,), "int32")
    branch_start_by_row = T.match_buffer(
        branch_start_handle, (sequence_length,), "int32"
    )
    plan = T.match_buffer(plan_handle, (sequence_length, 2), "int32")
    stats = T.match_buffer(stats_handle, (sequence_length, num_heads, 2), "float32")
    output = T.match_buffer(
        output_handle, (sequence_length, num_heads, head_dim), "float32"
    )
    T.func_attr({"tirx.noalias": True})

    for query_loop in T.serial(sequence_length):
        for head_loop in T.serial(num_heads):
            for lane_loop in T.serial(head_dim):
                with T.sblock("kev_attention_output"):
                    query_index = T.axis.spatial(sequence_length, query_loop)
                    head_index = T.axis.spatial(num_heads, head_loop)
                    lane = T.axis.spatial(head_dim, lane_loop)
                    T.reads(
                        query[0:sequence_length, 0:num_heads, 0:head_dim],
                        key[0:sequence_length, 0:num_heads, 0:head_dim],
                        value[0:sequence_length, 0:num_heads, 0:head_dim],
                        segments[0:sequence_length],
                        valid[0:sequence_length],
                        branch_start_by_row[0:sequence_length],
                        plan[0:sequence_length, 0:2],
                        stats[0:sequence_length, 0:num_heads, 0:2],
                    )
                    T.writes(output[query_index, head_index, 0:head_dim])
                    query_segment = segments[query_index]
                    branch_start = branch_start_by_row[query_index]
                    prefix_limit = T.min(
                        prefix_end, T.Cast("int32", query_index + T.int64(1))
                    )
                    selected_count = plan[query_index, 0]
                    all_masked = plan[query_index, 1] != 0
                    maximum = stats[query_index, head_index, 0]
                    denominator = stats[query_index, head_index, 1]
                    accumulator = T.float32(0.0)
                    for tile in T.serial(selected_count):
                        candidate_key = T.if_then_else(
                            (query_segment > 0) & (tile >= prefix_limit),
                            branch_start + tile - prefix_limit,
                            tile,
                        )
                        key_index = T.if_then_else(all_masked, tile, candidate_key)
                        key_segment = segments[key_index]
                        visible = (valid[key_index] != 0) & (
                            (key_segment == query_segment) | (key_segment == 0)
                        )
                        bias = T.if_then_else(
                            all_masked,
                            T.float32(FINITE_FP16_MASK),
                            T.if_then_else(
                                visible,
                                T.float32(0.0),
                                T.float32(FINITE_FP16_MASK),
                            ),
                        )
                        dot = T.float32(0.0)
                        for channel in T.serial(head_dim):
                            dot = dot + T.Cast(
                                "float32", query[query_index, head_index, channel]
                            ) * T.Cast("float32", key[key_index, head_index, channel])
                        score = dot * scale + bias
                        probability = T.exp(score - maximum) / denominator
                        probability_f16 = T.Cast("float16", probability)
                        accumulator = accumulator + T.Cast(
                            "float32", probability_f16
                        ) * T.Cast("float32", value[key_index, head_index, lane])
                    output[query_index, head_index, lane] = accumulator


def build_kev_attention_prim_funcs() -> tuple[tvm.tirx.PrimFunc, ...]:
    """Return the plan, normalization, and output PrimFuncs for dynamic S."""
    return _kev_attention_plan, _kev_attention_stats, _kev_attention_output


def schedule_kev_attention_for_webgpu(
    prim_funcs: tuple[tvm.tirx.PrimFunc, ...] | None = None,
) -> tuple[tvm.tirx.PrimFunc, ...]:
    """Bind query/head ranges to a 2D grid with one output worker per workgroup."""
    prim_funcs = build_kev_attention_prim_funcs() if prim_funcs is None else prim_funcs
    bindings = (
        ("kev_attention_plan", ("blockIdx.x",)),
        ("kev_attention_stats", ("blockIdx.x", "blockIdx.y")),
        ("kev_attention_output", ("blockIdx.x", "blockIdx.y", None)),
    )
    scheduled = []
    for prim_func, (block_name, thread_tags) in zip(prim_funcs, bindings, strict=True):
        schedule = s_tir.Schedule(prim_func)
        loops = schedule.get_loops(schedule.get_sblock(block_name))
        for loop, thread_tag in zip(loops, thread_tags, strict=True):
            if thread_tag is not None:
                schedule.bind(loop, thread_tag)
        scheduled.append(schedule.mod["main"])
    return tuple(scheduled)
