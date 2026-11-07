"""Small query-parallel online-softmax attention prototype for Kev's bias.

This is an isolated codegen and CPU-differential experiment. It does not edit
the package or the runtime. The TIR kernel uses one 256-invocation workgroup
per two query rows and head, and keeps only a bounded QK partial tile plus
scores in workgroup memory. The sequence length is a runtime argument.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import tvm
from tvm.script import ir as I
from tvm.script import tirx as T
from output_safety import create_fresh_output_directory, write_new_text_file


ROOT = Path(__file__).resolve().parent
HEADS = 16
HEAD_DIM = 128
QUERY_TILE = 2
KEY_TILE = int(os.environ.get("KEY_TILE", "64"))
if KEY_TILE not in (32, 64, 128) or HEAD_DIM % KEY_TILE != 0:
    raise ValueError("KEY_TILE must be 32, 64, or 128 for head dimension 128")
REDUCE_LANES = HEAD_DIM // KEY_TILE
SCALE = np.float16(0.08837890625)
MASKED_BIAS = np.float16(-65504.0)
NEG_INFINITY = -3.402823466e38


@I.ir_module(s_tir=True)
class OnlineAttention:
    @T.prim_func(s_tir=True)
    def main(
        q_handle: T.handle,
        k_handle: T.handle,
        v_handle: T.handle,
        bias_handle: T.handle,
        out_handle: T.handle,
        seq_len: T.int32,
    ):
        q = T.match_buffer(q_handle, (1, HEADS, seq_len, HEAD_DIM), "float16")
        k = T.match_buffer(k_handle, (1, HEADS, seq_len, HEAD_DIM), "float16")
        v = T.match_buffer(v_handle, (1, HEADS, seq_len, HEAD_DIM), "float16")
        bias = T.match_buffer(bias_handle, (1, seq_len, seq_len), "float16")
        out = T.match_buffer(out_handle, (1, HEADS, seq_len, HEAD_DIM), "float16")

        for query_block in T.thread_binding(
            T.ceildiv(seq_len, QUERY_TILE), thread="blockIdx.x"
        ):
            for head in T.thread_binding(HEADS, thread="blockIdx.y"):
                for query_lane in T.thread_binding(QUERY_TILE, thread="threadIdx.y"):
                    for lane in T.thread_binding(HEAD_DIM, thread="threadIdx.x"):
                        with T.sblock("attention_workgroup"):
                            T.reads()
                            T.writes()
                            partial = T.alloc_buffer(
                                (QUERY_TILE, KEY_TILE, REDUCE_LANES),
                                "float32",
                                scope="shared",
                            )
                            scores = T.alloc_buffer(
                                (QUERY_TILE, KEY_TILE), "float32", scope="shared"
                            )
                            state = T.alloc_buffer((3,), "float32", scope="local")
                            dot = T.alloc_buffer((1,), "float32", scope="local")
                            qk = T.alloc_buffer((1,), "float32", scope="local")
                            tile_max = T.alloc_buffer((1,), "float32", scope="local")
                            next_max = T.alloc_buffer((1,), "float32", scope="local")
                            weight_sum = T.alloc_buffer((1,), "float32", scope="local")
                            alpha = T.alloc_buffer((1,), "float32", scope="local")
                            value_sum = T.alloc_buffer((1,), "float32", scope="local")

                            row = query_block * QUERY_TILE + query_lane
                            query_row = T.min(row, seq_len - 1)

                            # Every lane maintains one output component and a
                            # redundant scalar softmax state for this row.
                            state[0] = NEG_INFINITY  # running maximum
                            state[1] = 0.0            # running denominator
                            state[2] = 0.0            # running numerator

                            for tile in T.serial(T.ceildiv(seq_len, KEY_TILE)):
                                # Reduction lanes cooperate on each key's
                                # 128-wide dot product.
                                key_in_tile = lane // REDUCE_LANES
                                reduce_lane = lane % REDUCE_LANES
                                key = tile * KEY_TILE + key_in_tile
                                dot[0] = 0.0
                                for chunk in T.serial(HEAD_DIM // REDUCE_LANES):
                                    dim = reduce_lane * (HEAD_DIM // REDUCE_LANES) + chunk
                                    if key < seq_len:
                                        dot[0] = dot[0] + T.cast(
                                            q[0, head, query_row, dim], "float32"
                                        ) * T.cast(k[0, head, key, dim], "float32")
                                partial[query_lane, key_in_tile, reduce_lane] = dot[0]
                                T.tvm_storage_sync("shared")

                                # One lane group reduces the partial sums for
                                # each key.
                                if lane < KEY_TILE:
                                    qk[0] = 0.0
                                    for reduce_i in T.serial(REDUCE_LANES):
                                        qk[0] = qk[0] + partial[
                                            query_lane, lane, reduce_i
                                        ]
                                    # Match the model's fp16 QK, scale, and bias
                                    # rounding before the online softmax.
                                    qk[0] = T.cast(T.cast(qk[0], "float16"), "float32")
                                    qk[0] = T.cast(
                                        T.cast(
                                            T.cast(qk[0], "float16")
                                            * T.cast(SCALE, "float16"),
                                            "float16",
                                        ),
                                        "float32",
                                    )
                                    key = tile * KEY_TILE + lane
                                    if key < seq_len:
                                        # Keep the finite -65504 bias. Kev's
                                        # contract is additive fp16 bias, not
                                        # a hard -infinity mask.
                                        qk[0] = T.cast(
                                            T.cast(qk[0], "float16")
                                            + bias[0, query_row, key],
                                            "float16",
                                        )
                                        scores[query_lane, lane] = T.cast(
                                            qk[0], "float32"
                                        )
                                    else:
                                        scores[query_lane, lane] = NEG_INFINITY
                                T.tvm_storage_sync("shared")

                                # Use the finite additive bias exactly. If a
                                # tile has only masked keys, its mass is later
                                # rescaled with the same online recurrence.
                                tile_max[0] = NEG_INFINITY
                                for key_lane in T.serial(KEY_TILE):
                                    key = tile * KEY_TILE + key_lane
                                    if key < seq_len:
                                        tile_max[0] = T.max(
                                            tile_max[0], scores[query_lane, key_lane]
                                        )
                                next_max[0] = T.max(state[0], tile_max[0])
                                if state[1] == 0.0:
                                    alpha[0] = 0.0
                                else:
                                    alpha[0] = T.exp(state[0] - next_max[0])

                                weight_sum[0] = 0.0
                                value_sum[0] = 0.0
                                for key_lane in T.serial(KEY_TILE):
                                    key = tile * KEY_TILE + key_lane
                                    if key < seq_len:
                                        weight = T.exp(
                                            scores[query_lane, key_lane] - next_max[0]
                                        )
                                        weight_sum[0] = weight_sum[0] + weight
                                        value_sum[0] = value_sum[0] + weight * T.cast(
                                            v[0, head, key, lane], "float32"
                                        )
                                state[2] = state[2] * alpha[0] + value_sum[0]
                                state[1] = state[1] * alpha[0] + weight_sum[0]
                                state[0] = next_max[0]

                                # The next iteration's first barrier waits for
                                # every lane to finish reading scores before
                                # the score writers overwrite that tile.

                            if row < seq_len:
                                out[0, head, row, lane] = T.cast(
                                    state[2] / state[1], "float16"
                                )


def make_bias(input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Return the exact Kev additive block-causal bias for one sequence."""
    seq_len = len(input_ids)
    segment = np.cumsum(
        (input_ids == 151660) & (attention_mask != 0), dtype=np.int32
    )
    result = np.full((seq_len, seq_len), MASKED_BIAS, dtype=np.float16)
    for row in range(seq_len):
        for col in range(seq_len):
            visible = (
                col <= row
                and attention_mask[col] != 0
                and (segment[row] == segment[col] or segment[col] == 0)
            ) or row == col
            if visible:
                result[row, col] = np.float16(0)
    return result


def full_f16_reference(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    """Full S-by-S fp16 reference, with explicit rounding at each ONNX op."""
    seq_len, heads, dim = q.shape[2], q.shape[1], q.shape[3]
    output = np.empty_like(q)
    for head in range(heads):
        qk = np.matmul(q[0, head], k[0, head].T).astype(np.float16)
        scaled = (qk * SCALE).astype(np.float16)
        logits = (scaled + bias).astype(np.float16)
        for row in range(seq_len):
            row_logits = logits[row].astype(np.float32)
            row_logits -= np.max(row_logits)
            probs = np.exp(row_logits).astype(np.float32)
            probs /= np.sum(probs, dtype=np.float32)
            output[0, head, row] = np.matmul(
                probs.astype(np.float16), v[0, head]
            ).astype(np.float16)
    return output


def online_f32_differential(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    """CPU simulation of the TIR online recurrence for differential checking."""
    seq_len, heads, dim = q.shape[2], q.shape[1], q.shape[3]
    result = np.empty_like(q)
    for head in range(heads):
        for row in range(seq_len):
            max_seen = -np.inf
            denom = np.float32(0)
            numerator = np.zeros((dim,), dtype=np.float32)
            for start in range(0, seq_len, KEY_TILE):
                end = min(start + KEY_TILE, seq_len)
                # Match TIR's fp16 rounding at QK output, scale, and bias add.
                dots = np.matmul(
                    k[0, head, start:end].astype(np.float32),
                    q[0, head, row].astype(np.float32),
                ).astype(np.float16)
                scaled = (dots * SCALE).astype(np.float16)
                scores = (scaled + bias[row, start:end]).astype(np.float16)
                scores32 = scores.astype(np.float32)
                new_max = max(max_seen, float(np.max(scores32)))
                alpha = np.float32(0) if denom == 0 else np.exp(
                    np.float32(max_seen - new_max)
                )
                weights = np.exp(scores32 - np.float32(new_max)).astype(np.float32)
                numerator = numerator * alpha + np.matmul(
                    weights, v[0, head, start:end].astype(np.float32)
                )
                denom = denom * alpha + np.sum(weights, dtype=np.float32)
                max_seen = new_max
            result[0, head, row] = (numerator / denom).astype(np.float16)
    return result


def run_cpu_differential() -> dict:
    rng = np.random.default_rng(8128)
    seq_len = 17
    input_ids = np.array(
        [151659, 151660, 7, 8, 9, 10, 11, 12, 151660, 151660,
         151648, 9834, 151649, 151661, 13, 151660, 151661]
    )
    attention_mask = np.ones(seq_len, dtype=np.int32)
    attention_mask[-1] = 0
    bias = make_bias(input_ids, attention_mask)
    q = rng.normal(0, 0.3, (1, HEADS, seq_len, HEAD_DIM)).astype(np.float16)
    k = rng.normal(0, 0.3, (1, HEADS, seq_len, HEAD_DIM)).astype(np.float16)
    v = rng.normal(0, 0.3, (1, HEADS, seq_len, HEAD_DIM)).astype(np.float16)
    reference = full_f16_reference(q, k, v, bias)
    candidate = online_f32_differential(q, k, v, bias)
    difference = np.abs(reference.astype(np.float32) - candidate.astype(np.float32))
    return {
        "sequence_length": seq_len,
        "heads": HEADS,
        "head_dim": HEAD_DIM,
        "key_tile": KEY_TILE,
        "query_tile": QUERY_TILE,
        "delimiter_positions": np.flatnonzero(input_ids == 151660).tolist(),
        "delimiter_at_key_tile_boundary": [8, 9],
        "padding_position": seq_len - 1,
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "exact_fp16_elements": int(np.count_nonzero(difference == 0)),
        "total_elements": int(difference.size),
        "tolerance": 0.002,
        "within_tolerance": bool(np.max(difference) <= 0.002),
    }


def run_webgpu_codegen(output_directory: Path) -> dict:
    executable = tvm.compile(OnlineAttention, target="webgpu")
    imported = executable.mod.imports[0]
    source = imported.inspect_source("wgsl")
    assert f"@workgroup_size({HEAD_DIM}, {QUERY_TILE}, 1)" in source
    assert (
        f"var<workgroup> partial : array<f32, {QUERY_TILE * KEY_TILE * REDUCE_LANES}>;"
        in source
    )
    assert f"var<workgroup> scores : array<f32, {QUERY_TILE * KEY_TILE}>;" in source
    barrier_count = source.count("workgroupBarrier();")
    if barrier_count != 2:
        raise AssertionError(f"Expected two workgroup barriers per key tile, found {barrier_count}")
    lines = source.splitlines()
    tile_loop = next(i for i, line in enumerate(lines) if "for (var tile : i32" in line)
    barriers = [i for i, line in enumerate(lines) if "workgroupBarrier();" in line]
    partial_stores = [
        i
        for i, line in enumerate(lines)
        if line.lstrip().startswith("partial[") and "] = dot[0i];" in line
    ]
    score_stores = [
        i
        for i, line in enumerate(lines)
        if line.lstrip().startswith("scores[") and "] = " in line
    ]
    tile_max_updates = [
        i
        for i, line in enumerate(lines)
        if "tile_max[0i] =" in line and i > barriers[1]
    ]
    loop_body_indent = len(lines[tile_loop]) - len(lines[tile_loop].lstrip()) + 2
    if (
        len(partial_stores) != 1
        or len(score_stores) < 1
        or len(tile_max_updates) < 1
        or not tile_loop < partial_stores[0] < barriers[0]
        or not barriers[0] < min(score_stores) <= max(score_stores) < barriers[1]
        or not barriers[1] < tile_max_updates[0]
        or any(len(lines[index]) - len(lines[index].lstrip()) != loop_body_indent for index in barriers)
    ):
        raise AssertionError("WGSL barriers are not uniform and ordered around shared-tile reuse")
    static_float_arrays = [
        int(size) for size in re.findall(r"array<f32,\s*(\d+)>", source)
    ]
    assert max(static_float_arrays) == QUERY_TILE * KEY_TILE * REDUCE_LANES
    source_file = f"kev_online_attention_k{KEY_TILE}.wgsl"
    write_new_text_file(output_directory / source_file, source)
    return {
        "target": "webgpu",
        "source_file": source_file,
        "source_bytes": len(source.encode()),
        "has_workgroup_barrier": "workgroupBarrier" in source,
        "has_workgroup_storage": "var<workgroup>" in source,
        "workgroup_barriers_per_key_tile": barrier_count,
        "full_attention_score_shape": None,
        "retained_input_bias_shape": "[1, seq_len, seq_len] float16",
        "workgroup_size": [HEAD_DIM, QUERY_TILE, 1],
        "workgroups_per_dispatch": "ceil(seq_len / 2) * 16",
        "workgroup_scratch_bytes": QUERY_TILE
        * (KEY_TILE * REDUCE_LANES + KEY_TILE)
        * 4,
        "workgroup_scratch_float_elements": QUERY_TILE
        * (KEY_TILE * REDUCE_LANES + KEY_TILE),
        "largest_static_float_array_elements": max(static_float_arrays),
        "max_qk_products_per_invocation_per_key_tile": HEAD_DIM // REDUCE_LANES,
        "max_pv_terms_per_invocation_per_key_tile": KEY_TILE,
        "seq_len_dynamic_argument": True,
        "s8192_key_tiles_per_query": 8192 // KEY_TILE,
        "s8192_workgroups_per_dispatch": (8192 // QUERY_TILE) * HEADS,
        "s8192_barrier_instances_across_28_layers": (
            (8192 // QUERY_TILE)
            * HEADS
            * (8192 // KEY_TILE)
            * barrier_count
            * 28
        ),
        "gpu_inference_run": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="A new directory for generated WGSL and report files.",
    )
    args = parser.parse_args()
    output_directory = create_fresh_output_directory(args.output_dir)
    differential = run_cpu_differential()
    codegen = run_webgpu_codegen(output_directory)
    report = {
        "cpu_differential": differential,
        "webgpu_codegen": codegen,
        "numerical_note": (
            "The online path rounds each QK result, scale, and bias addition to fp16, "
            "then accumulates softmax state and PV numerator in fp32. It changes "
            "softmax/PV reduction order versus full-matrix reduction."
        ),
    }
    write_new_text_file(output_directory / "report.json", json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
