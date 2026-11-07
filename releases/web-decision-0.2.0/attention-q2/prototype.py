from __future__ import annotations

from functools import cache

import numpy as np
import tvm
from tvm.script import tirx as T


MAX_BUFFER_SIZE = 2_147_483_644
HEADS = 16
HEAD_SIZE = 128
DTYPE_BYTES = 2
DEFAULT_TILE_ROWS = 64
ATTENTION_SCALE = np.float16(0.08837890625)
FIM_MIDDLE_TOKEN_ID = 151660
MASKED_SCORE = np.float16(-65504.0)


def choose_tile_rows(sequence_length: int, max_buffer_size: int) -> int:
    if sequence_length <= 0 or max_buffer_size <= 0:
        raise ValueError("Sequence length and buffer size must be positive")
    rows_per_buffer = max_buffer_size // (HEADS * sequence_length * DTYPE_BYTES)
    if rows_per_buffer == 0:
        raise ValueError("One query row exceeds the device buffer limit")
    return min(DEFAULT_TILE_ROWS, rows_per_buffer)


def tiled_scratch_bytes(sequence_length: int, tile_rows: int) -> int:
    return HEADS * tile_rows * sequence_length * DTYPE_BYTES


def kev_additive_bias(input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Rebuild the ONNX block-causal bias for one unpadded model input."""
    input_ids = np.asarray(input_ids)
    attention_mask = np.asarray(attention_mask)
    if input_ids.ndim != 1 or attention_mask.shape != input_ids.shape:
        raise ValueError("Expected equally sized one-dimensional input arrays")

    valid = attention_mask != 0
    starts = (input_ids == FIM_MIDDLE_TOKEN_ID) & valid
    segment = np.cumsum(starts, dtype=np.int64)
    length = len(input_ids)
    row = np.arange(length)[:, None]
    column = np.arange(length)[None, :]
    visible = (
        (column <= row)
        & valid[None, :]
        & ((segment[:, None] == segment[None, :]) | (segment[None, :] == 0))
    ) | (row == column)

    bias = np.full((1, 1, length, length), MASKED_SCORE, dtype="float16")
    bias[0, 0, visible] = np.float16(0.0)
    return bias


@cache
def attention_tile_primfunc():
    source = '''
@T.prim_func(s_tir=True)
def kev_attention_tile[tile_rows, sequence_length](
    query: T.Buffer((T.int64(1), T.int64(16), tile_rows, T.int64(128)), "float16"),
    key: T.Buffer((T.int64(1), T.int64(16), T.int64(128), sequence_length), "float16"),
    bias: T.Buffer((T.int64(1), T.int64(1), tile_rows, sequence_length), "float16"),
    value: T.Buffer((T.int64(1), T.int64(16), sequence_length, T.int64(128)), "float16"),
    output: T.Buffer((T.int64(1), T.int64(16), tile_rows, T.int64(128)), "float16"),
):
    score_row = T.alloc_buffer((sequence_length,), "float16", scope="local")
    for query_head in T.serial(tile_rows * T.int64(16)):
        query_row = query_head // T.int64(16)
        head = query_head % T.int64(16)
        max_score = T.float16(-65504.0)
        for key_row in T.serial(sequence_length):
            dot = T.float16(0.0)
            for channel in T.serial(T.int64(128)):
                dot = T.Cast(
                    "float16",
                    dot + query[T.int64(0), head, query_row, channel]
                    * key[T.int64(0), head, channel, key_row],
                )
            score = T.Cast(
                "float16",
                dot * T.float16(0.08837890625)
                + bias[T.int64(0), T.int64(0), query_row, key_row],
            )
            score_row[key_row] = score
            max_score = T.max(max_score, score)
        denominator = T.float16(0.0)
        for key_row in T.serial(sequence_length):
            weight = T.exp(score_row[key_row] - max_score)
            score_row[key_row] = weight
            denominator = T.Cast("float16", denominator + weight)
        for channel in T.serial(T.int64(128)):
            accumulator = T.float16(0.0)
            for key_row in T.serial(sequence_length):
                probability = T.Cast("float16", score_row[key_row] / denominator)
                accumulator = T.Cast(
                    "float16",
                    accumulator + probability * value[T.int64(0), head, key_row, channel],
                )
            output[T.int64(0), head, query_row, channel] = accumulator
'''
    return tvm.script.from_source(source, extra_vars={"T": T})


@cache
def _attention_tile_cpu_module():
    return tvm.build(attention_tile_primfunc(), target="llvm")


def run_query_tiled_attention(
    query: np.ndarray,
    key: np.ndarray,
    bias: np.ndarray,
    value: np.ndarray,
    tile_rows: int,
) -> np.ndarray:
    sequence_length = query.shape[2]
    if tile_rows <= 0:
        raise ValueError("Tile row count must be positive")

    device = tvm.cpu(0)
    run_tile = _attention_tile_cpu_module()["kev_attention_tile"]
    output_tiles = []
    for start in range(0, sequence_length, tile_rows):
        end = min(start + tile_rows, sequence_length)
        query_tile = query[:, :, start:end, :]
        bias_tile = bias[:, :, start:end, :]
        output_tile = tvm.runtime.empty(
            (1, HEADS, end - start, HEAD_SIZE), "float16", device
        )
        run_tile(
            tvm.runtime.tensor(query_tile, device),
            tvm.runtime.tensor(key, device),
            tvm.runtime.tensor(bias_tile, device),
            tvm.runtime.tensor(value, device),
            output_tile,
        )
        output_tiles.append(output_tile.numpy())
    return np.concatenate(output_tiles, axis=2)


def full_attention_reference(
    query: np.ndarray,
    key: np.ndarray,
    bias: np.ndarray,
    value: np.ndarray,
) -> np.ndarray:
    scores = np.matmul(query, key).astype("float16")
    scores = (scores * ATTENTION_SCALE).astype("float16")
    scores = (scores + bias).astype("float16")
    maximum = np.max(scores, axis=-1, keepdims=True)
    weights = np.exp((scores - maximum).astype("float16")).astype("float16")
    denominator = np.sum(weights, axis=-1, keepdims=True, dtype="float16")
    probabilities = (weights / denominator).astype("float16")
    return np.matmul(probabilities, value).astype("float16")
