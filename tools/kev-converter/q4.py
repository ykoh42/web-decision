"""Packed Q4 formats and Relax TE kernels used by the direct Kev TVM build."""

import numpy as np
from tvm import te, tirx


def pack_nibbles(values: np.ndarray) -> np.ndarray:
    """Pack signed two's-complement Q4 values, low nibble first."""
    codes = np.asarray(values, dtype=np.int8)
    if codes.ndim != 2 or codes.shape[1] % 8:
        raise ValueError("Q4 rows must be two-dimensional and divisible by eight.")
    rows, width = codes.shape
    nibbles = codes.reshape(rows, width // 8, 8).astype(np.uint8) & np.uint8(15)
    packed = np.zeros((rows, width // 8), dtype=np.uint32)
    for shift in range(8):
        packed |= nibbles[:, :, shift].astype(np.uint32) << np.uint32(shift * 4)
    return packed


def pack_offset_nibbles(values: np.ndarray) -> np.ndarray:
    """Pack unsigned Q4 codes, low nibble first."""
    codes = np.asarray(values, dtype=np.uint8)
    if codes.ndim != 2 or codes.shape[1] % 8:
        raise ValueError("Q4 rows must be two-dimensional and divisible by eight.")
    if np.any(codes > 15):
        raise ValueError("Unsigned Q4 codes must be in 0..15.")
    rows, width = codes.shape
    nibbles = codes.reshape(rows, width // 8, 8).astype(np.uint32)
    packed = np.zeros((rows, width // 8), dtype=np.uint32)
    for shift in range(8):
        packed |= nibbles[:, :, shift] << np.uint32(shift * 4)
    return packed


def unpack_nibbles(packed: np.ndarray) -> np.ndarray:
    words = np.asarray(packed, dtype=np.uint32)
    if words.ndim != 2:
        raise ValueError("Packed Q4 weights must be two-dimensional.")
    rows, word_count = words.shape
    codes = np.empty((rows, word_count * 8), dtype=np.int8)
    for shift in range(8):
        nibble = (words >> np.uint32(shift * 4)) & np.uint32(15)
        signed = nibble.astype(np.int8)
        signed[signed >= 8] -= 16
        codes[:, shift::8] = signed
    return codes


def unpack_offset_nibbles(packed: np.ndarray) -> np.ndarray:
    """Unpack unsigned Q4 codes without applying a zero-point offset."""
    words = np.asarray(packed, dtype=np.uint32)
    if words.ndim != 2:
        raise ValueError("Packed Q4 weights must be two-dimensional.")
    codes = np.empty((words.shape[0], words.shape[1] * 8), dtype=np.uint8)
    for shift in range(8):
        codes[:, shift::8] = ((words >> np.uint32(shift * 4)) & np.uint32(15)).astype(
            np.uint8
        )
    return codes


def quantize_rows(
    values: np.ndarray, block_size: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize signed embedding rows using the GatherBlockQuantized scale rule."""
    rows = np.asarray(values, dtype=np.float16)
    if rows.ndim != 2 or block_size % 8 or rows.shape[1] % block_size:
        raise ValueError(
            "Q4 weights require 2D rows aligned to 8-value, 32-value blocks."
        )
    groups = rows.reshape(
        rows.shape[0], rows.shape[1] // block_size, block_size
    ).astype(np.float32)
    maxima = np.max(groups, axis=2)
    minima = np.min(groups, axis=2)
    signed_extreme = np.where(np.abs(maxima) > np.abs(minima), maxima, minima)
    scales = (signed_extreme / -8).astype(np.float16)
    divisors = scales.astype(np.float32)
    divisors = np.where(divisors != 0, divisors, 1.0)
    codes = np.rint(groups / divisors[:, :, None]).clip(-8, 7).astype(np.int8)
    return pack_nibbles(codes.reshape(rows.shape)), scales


def quantize_matmul_rows(
    values: np.ndarray, block_size: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize MatMulNBits rows as offset uint4 using signed-extreme scales."""
    rows = np.asarray(values, dtype=np.float16)
    if rows.ndim != 2 or block_size % 8 or rows.shape[1] % block_size:
        raise ValueError(
            "Q4 weights require 2D rows aligned to 8-value, 32-value blocks."
        )
    groups = rows.reshape(
        rows.shape[0], rows.shape[1] // block_size, block_size
    ).astype(np.float32)
    maxima = np.max(groups, axis=2)
    minima = np.min(groups, axis=2)
    signed_extreme = np.where(np.abs(maxima) > np.abs(minima), maxima, minima)
    precise_scales = signed_extreme / -8
    scales = precise_scales.astype(np.float16)
    reciprocal_scales = np.divide(
        np.float32(1),
        precise_scales,
        out=np.zeros_like(precise_scales, dtype=np.float32),
        where=precise_scales != 0,
    )
    shifted_codes = groups * reciprocal_scales[:, :, None] + np.float32(8)
    unsigned_codes = (
        np.floor(shifted_codes + np.float32(0.5)).clip(0, 15).astype(np.uint8)
    )
    return pack_offset_nibbles(unsigned_codes.reshape(rows.shape)), scales


def dequantize_rows(
    packed: np.ndarray, scales: np.ndarray, block_size: int = 32
) -> np.ndarray:
    codes = unpack_nibbles(packed)
    if codes.shape[1] % block_size or scales.shape != (
        codes.shape[0],
        codes.shape[1] // block_size,
    ):
        raise ValueError("Q4 scales do not match the packed rows.")
    groups = codes.reshape(
        codes.shape[0], codes.shape[1] // block_size, block_size
    ).astype(np.float32)
    return (groups * np.asarray(scales, dtype=np.float32)[:, :, None]).reshape(
        codes.shape
    )


def dequantize_matmul_rows(
    packed: np.ndarray, scales: np.ndarray, block_size: int = 32
) -> np.ndarray:
    codes = unpack_offset_nibbles(packed).astype(np.int16) - 8
    if codes.shape[1] % block_size or scales.shape != (
        codes.shape[0],
        codes.shape[1] // block_size,
    ):
        raise ValueError("MatMulNBits scales do not match the packed rows.")
    groups = codes.reshape(
        codes.shape[0], codes.shape[1] // block_size, block_size
    ).astype(np.float32)
    return (groups * np.asarray(scales, dtype=np.float32)[:, :, None]).reshape(
        codes.shape
    )


def matmul_te(x, packed, scales, block_size: int = 32):
    rows, width = x.shape
    width = int(width)
    columns = int(packed.shape[0])
    if (
        int(packed.shape[1]) != width // 8
        or int(scales.shape[1]) != width // block_size
    ):
        raise ValueError("Packed Q4 dimensions do not match the matmul input.")
    reduction = te.reduce_axis((0, width), "q4_k")

    def multiply(row, column, index):
        shift = tirx.Cast("uint32", (index % 8) * 4)
        nibble = tirx.bitwise_and(
            tirx.shift_right(packed[column, index // 8], shift),
            tirx.const(15, "uint32"),
        )
        value = tirx.Cast("int32", nibble) - tirx.const(8, "int32")
        scale = tirx.Cast("float32", scales[column, index // block_size])
        weight = tirx.Cast("float32", value) * scale
        return tirx.Cast("float32", x[row, index]) * weight

    return te.compute(
        (rows, columns),
        lambda row, column: te.sum(multiply(row, column, reduction), axis=reduction),
        name="kev_q4_matmul",
    )


def gather_te(packed, token_ids, scales, block_size: int = 32):
    length = token_ids.shape[0]
    rows, words = map(int, packed.shape)
    width = words * 8
    if int(scales.shape[0]) != rows or int(scales.shape[1]) != width // block_size:
        raise ValueError("Packed Q4 embedding scales do not match the embedding rows.")

    def value(position, column):
        token = tirx.Cast("int32", token_ids[position])
        shift = tirx.Cast("uint32", (column % 8) * 4)
        nibble = tirx.bitwise_and(
            tirx.shift_right(packed[token, column // 8], shift),
            tirx.const(15, "uint32"),
        )
        signed = tirx.if_then_else(
            nibble < tirx.const(8, "uint32"),
            tirx.Cast("int32", nibble),
            tirx.Cast("int32", nibble) - tirx.const(16, "int32"),
        )
        scale = scales[token, column // block_size]
        return tirx.Cast(
            "float16", tirx.Cast("float32", signed) * tirx.Cast("float32", scale)
        )

    return te.compute((length, width), value, name="kev_q4_gather")
