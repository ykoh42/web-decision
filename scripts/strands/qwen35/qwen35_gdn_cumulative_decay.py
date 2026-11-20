"""Correctly rounded FP32 sums of each 64-token Qwen GDN chunk's nonpositive decays."""

from tvm.script import tirx as T


@T.macro
def _add_magnitude(raw, digits, special):
    exponent = T.Cast("int32", raw >> T.uint32(23))
    if exponent == 255:
        special[0] = T.max(special[0], raw)
    else:
        mantissa = (raw & T.uint32(0x007FFFFF)) | T.if_then_else(
            exponent != 0, T.uint32(0x00800000), T.uint32(0)
        )
        index = T.max(0, exponent - 1)
        digit = index // 16
        shift = T.Cast("uint32", index % 16)
        digits[digit] += (mantissa << shift) & T.uint32(0xFFFF)
        digits[digit + 1] += (mantissa >> (T.uint32(16) - shift)) & T.uint32(0xFFFF)
        if shift > T.uint32(8):
            digits[digit + 2] += mantissa >> (T.uint32(32) - shift)


@T.macro
def _normalize_digits(digits, top):
    high_digit: T.int32 = 0
    for digit in range(18):
        digits[digit + 1] += digits[digit] >> T.uint32(16)
        digits[digit] = digits[digit] & T.uint32(0xFFFF)
        if digits[digit] != T.uint32(0):
            high_digit = digit
    top[0] = high_digit * 16
    for bit in range(1, 16):
        if digits[high_digit] >= (T.uint32(1) << T.Cast("uint32", bit)):
            top[0] = high_digit * 16 + bit


@T.macro
def _round_normal(digits, top, magnitude):
    cut = top - 23
    digit = cut // 16
    shift = T.Cast("uint32", cut % 16)
    significand = (digits[digit] | (digits[digit + 1] << T.uint32(16))) >> shift
    if shift != T.uint32(0):
        significand = significand | (digits[digit + 2] << (T.uint32(32) - shift))
    significand = significand & T.uint32(0x00FFFFFF)
    if cut > 0:
        guard_idx = (cut - 1) // 16
        guard = T.uint32(1) << T.Cast("uint32", (cut - 1) % 16)
        sticky = digits[guard_idx] & (guard - T.uint32(1))
        for j in range(guard_idx):
            sticky = sticky | digits[j]
        if (digits[guard_idx] & guard) != T.uint32(0) and (
            sticky != T.uint32(0) or (significand & T.uint32(1)) != T.uint32(0)
        ):
            significand += T.uint32(1)
    magnitude[0] = T.min(
        T.uint32(0x7F800000), (T.Cast("uint32", cut) << T.uint32(23)) + significand
    )


@T.prim_func(s_tir=True)
def qwen35_gdn_cumsum64(
    batch: T.int32, chunks: T.int32, values: T.handle, result: T.handle
):
    T.func_attr({"tirx.noalias": True})
    decay = T.match_buffer(values, (batch, 16, chunks, 64), "float32")
    cumulative = T.match_buffer(result, (batch, 16, chunks, 64), "float32")
    with T.sblock("root"):
        T.reads()
        T.writes()
        for record in T.thread_binding(batch * 16 * chunks, thread="blockIdx.x"):
            for lane in T.thread_binding(64, thread="threadIdx.x"):
                # Each base-65536 digit receives at most 64 contributions. Eighteen
                # digits cover FP32 magnitudes in units of 2**-149 and their carries.
                # The final two zeros allow a 24-bit read across the highest digit.
                digits = T.alloc_local((20,), "uint32")
                special = T.alloc_local((1,), "uint32")
                top = T.alloc_local((1,), "int32")
                magnitude = T.alloc_local((1,), "uint32")
                for part in range(20):
                    digits[part] = T.uint32(0)
                special[0] = T.uint32(0)
                batch_idx = record // (16 * chunks)
                head = (record // chunks) % 16
                chunk = record % chunks
                for token in range(lane + 1):
                    raw = T.reinterpret(
                        "uint32", decay[batch_idx, head, chunk, token]
                    ) & T.uint32(0x7FFFFFFF)
                    _add_magnitude(raw, digits, special)
                _normalize_digits(digits, top)
                magnitude[0] = digits[0] | (digits[1] << T.uint32(16))
                if top[0] >= 23:
                    _round_normal(digits, top[0], magnitude)
                if special[0] != T.uint32(0):
                    magnitude[0] = special[0]
                signed = T.if_then_else(
                    magnitude[0] != T.uint32(0),
                    magnitude[0] | T.uint32(0x80000000),
                    T.uint32(0),
                )
                cumulative[batch_idx, head, chunk, lane] = T.reinterpret(
                    "float32", signed
                )
