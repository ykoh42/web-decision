"""Local FP32 prefix-bidirectional adaptation of the earlier Gemma tiled probe."""

from tvm.script import ir as I
from tvm.script import tirx as T

BATCH_SIZE = 1
QUERY_HEADS = 12
KEY_VALUE_HEADS = 12
M4_WEBGPU_SHARED_BYTES = 32768
WEBGPU_MAX_WORKGROUP_INVOCATIONS = 256
NEGATIVE_SCORE_SENTINEL = -1.0e30


@T.macro
def score_reduction_yes(
    workgroup_tile: T.Buffer,
    workgroup_tile_alt: T.Buffer,
    key_index: T.int32,
    feature_index: T.int32,
    key_allowed: T.bool,
):
    for reduction_level in T.serial(4):
        reduction_stride = T.shift_right(16, reduction_level + 1)
        other_lane = T.if_then_else(
            feature_index < reduction_stride,
            feature_index + reduction_stride,
            feature_index,
        )
        if reduction_level % 2 == 0:
            current = workgroup_tile[key_index, feature_index]
            other = workgroup_tile[key_index, other_lane]
            workgroup_tile_alt[key_index, feature_index] = T.if_then_else(
                feature_index < reduction_stride,
                current + other,
                current,
            )
        else:
            current = workgroup_tile_alt[key_index, feature_index]
            other = workgroup_tile_alt[key_index, other_lane]
            workgroup_tile[key_index, feature_index] = T.if_then_else(
                feature_index < reduction_stride,
                current + other,
                current,
            )
        T.tvm_storage_sync("shared")
    if feature_index == 0:
        workgroup_tile_alt[key_index, 0] = T.if_then_else(
            key_allowed,
            workgroup_tile[key_index, 0],
            T.float32(NEGATIVE_SCORE_SENTINEL),
        )
    workgroup_tile_alt[key_index, 1] = T.cast(key_allowed, "float32")


@T.macro
def score_reduction_no(
    workgroup_tile: T.Buffer,
    key_index: T.int32,
    feature_index: T.int32,
    feature_lanes: T.int32,
    key_allowed: T.bool,
):
    for reduction_level in T.serial(4):
        reduction_stride = T.shift_right(16, reduction_level + 1)
        current = workgroup_tile[
            0,
            key_index * feature_lanes + feature_index,
        ]
        other_lane = T.if_then_else(
            feature_index < reduction_stride,
            feature_index + reduction_stride,
            feature_index,
        )
        other = workgroup_tile[
            0,
            key_index * feature_lanes + other_lane,
        ]
        workgroup_tile[
            0,
            key_index * feature_lanes + feature_index,
        ] = T.if_then_else(
            feature_index < reduction_stride,
            current + other,
            current,
        )
        T.tvm_storage_sync("shared")

    if feature_index == 0:
        workgroup_tile[1, key_index * feature_lanes] = T.if_then_else(
            key_allowed,
            workgroup_tile[
                0,
                key_index * feature_lanes,
            ],
            T.float32(NEGATIVE_SCORE_SENTINEL),
        )
    if feature_index == 1:
        workgroup_tile[
            1,
            key_index * feature_lanes + 1,
        ] = T.cast(key_allowed, "float32")


@T.macro
def value_reduction_yes(
    workgroup_tile: T.Buffer,
    workgroup_tile_alt: T.Buffer,
    key_index: T.int32,
    feature_index: T.int32,
    features_per_lane: T.int32,
    key_tile_size: T.int32,
):
    for reduction_level in T.serial(key_tile_size.bit_length() - 1):
        reduction_stride = T.shift_right(
            key_tile_size,
            reduction_level + 1,
        )
        other_lane = T.if_then_else(
            key_index < reduction_stride,
            key_index + reduction_stride,
            key_index,
        )
        if reduction_level % 2 == 0:
            for feature_offset in T.serial(features_per_lane):
                feature = feature_index * features_per_lane + feature_offset
                current = workgroup_tile[key_index, feature]
                other = workgroup_tile[other_lane, feature]
                workgroup_tile_alt[key_index, feature] = T.if_then_else(
                    key_index < reduction_stride,
                    current + other,
                    current,
                )
        else:
            for feature_offset in T.serial(features_per_lane):
                feature = feature_index * features_per_lane + feature_offset
                current = workgroup_tile_alt[key_index, feature]
                other = workgroup_tile_alt[other_lane, feature]
                workgroup_tile[key_index, feature] = T.if_then_else(
                    key_index < reduction_stride,
                    current + other,
                    current,
                )
        T.tvm_storage_sync("shared")


@T.macro
def value_reduction_no(
    workgroup_tile: T.Buffer,
    key_index: T.int32,
    feature_index: T.int32,
    features_per_lane: T.int32,
    key_tile_size: T.int32,
):
    for reduction_level in T.serial(key_tile_size.bit_length() - 1):
        reduction_stride = T.shift_right(
            key_tile_size,
            reduction_level + 1,
        )
        other_lane = T.if_then_else(
            key_index < reduction_stride,
            key_index + reduction_stride,
            key_index,
        )
        for feature_offset in T.serial(features_per_lane):
            feature = feature_index * features_per_lane + feature_offset
            current = workgroup_tile[key_index, feature]
            other = workgroup_tile[other_lane, feature]
            workgroup_tile[key_index, feature] = T.if_then_else(
                key_index < reduction_stride,
                current + other,
                current,
            )
        T.tvm_storage_sync("shared")


@T.macro
def update_tile_max(
    workgroup_tile: T.Buffer,
    workgroup_tile_alt: T.Buffer,
    tile_max: T.Buffer,
    tile_valid: T.Buffer,
    key_tile_size: T.int32,
    feature_lanes: T.int32,
    use_ping_pong: T.bool,
):
    for score_lane in T.serial(key_tile_size):
        if use_ping_pong:
            tile_score = workgroup_tile_alt[score_lane, 0]
            score_valid = workgroup_tile_alt[score_lane, 1] != T.float32(0.0)
        else:
            tile_score = workgroup_tile[1, score_lane * feature_lanes]
            score_valid = workgroup_tile[
                1,
                score_lane * feature_lanes + 1,
            ] != T.float32(0.0)
        if score_valid:
            tile_max[0] = T.max(tile_max[0], tile_score)
            tile_valid[0] = 1


@T.macro
def update_tile_denominator(
    workgroup_tile: T.Buffer,
    workgroup_tile_alt: T.Buffer,
    tile_denominator: T.Buffer,
    next_max: T.float32,
    key_tile_size: T.int32,
    feature_lanes: T.int32,
    use_ping_pong: T.bool,
):
    for score_lane in T.serial(key_tile_size):
        if use_ping_pong:
            tile_score = workgroup_tile_alt[score_lane, 0]
            score_valid = workgroup_tile_alt[score_lane, 1] != T.float32(0.0)
        else:
            tile_score = workgroup_tile[1, score_lane * feature_lanes]
            score_valid = workgroup_tile[
                1,
                score_lane * feature_lanes + 1,
            ] != T.float32(0.0)
        if score_valid:
            tile_denominator[0] = tile_denominator[0] + T.exp(tile_score - next_max)


@T.macro
def compute_dot(
    query: T.Buffer,
    key: T.Buffer,
    dot_partial: T.Buffer,
    key_allowed: T.bool,
    key_position: T.int32,
    position: T.int32,
    head: T.int32,
    feature_index: T.int32,
    feature_lanes: T.int32,
    features_per_lane: T.int32,
):
    if key_allowed:
        for feature_group in T.serial(features_per_lane):
            feature = feature_group * feature_lanes + feature_index
            dot_partial[0] = dot_partial[0] + T.cast(
                query[
                    0,
                    position,
                    head,
                    feature,
                ],
                "float32",
            ) * T.cast(
                key[
                    0,
                    key_position,
                    head,
                    feature,
                ],
                "float32",
            )


@T.macro
def accumulate_value(
    accumulator: T.Buffer,
    workgroup_tile: T.Buffer,
    workgroup_tile_alt: T.Buffer,
    feature_index: T.int32,
    features_per_lane: T.int32,
    key_index: T.int32,
    key_tile_size: T.int32,
    old_scale: T.float32,
    use_ping_pong: T.bool,
):
    if key_index == 0:
        for feature_offset in T.serial(features_per_lane):
            feature = feature_index * features_per_lane + feature_offset
            if use_ping_pong and key_tile_size.bit_length() % 2 == 0:
                reduced_value = workgroup_tile_alt[0, feature]
            else:
                reduced_value = workgroup_tile[0, feature]
            accumulator[feature_offset] = (
                accumulator[feature_offset] * old_scale + reduced_value
            )


@T.macro
def process_tile(
    query: T.Buffer,
    key: T.Buffer,
    value: T.Buffer,
    valid: T.Buffer,
    split: T.Buffer,
    workgroup_tile: T.Buffer,
    workgroup_tile_alt: T.Buffer,
    accumulator: T.Buffer,
    denominator: T.Buffer,
    dot_partial: T.Buffer,
    running_max: T.Buffer,
    running_valid: T.Buffer,
    tile_denominator: T.Buffer,
    tile_max: T.Buffer,
    tile_valid: T.Buffer,
    feature_index: T.int32,
    feature_lanes: T.int32,
    features_per_lane: T.int32,
    first_key_tile: T.int32,
    head: T.int32,
    key_index: T.int32,
    key_tile_offset: T.int32,
    key_tile_size: T.int32,
    position: T.int32,
    sequence: T.int32,
    use_ping_pong: T.bool,
):
    key_tile = first_key_tile + key_tile_offset
    key_position = key_tile * key_tile_size + key_index
    key_allowed = (
        key_position < sequence
        and (position >= split[0] or key_position < split[0])
        and valid[0, key_position] != 0
    )

    # Split the q dot k reduction over the 16
    # x lanes, then finish its short reduction
    # in one lane for each key row.
    dot_partial[0] = T.float32(0.0)
    compute_dot(
        query,
        key,
        dot_partial,
        key_allowed,
        key_position,
        position,
        head,
        feature_index,
        feature_lanes,
        features_per_lane,
    )
    if use_ping_pong:
        workgroup_tile[key_index, feature_index] = dot_partial[0]
    else:
        workgroup_tile[
            0,
            key_index * feature_lanes + feature_index,
        ] = dot_partial[0]
    T.tvm_storage_sync("shared")

    if use_ping_pong:
        score_reduction_yes(
            workgroup_tile, workgroup_tile_alt, key_index, feature_index, key_allowed
        )
    else:
        score_reduction_no(
            workgroup_tile, key_index, feature_index, feature_lanes, key_allowed
        )
    T.tvm_storage_sync("shared")

    tile_max[0] = T.float32(NEGATIVE_SCORE_SENTINEL)
    tile_denominator[0] = T.float32(0.0)
    tile_valid[0] = 0
    update_tile_max(
        workgroup_tile,
        workgroup_tile_alt,
        tile_max,
        tile_valid,
        key_tile_size,
        feature_lanes,
        use_ping_pong,
    )

    next_max = T.if_then_else(
        tile_valid[0] != 0,
        T.if_then_else(
            running_valid[0] != 0,
            T.max(running_max[0], tile_max[0]),
            tile_max[0],
        ),
        running_max[0],
    )
    old_scale = T.if_then_else(
        tile_valid[0] == 0,
        T.float32(1.0),
        T.if_then_else(
            running_valid[0] != 0,
            T.exp(running_max[0] - next_max),
            T.float32(0.0),
        ),
    )
    update_tile_denominator(
        workgroup_tile,
        workgroup_tile_alt,
        tile_denominator,
        next_max,
        key_tile_size,
        feature_lanes,
        use_ping_pong,
    )
    if use_ping_pong:
        current_score = workgroup_tile_alt[key_index, 0]
    else:
        current_score = workgroup_tile[1, key_index * feature_lanes]
    key_weight = T.if_then_else(
        T.And(
            tile_valid[0] != 0,
            key_allowed,
        ),
        T.exp(current_score - next_max),
        T.float32(0.0),
    )

    # All score reads finish before the shared
    # tile is reused for V partials.
    T.tvm_storage_sync("shared")
    for feature_offset in T.serial(features_per_lane):
        feature = feature_index * features_per_lane + feature_offset
        weighted_value = T.if_then_else(
            key_allowed,
            key_weight
            * T.cast(
                value[
                    0,
                    key_position,
                    head,
                    feature,
                ],
                "float32",
            ),
            T.float32(0.0),
        )
        workgroup_tile[key_index, feature] = weighted_value
    T.tvm_storage_sync("shared")

    if use_ping_pong:
        value_reduction_yes(
            workgroup_tile,
            workgroup_tile_alt,
            key_index,
            feature_index,
            features_per_lane,
            key_tile_size,
        )
    else:
        value_reduction_no(
            workgroup_tile, key_index, feature_index, features_per_lane, key_tile_size
        )

    accumulate_value(
        accumulator,
        workgroup_tile,
        workgroup_tile_alt,
        feature_index,
        features_per_lane,
        key_index,
        key_tile_size,
        old_scale,
        use_ping_pong,
    )
    running_max[0] = next_max
    running_valid[0] = T.if_then_else(
        tile_valid[0] != 0,
        T.int32(1),
        running_valid[0],
    )
    denominator[0] = denominator[0] * old_scale + tile_denominator[0]
    T.tvm_storage_sync("shared")


@T.macro
def attention_body(
    query: T.Buffer,
    key: T.Buffer,
    value: T.Buffer,
    valid: T.Buffer,
    split: T.Buffer,
    output: T.Buffer,
    sequence: T.int32,
    head_chunk: T.int32,
    query_position: T.int32,
    key_lane: T.int32,
    feature_lane: T.int32,
    query_tile_size: T.int32,
    key_tile_size: T.int32,
    feature_lanes: T.int32,
    features_per_lane: T.int32,
    head_dim: T.int32,
    use_ping_pong: T.bool,
):
    head = T.axis.spatial(QUERY_HEADS, head_chunk % QUERY_HEADS)
    position = T.axis.spatial(
        sequence,
        T.min(
            query_position + (head_chunk // QUERY_HEADS) * query_tile_size, sequence - 1
        ),
    )
    active_query = T.axis.spatial(
        2,
        T.if_then_else(
            query_position + (head_chunk // QUERY_HEADS) * query_tile_size < sequence,
            1,
            0,
        ),
    )
    key_index = T.axis.spatial(key_tile_size, key_lane)
    feature_index = T.axis.spatial(feature_lanes, feature_lane)
    workgroup_tile = T.sblock_alloc_buffer(
        (key_tile_size, head_dim),
        "float32",
        scope="shared",
    )
    if use_ping_pong:
        workgroup_tile_alt = T.sblock_alloc_buffer(
            (key_tile_size, head_dim),
            "float32",
            scope="shared",
        )
    if use_ping_pong:
        T.reads(
            query[0, position, head, 0:head_dim],
            key[0, 0:sequence, 0:KEY_VALUE_HEADS, 0:head_dim],
            value[0, 0:sequence, 0:KEY_VALUE_HEADS, 0:head_dim],
            valid[0, 0:sequence],
            split[0],
            workgroup_tile[0:key_tile_size, 0:head_dim],
            workgroup_tile_alt[0:key_tile_size, 0:head_dim],
        )
    else:
        T.reads(
            query[0, position, head, 0:head_dim],
            key[0, 0:sequence, 0:KEY_VALUE_HEADS, 0:head_dim],
            value[0, 0:sequence, 0:KEY_VALUE_HEADS, 0:head_dim],
            valid[0, 0:sequence],
            split[0],
            workgroup_tile[0:key_tile_size, 0:head_dim],
        )
    if use_ping_pong:
        T.writes(
            output[0, position, head, 0:head_dim],
            workgroup_tile[0:key_tile_size, 0:head_dim],
            workgroup_tile_alt[0:key_tile_size, 0:head_dim],
        )
    else:
        T.writes(
            output[0, position, head, 0:head_dim],
            workgroup_tile[0:key_tile_size, 0:head_dim],
        )
    # The first row stores 16x16 QK partials; the second
    # row holds tile scores; the full buffer is reused for
    # weighted values and their key-axis reduction.
    accumulator = T.sblock_alloc_buffer(
        (features_per_lane,),
        "float32",
        scope="local",
    )
    running_max = T.sblock_alloc_buffer((1,), "float32", scope="local")
    denominator = T.sblock_alloc_buffer((1,), "float32", scope="local")
    running_valid = T.sblock_alloc_buffer((1,), "int32", scope="local")
    dot_partial = T.sblock_alloc_buffer((1,), "float32", scope="local")
    tile_max = T.sblock_alloc_buffer((1,), "float32", scope="local")
    tile_denominator = T.sblock_alloc_buffer((1,), "float32", scope="local")
    tile_valid = T.sblock_alloc_buffer((1,), "int32", scope="local")
    for feature_offset in T.serial(features_per_lane):
        accumulator[feature_offset] = T.float32(0.0)
    running_max[0] = T.float32(NEGATIVE_SCORE_SENTINEL)
    denominator[0] = T.float32(0.0)
    running_valid[0] = 0

    first_key_tile = T.int32(0)
    last_key_tile = T.ceildiv(
        T.if_then_else(position < split[0], split[0], sequence), key_tile_size
    )
    for key_tile_offset in T.serial(last_key_tile - first_key_tile):
        if use_ping_pong:
            process_tile(
                query,
                key,
                value,
                valid,
                split,
                workgroup_tile,
                workgroup_tile_alt,
                accumulator,
                denominator,
                dot_partial,
                running_max,
                running_valid,
                tile_denominator,
                tile_max,
                tile_valid,
                feature_index,
                feature_lanes,
                features_per_lane,
                first_key_tile,
                head,
                key_index,
                key_tile_offset,
                key_tile_size,
                position,
                sequence,
                use_ping_pong,
            )
        else:
            process_tile(
                query,
                key,
                value,
                valid,
                split,
                workgroup_tile,
                workgroup_tile,
                accumulator,
                denominator,
                dot_partial,
                running_max,
                running_valid,
                tile_denominator,
                tile_max,
                tile_valid,
                feature_index,
                feature_lanes,
                features_per_lane,
                first_key_tile,
                head,
                key_index,
                key_tile_offset,
                key_tile_size,
                position,
                sequence,
                use_ping_pong,
            )

    if key_index == 0 and active_query != 0:
        for feature_offset in T.serial(features_per_lane):
            feature = feature_index * features_per_lane + feature_offset
            output[
                0,
                position,
                head,
                feature,
            ] = T.cast(
                T.if_then_else(
                    denominator[0] > T.float32(0.0),
                    accumulator[feature_offset] / denominator[0],
                    T.float32(0.0),
                ),
                "float32",
            )


def attention_config(
    head_dim,
    sliding_window,
    use_ping_pong,
    max_shared_storage_bytes,
    max_workgroup_invocations,
):
    if head_dim != 64:
        raise ValueError(
            "the parallel GPU kernel supports Gemma 4 head dimensions 256 and 512"
        )
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError("sliding_window must be positive or None")

    feature_lanes = 16
    candidate_key_tiles = (16, 8)
    buffers_per_tile = 2 if use_ping_pong else 1
    key_tile_size = next(
        (
            tile
            for tile in candidate_key_tiles
            if tile * head_dim * 4 * buffers_per_tile <= max_shared_storage_bytes
            and tile * feature_lanes <= max_workgroup_invocations
        ),
        0,
    )
    if key_tile_size == 0:
        raise ValueError(
            "WebGPU limits cannot hold the minimum 8-key E2B attention tile."
        )
    features_per_lane = head_dim // feature_lanes
    window = sliding_window or 0
    return feature_lanes, key_tile_size, features_per_lane, window


def make_parallel_gpu_prim_func(
    head_dim: int,
    sliding_window: int | None,
    *,
    max_shared_storage_bytes: int = M4_WEBGPU_SHARED_BYTES,
    max_workgroup_invocations: int = WEBGPU_MAX_WORKGROUP_INVOCATIONS,
    use_ping_pong: bool = False,
    query_tile_size: int = 32768,
):
    """Return a tiled attention kernel sized to the requested WebGPU limits.

    Sixteen lanes split query/key features and eight or sixteen lanes split a
    tile of keys. The one key-tile-by-head-dimension shared tile doubles as the
    QK partial buffer and the weighted-value reduction buffer, keeping D=512
    at the requested workgroup-storage limit.
    """
    feature_lanes, key_tile_size, features_per_lane, window = attention_config(
        head_dim,
        sliding_window,
        use_ping_pong,
        max_shared_storage_bytes,
        max_workgroup_invocations,
    )

    @I.ir_module(s_tir=True)
    class ParallelKernels:
        @T.prim_func(s_tir=True)
        def online_parallel(
            query_handle: T.handle,
            key_handle: T.handle,
            value_handle: T.handle,
            valid_handle: T.handle,
            split_handle: T.handle,
            output_handle: T.handle,
        ):
            sequence = T.int32()
            query = T.match_buffer(
                query_handle,
                (BATCH_SIZE, sequence, QUERY_HEADS, head_dim),
                "float32",
            )
            key = T.match_buffer(
                key_handle,
                (BATCH_SIZE, sequence, KEY_VALUE_HEADS, head_dim),
                "float32",
            )
            value = T.match_buffer(
                value_handle,
                (BATCH_SIZE, sequence, KEY_VALUE_HEADS, head_dim),
                "float32",
            )
            valid = T.match_buffer(valid_handle, (BATCH_SIZE, sequence), "int32")
            split = T.match_buffer(split_handle, (1,), "int32")
            output = T.match_buffer(
                output_handle,
                (BATCH_SIZE, sequence, QUERY_HEADS, head_dim),
                "float32",
            )

            with T.sblock("root"):
                for head_chunk in T.thread_binding(
                    QUERY_HEADS * T.ceildiv(sequence, query_tile_size),
                    thread="blockIdx.y",
                ):
                    for query_position in T.thread_binding(
                        T.min(sequence, query_tile_size), thread="blockIdx.x"
                    ):
                        for key_lane in T.thread_binding(
                            key_tile_size, thread="threadIdx.y"
                        ):
                            for feature_lane in T.thread_binding(
                                feature_lanes, thread="threadIdx.x"
                            ):
                                with T.sblock("attention"):
                                    attention_body(
                                        query,
                                        key,
                                        value,
                                        valid,
                                        split,
                                        output,
                                        sequence,
                                        head_chunk,
                                        query_position,
                                        key_lane,
                                        feature_lane,
                                        query_tile_size,
                                        key_tile_size,
                                        feature_lanes,
                                        features_per_lane,
                                        head_dim,
                                        use_ping_pong,
                                    )

    return ParallelKernels["online_parallel"]
