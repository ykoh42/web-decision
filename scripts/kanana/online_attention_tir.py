"""Kanana's tiled, bounded-memory causal attention lowering for TVM WebGPU."""


def _make_score_macro(T, tile_x, tile_y, tile_z, log2e):
    @T.macro
    def compute_scores(
        q_tile: T.Buffer,
        k_tile: T.Buffer,
        score_local: T.Buffer,
        score_shared: T.Buffer,
        attention_bias: T.Buffer,
        key_start: T.int32,
        causal_scale: T.float32,
        current_key_length: T.int32,
    ):
        with T.sblock():
            for row, column, dimension in T.grid(tile_x, tile_z, tile_y):
                with T.sblock("S_gemm"):
                    i, j, k = T.axis.remap("SSR", [row, column, dimension])
                    with T.init():
                        score_local[i, j] = T.float32(0.0)
                    score_local[i, j] += T.cast(q_tile[i, k], "float32") * T.cast(
                        k_tile[j, k], "float32"
                    ) * causal_scale * log2e + T.if_then_else(
                        T.And(
                            k == tile_y - 1,
                            key_start + j < current_key_length,
                        ),
                        attention_bias[0, 0, 0, key_start + j] * log2e,
                        T.float32(0.0),
                    )
        T.tvm_storage_sync("shared")
        for row, column in T.grid(tile_x, tile_z):
            with T.sblock("S_store"):
                i, j = T.axis.remap("SS", [row, column])
                score_shared[i, j] = score_local[i, j]
        T.tvm_storage_sync("shared")

    return compute_scores


def _make_query_load_macro(T, tile_x, head_dim):
    @T.macro
    def load_query(
        query: T.Buffer,
        q_tile: T.Buffer,
        batch_index: T.int32,
        query_start: T.int32,
        head: T.int32,
        query_length: T.int32,
    ):
        for row, dimension in T.grid(tile_x, head_dim):
            with T.sblock("Q_load"):
                i, j = T.axis.remap("SS", [row, dimension])
                T.reads()
                T.writes()
                current_row: T.let[T.int32] = query_start + i
                if current_row < query_length:
                    q_tile[i, j] = query[batch_index, current_row, head, j]
                else:
                    q_tile[i, j] = T.float32(0.0)

    return load_query


def _make_key_load_macro(T, tile_z, head_dim):
    @T.macro
    def load_key(
        key: T.Buffer,
        k_tile: T.Buffer,
        batch_index: T.int32,
        key_start: T.int32,
        head: T.int32,
        key_length: T.int32,
    ):
        for row, dimension in T.grid(tile_z, head_dim):
            with T.sblock("K_load"):
                i, j = T.axis.remap("SS", [row, dimension])
                T.reads()
                T.writes()
                current_key: T.let[T.int32] = key_start + i
                if current_key < key_length:
                    k_tile[i, j] = key[batch_index, current_key, head, j]
                else:
                    k_tile[i, j] = T.float32(0.0)

    return load_key


def _make_value_load_macro(T, tile_z, head_dim):
    @T.macro
    def load_value(
        value: T.Buffer,
        v_tile: T.Buffer,
        batch_index: T.int32,
        key_start: T.int32,
        head: T.int32,
        key_length: T.int32,
    ):
        for row, dimension in T.grid(tile_z, head_dim):
            with T.sblock("V_load"):
                i, j = T.axis.remap("SS", [row, dimension])
                T.reads()
                T.writes()
                current_key: T.let[T.int32] = key_start + i
                if current_key < key_length:
                    v_tile[i, j] = value[batch_index, current_key, head, j]
                else:
                    v_tile[i, j] = T.float32(0.0)

    return load_value


def _make_output_store_macro(T, tile_x, head_dim):
    @T.macro
    def store_output(
        output: T.Buffer,
        output_local: T.Buffer,
        denominator_shared: T.Buffer,
        batch_index: T.int32,
        query_start: T.int32,
        head: T.int32,
        query_length: T.int32,
    ):
        for row, dimension in T.grid(tile_x, head_dim):
            with T.sblock("O_store"):
                i, j = T.axis.remap("SS", [row, dimension])
                current_row: T.let[T.int32] = query_start + i
                if current_row < query_length:
                    output[batch_index, current_row, head, j] = (
                        output_local[i, j] / denominator_shared[i]
                    )

    return store_output


def _make_lse_store_macro(T, tile_x):
    @T.macro
    def store_lse(
        lse: T.Buffer,
        max_shared: T.Buffer,
        denominator_shared: T.Buffer,
        batch_index: T.int32,
        query_start: T.int32,
        head: T.int32,
        query_length: T.int32,
    ):
        for row in T.grid(tile_x):
            with T.sblock("lse_store"):
                i = T.axis.remap("S", [row])
                current_row: T.let[T.int32] = query_start + i
                if current_row < query_length:
                    lse[batch_index, current_row, head] = max_shared[i] + T.log2(
                        denominator_shared[i]
                    )

    return store_lse


def _make_attention_module(
    ir_script,
    T,
    head_count,
    head_dim,
    group_size,
    bdx,
    num_warps,
    tile_x,
    tile_z,
    alloc_mha_buffers,
    alloc_softmax_buffers,
    init_states,
    softmax_update_causal,
    compute_o_gemm,
    compute_scores,
    load_query,
    load_key,
    load_value,
    store_output,
    store_lse,
    scale,
):
    @ir_script.ir_module(s_tir=True)
    class AttentionModule:
        @T.prim_func(s_tir=True)
        def online_attention(
            query_handle: T.handle,
            key_handle: T.handle,
            value_handle: T.handle,
            bias_handle: T.handle,
            output_handle: T.handle,
            lse_handle: T.handle,
        ):
            T.func_attr({"tirx.noalias": True})
            batch_size = T.int32()
            query_length = T.int32()
            key_length = T.int32()
            query = T.match_buffer(
                query_handle,
                (batch_size, query_length, head_count, head_dim),
                "float32",
            )
            key = T.match_buffer(
                key_handle, (batch_size, key_length, head_count, head_dim), "float32"
            )
            value = T.match_buffer(
                value_handle, (batch_size, key_length, head_count, head_dim), "float32"
            )
            bias = T.match_buffer(bias_handle, (1, 1, 1, key_length), "float32")
            output = T.match_buffer(
                output_handle,
                (batch_size, query_length, head_count, head_dim),
                "float32",
            )
            lse = T.match_buffer(
                lse_handle, (batch_size, query_length, head_count), "float32"
            )

            batch_tiles: T.let[T.int32] = T.ceildiv(query_length * group_size, tile_x)
            for block_x in T.thread_binding(
                T.cast(batch_size, "int32") * batch_tiles, thread="blockIdx.x"
            ):
                for block_y in T.thread_binding(head_count, thread="blockIdx.y"):
                    for thread_y in T.thread_binding(num_warps, thread="threadIdx.y"):
                        for thread_x in T.thread_binding(bdx, thread="threadIdx.x"):
                            with T.sblock("attn"):
                                block_id, head, ty, tx = T.axis.remap(
                                    "SSSS", [block_x, block_y, thread_y, thread_x]
                                )
                                T.reads()
                                T.writes()
                                q_tile, k_tile, v_tile, output_local = (
                                    alloc_mha_buffers(
                                        tile_x, tile_z, head_dim, head_dim, "float32"
                                    )
                                )
                                (
                                    score_shared,
                                    score_local,
                                    max_shared,
                                    previous_max_shared,
                                    denominator_shared,
                                    max_new,
                                    max_previous,
                                    denominator_new,
                                ) = alloc_softmax_buffers(
                                    tile_x, tile_z, bdx, num_warps
                                )

                                batch_index: T.let[T.int32] = block_id // batch_tiles
                                tile_index: T.let[T.int32] = block_id % batch_tiles
                                query_start: T.let[T.int32] = tile_index * tile_x
                                T.tvm_storage_sync("shared")
                                init_states(
                                    max_shared, denominator_shared, output_local, ty, tx
                                )
                                load_query(
                                    query,
                                    q_tile,
                                    batch_index,
                                    query_start,
                                    head,
                                    query_length,
                                )
                                T.tvm_storage_sync("shared")

                                key_limit: T.let[T.int32] = T.min(
                                    query_start + tile_x, key_length
                                )
                                for tile_id in T.serial(T.ceildiv(key_limit, tile_z)):
                                    key_start: T.let[T.int32] = tile_id * tile_z
                                    load_key(
                                        key,
                                        k_tile,
                                        batch_index,
                                        key_start,
                                        head,
                                        key_length,
                                    )
                                    T.tvm_storage_sync("shared")
                                    load_value(
                                        value,
                                        v_tile,
                                        batch_index,
                                        key_start,
                                        head,
                                        key_length,
                                    )
                                    T.tvm_storage_sync("shared")
                                    compute_scores(
                                        q_tile,
                                        k_tile,
                                        score_local,
                                        score_shared,
                                        bias,
                                        key_start,
                                        T.float32(scale),
                                        key_length,
                                    )
                                    softmax_update_causal(
                                        score_shared,
                                        max_shared,
                                        denominator_shared,
                                        previous_max_shared,
                                        max_new,
                                        max_previous,
                                        denominator_new,
                                        ty,
                                        tx,
                                        query_start,
                                        key_start,
                                        1,
                                        key_length,
                                        query_length,
                                        0,
                                    )
                                    compute_o_gemm(
                                        score_shared,
                                        v_tile,
                                        output_local,
                                        previous_max_shared,
                                        max_shared,
                                    )

                                store_output(
                                    output,
                                    output_local,
                                    denominator_shared,
                                    batch_index,
                                    query_start,
                                    head,
                                    query_length,
                                )
                                store_lse(
                                    lse,
                                    max_shared,
                                    denominator_shared,
                                    batch_index,
                                    query_start,
                                    head,
                                    query_length,
                                )

    return AttentionModule


def _static_dimension(dimension):
    value = getattr(dimension, "value", None)
    return int(value) if value is not None else None


def _same_dimension(tvm, left, right):
    try:
        tvm.ir.assert_structural_equal(left, right)
    except ValueError:
        return False
    return True


def _validate_qkv_shapes(tvm, call, head_count, head_dim):
    shapes = []
    for tensor in call.args[:3]:
        info = tensor.ty
        if str(info.dtype) != "float32" or len(info.shape) != 4:
            raise RuntimeError(
                "Kanana online attention expects four-dimensional float32 Q, K, and V."
            )
        shapes.append(info.shape)

    expected_shape = (1, None, head_count, head_dim)
    if any(
        tuple(_static_dimension(dimension) for dimension in shape) != expected_shape
        for shape in shapes
    ):
        raise RuntimeError(
            "Kanana online attention received an unsupported Q/K/V shape."
        )
    if not _same_dimension(tvm, shapes[0][1], shapes[1][1]) or not _same_dimension(
        tvm, shapes[0][1], shapes[2][1]
    ):
        raise RuntimeError(
            "Kanana online attention requires Q, K, and V sequence lengths to match."
        )
    return shapes, expected_shape


def _validate_attention_mask(tvm, call, query_shape):
    if call.args[3] is None:
        raise RuntimeError(
            "Kanana online attention requires the sequence-length padding mask."
        )
    bias_shape = call.args[3].ty.shape
    if len(bias_shape) != 4:
        raise RuntimeError(
            "Kanana online attention requires a four-dimensional padding mask."
        )
    bias_prefix = tuple(_static_dimension(bias_shape[index]) for index in range(3))
    if (
        str(call.args[3].ty.dtype) != "float32"
        or bias_prefix != (1, 1, 1)
        or not _same_dimension(tvm, query_shape[1], bias_shape[3])
    ):
        raise RuntimeError(
            "Kanana online attention requires a broadcastable [1, 1, 1, sequence] mask."
        )


def _validate_attention_call(tvm, call, head_count, head_dim, scale):
    if call.attrs.causal_mask != "TopLeft" or call.attrs.window_size is not None:
        raise RuntimeError(
            "Kanana online attention requires full-sequence TopLeft causal attention."
        )
    shapes, expected_shape = _validate_qkv_shapes(tvm, call, head_count, head_dim)
    _validate_attention_mask(tvm, call, shapes[0])
    actual_scale = getattr(call.attrs.scale, "value", call.attrs.scale)
    if actual_scale is None or abs(float(actual_scale) - scale) > 2e-8:
        raise RuntimeError(
            "Kanana online attention requires the pinned 1/sqrt(128) scale."
        )
    _validate_attention_output(tvm, call, shapes[0], expected_shape)
    return shapes


def _validate_attention_output(tvm, call, query_shape, expected_shape):
    output_shape = call.ty.shape
    if (
        str(call.ty.dtype) != "float32"
        or len(output_shape) != 4
        or tuple(_static_dimension(dimension) for dimension in output_shape)
        != expected_shape
        or not _same_dimension(tvm, query_shape[1], output_shape[1])
    ):
        raise RuntimeError(
            "Kanana online attention received an unsupported output shape or dtype."
        )


def _make_attention_function(
    tvm,
    schedule_kernel,
    module,
    load_vec,
    bdx,
    num_warps,
    tile_x,
    tile_y,
    tile_z,
    function_name,
):
    schedule = tvm.s_tir.Schedule(module["online_attention"])
    schedule = schedule_kernel(
        schedule, load_vec, bdx, num_warps, tile_x, tile_y, tile_z, False, False
    )
    return schedule.mod["main"].with_attr("tirx.is_scheduled", True), function_name


def _emit_attention_output(
    relax, block_builder, attention_function, call, shapes, head_count
):
    output_info = call.ty
    lse_info = relax.TensorType((1, shapes[0][1], head_count), "float32")
    outputs = block_builder.emit(
        relax.op.call_tir(
            attention_function,
            relax.Tuple(call.args),
            out_ty=[output_info, lse_info],
        )
    )
    return block_builder.emit(relax.TupleGetItem(outputs, 0))


def _make_attention_legalizer(
    tvm,
    relax,
    schedule_kernel,
    module,
    lowering,
    head_count,
    head_dim,
    scale,
    load_vec,
    bdx,
    num_warps,
    tile_x,
    tile_y,
    tile_z,
    function_name,
):
    attention_function = [None]

    def legalize_attention(block_builder, call):
        lowering["loweredAttentionOps"] += 1
        shapes = _validate_attention_call(tvm, call, head_count, head_dim, scale)
        if attention_function[0] is None:
            function, name = _make_attention_function(
                tvm,
                schedule_kernel,
                module,
                load_vec,
                bdx,
                num_warps,
                tile_x,
                tile_y,
                tile_z,
                function_name,
            )
            attention_function[0] = block_builder.add_func(function, name)
        return _emit_attention_output(
            relax, block_builder, attention_function[0], call, shapes, head_count
        )

    return legalize_attention


def install_online_attention_tir(tvm, module, target):
    from tvm import relax
    from tvm.relax.frontend.nn.llm._kernel_common import (
        _alloc_mha_qkvo_buffers,
        _alloc_softmax_state_buffers,
        _get_prefill_kernel_config,
        _make_prefill_macros,
        _schedule_prefill_kernel,
    )
    from tvm.script import ir as ir_script
    from tvm.script import tirx as T

    head_count = 32
    head_dim = 128
    scale = 0.08838834764831845
    log2e = 1.4426950408889634
    _, load_vec, group_size, bdx, num_warps, tile_x, tile_y, tile_z = (
        _get_prefill_kernel_config(head_count, head_count, head_dim, "float32", target)
    )
    init_states, _, softmax_update_causal, compute_o_gemm, *_ = _make_prefill_macros(
        tile_x, tile_y, tile_z, tile_y, bdx, num_warps, group_size
    )
    macros = (
        _make_score_macro(T, tile_x, tile_y, tile_z, log2e),
        _make_query_load_macro(T, tile_x, head_dim),
        _make_key_load_macro(T, tile_z, head_dim),
        _make_value_load_macro(T, tile_z, head_dim),
        _make_output_store_macro(T, tile_x, head_dim),
        _make_lse_store_macro(T, tile_x),
    )
    attention_module = _make_attention_module(
        ir_script,
        T,
        head_count,
        head_dim,
        group_size,
        bdx,
        num_warps,
        tile_x,
        tile_z,
        _alloc_mha_qkvo_buffers,
        _alloc_softmax_state_buffers,
        init_states,
        softmax_update_causal,
        compute_o_gemm,
        *macros,
        scale,
    )
    function_name = "kanana_online_attention"
    lowering = {
        "function": function_name,
        "operator": "relax.nn.attention_bias",
        "algorithm": "tiled_online_softmax",
        "sequenceDimension": "dynamic",
        "queryTile": tile_x,
        "keyTile": tile_z,
        "threadsPerBlock": bdx * num_warps,
        "causalKeyLimit": "query_tile_end",
        "intermediateScoreMatrix": False,
        "loweredAttentionOps": 0,
    }
    legalizer = _make_attention_legalizer(
        tvm,
        relax,
        _schedule_prefill_kernel,
        attention_module,
        lowering,
        head_count,
        head_dim,
        scale,
        load_vec,
        bdx,
        num_warps,
        tile_x,
        tile_y,
        tile_z,
        function_name,
    )
    tvm.ir.register_op_attr("relax.nn.attention_bias", "FLegalize", legalizer, level=11)
    return lowering
