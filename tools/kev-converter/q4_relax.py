"""Replace pinned Kev backbone weights with direct TVM Q4 parameters."""

import re

import numpy as np
import tvm
from q4 import gather_te, matmul_te, quantize_matmul_rows, quantize_rows
from tvm import relax, s_tir, te

BLOCK_SIZE = 32
Q4_GPU_OUTPUT_TILE = 8
Q4_GPU_EXPERIMENTAL_ROW4_SCHEDULE = "row4_out8_k8"
EMBEDDING_WEIGHT = "backbone.embed_tokens.weight"
EMBEDDING_VARIABLE = f"p_{EMBEDDING_WEIGHT.replace('.', '_')}"
LINEAR_WEIGHT = re.compile(
    r"backbone\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)\.weight"
)


def is_q4_weight(name: str) -> bool:
    return name == EMBEDDING_WEIGHT or LINEAR_WEIGHT.fullmatch(name) is not None


def quantize_weight(name: str, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize an embedding row or a transposed Linear weight by output row."""
    matrix = np.asarray(value, dtype=np.float16)
    if name == EMBEDDING_WEIGHT:
        rows = matrix
        return quantize_rows(rows, block_size=BLOCK_SIZE)
    elif LINEAR_WEIGHT.fullmatch(name):
        if matrix.ndim != 2:
            raise ValueError(f"Linear parameter {name} must be a matrix.")
        rows = matrix.T
        return quantize_matmul_rows(rows, block_size=BLOCK_SIZE)
    else:
        raise ValueError(f"Parameter is outside the pinned Kev Q4 set: {name}")


def _dimensions(tensor_info):
    try:
        return tuple(tensor_info.shape)
    except (AttributeError, TypeError) as error:
        raise ValueError(
            "Q4 graph integration requires a known tensor rank."
        ) from error


def _static_dimensions(tensor_info) -> tuple[int, ...]:
    try:
        return tuple(int(dimension) for dimension in tensor_info.shape)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("Q4 weight dimensions must be static.") from error


def _type_from_call(call):
    output_type = call.ty
    if not isinstance(output_type, relax.TensorType):
        raise ValueError(
            f"Q4 replacement requires a tensor output type, found {output_type}."
        )
    return output_type


def _resolve_parameter(expression, target_names: set[str], lookup_binding):
    seen = set()
    while True:
        if isinstance(expression, relax.Var):
            name = str(expression.name)
            if name in target_names:
                return name
            if name in seen:
                return None
            seen.add(name)
            expression = lookup_binding(expression)
            if expression is None:
                return None
            continue
        if isinstance(expression, relax.Call) and expression.op.name == "relax.astype":
            expression = expression.args[0]
            continue
        return None


def _largest_power_of_two(value: int) -> int:
    """Return the largest power of two no greater than a positive limit."""
    if value < 1:
        raise ValueError("GPU workgroup dimensions must be positive.")
    return 1 << (value.bit_length() - 1)


def _gpu_matmul_tile(target) -> tuple[int, int] | None:
    """Choose Q4 output and reduction lanes from an explicit GPU target."""
    if target is None:
        return None
    target = (
        target if isinstance(target, tvm.target.Target) else tvm.target.Target(target)
    )
    if "gpu" not in target.keys:
        return None

    attrs = target.attrs
    max_threads = int(attrs.get("max_num_threads", 256))
    if max_threads < 1:
        raise ValueError("GPU target max_num_threads must be positive.")

    output_lanes = min(Q4_GPU_OUTPUT_TILE, max_threads)
    reduction_limit = max_threads // output_lanes
    if target.kind.name == "webgpu":
        # WGSL guarantees 256 invocations along an axis. A larger target-provided
        # workgroup limit can grow the total tile while keeping x within this axis
        # limit; a target may override the axis value when its adapter supports more.
        reduction_limit = min(
            reduction_limit, int(attrs.get("max_workgroup_size_x", 256))
        )
    reduction_lanes = _largest_power_of_two(reduction_limit)
    return output_lanes, reduction_lanes


def _q4_matmul_schedule_tile(target, experimental_schedule):
    if experimental_schedule is None:
        tile = _gpu_matmul_tile(target)
        return (1, *tile) if tile is not None else None
    if experimental_schedule != Q4_GPU_EXPERIMENTAL_ROW4_SCHEDULE:
        raise ValueError(f"Unknown experimental Q4 schedule: {experimental_schedule}")
    if target.kind.name not in ("llvm", "webgpu"):
        raise ValueError("The experimental Q4 row tile supports LLVM and WebGPU.")
    return 4, 8, 8


def _schedule_q4_matmul(prim_func, target, experimental_schedule=None):
    """Schedule GPU Q4 matmuls; keep experimental layouts explicit opt-ins."""
    if target is None:
        if experimental_schedule is None:
            return prim_func
        raise ValueError(
            "An explicit target is required for an experimental Q4 schedule."
        )
    target = (
        target if isinstance(target, tvm.target.Target) else tvm.target.Target(target)
    )
    is_gpu = "gpu" in target.keys
    tile = _q4_matmul_schedule_tile(target, experimental_schedule)
    if tile is None:
        return prim_func
    row_tile, output_lanes, reduction_lanes = tile

    schedule = s_tir.Schedule(prim_func)
    block = schedule.get_sblock("kev_q4_matmul")
    row, column, reduction = schedule.get_loops(block)
    if row_tile > 1:
        row_outer, row_inner = schedule.split(row, factors=[None, row_tile])
    else:
        row_outer, row_inner = row, None
    column_outer, column_inner = schedule.split(column, factors=[None, output_lanes])
    reduction_outer, reduction_inner = schedule.split(
        reduction, factors=[None, reduction_lanes]
    )
    if row_inner is None:
        schedule.reorder(
            row_outer,
            column_outer,
            column_inner,
            reduction_inner,
            reduction_outer,
        )
    else:
        schedule.reorder(
            row_outer,
            column_outer,
            row_inner,
            column_inner,
            reduction_inner,
            reduction_outer,
        )
    if is_gpu:
        schedule.bind(row_outer, "blockIdx.y")
        schedule.bind(column_outer, "blockIdx.x")
        schedule.bind(column_inner, "threadIdx.y")
        schedule.bind(reduction_inner, "threadIdx.x")
        if row_inner is not None:
            schedule.bind(row_inner, "threadIdx.z")
    return schedule.mod["main"]


@relax.expr_functor.mutator
class Q4GraphMutator(relax.PyExprMutator):
    def __init__(
        self, module, builder, q4_parameters, target=None, experimental_schedule=None
    ):
        super().__init__(module)
        self.builder = builder
        self.target = target
        self.experimental_schedule = experimental_schedule
        self.q4_parameters = {str(name): value for name, value in q4_parameters.items()}
        self.target_names = set(self.q4_parameters)
        self.matmul_kernels = {}
        self.gather_kernels = {}
        self.replaced_matmuls = 0
        self.replaced_gathers = 0
        self.replaced_weight_counts = {}

    def _record_replacement(self, target_name):
        self.replaced_weight_counts[target_name] = (
            self.replaced_weight_counts.get(target_name, 0) + 1
        )

    def _matmul_kernel(self, input_info, packed_var):
        rows, width = _dimensions(input_info)
        output_columns, packed_width = _dimensions(packed_var.ty)
        input_dtype = input_info.dtype
        key = (rows, width, output_columns, input_dtype)
        if key not in self.matmul_kernels:
            input_tensor = te.placeholder(
                (rows, width), dtype=input_dtype, name="q4_input"
            )
            packed_tensor = te.placeholder(
                (output_columns, packed_width), dtype="uint32", name="q4_packed"
            )
            scale_tensor = te.placeholder(
                (output_columns, width // BLOCK_SIZE), dtype="float16", name="q4_scales"
            )
            output = matmul_te(
                input_tensor,
                packed_tensor,
                scale_tensor,
                block_size=BLOCK_SIZE,
            )
            function_name = f"kev_q4_matmul_{len(self.matmul_kernels)}"
            prim_func = te.create_prim_func(
                [input_tensor, packed_tensor, scale_tensor, output]
            ).with_attr("global_symbol", function_name)
            prim_func = _schedule_q4_matmul(
                prim_func, self.target, self.experimental_schedule
            )
            self.matmul_kernels[key] = self.builder.add_func(prim_func, function_name)
        return self.matmul_kernels[key]

    def _gather_kernel(self, ids_info, packed_var):
        (length,) = _dimensions(ids_info)
        rows, words = _dimensions(packed_var.ty)
        width = words * 8
        ids_dtype = ids_info.dtype
        key = (length, rows, width, ids_dtype)
        if key not in self.gather_kernels:
            packed_tensor = te.placeholder(
                (rows, words), dtype="uint32", name="q4_embedding"
            )
            ids_tensor = te.placeholder((length,), dtype=ids_dtype, name="token_ids")
            scales_tensor = te.placeholder(
                (rows, width // BLOCK_SIZE), dtype="float16", name="q4_embedding_scales"
            )
            output = gather_te(
                packed_tensor, ids_tensor, scales_tensor, block_size=BLOCK_SIZE
            )
            function_name = f"kev_q4_gather_{len(self.gather_kernels)}"
            prim_func = te.create_prim_func(
                [packed_tensor, ids_tensor, scales_tensor, output]
            ).with_attr("global_symbol", function_name)
            self.gather_kernels[key] = self.builder.add_func(prim_func, function_name)
        return self.gather_kernels[key]

    def visit_call_(self, call):
        operation_name = call.op.name
        target_name = None
        if operation_name == "relax.matmul":
            target_name = _resolve_parameter(
                call.args[1], self.target_names, self.lookup_binding
            )
        elif operation_name == "relax.take":
            target_name = _resolve_parameter(
                call.args[0], self.target_names, self.lookup_binding
            )
        updated = super().visit_call_(call)
        if operation_name == "relax.matmul":
            if target_name is not None:
                packed_var, scales_var = self.q4_parameters[target_name]
                lhs = updated.args[0]
                output_type = _type_from_call(updated)
                kernel = self._matmul_kernel(lhs.ty, packed_var)
                self.replaced_matmuls += 1
                self._record_replacement(target_name)
                accumulation_type = relax.TensorType(output_type.shape, "float32")
                result = relax.call_tir(
                    kernel, [lhs, packed_var, scales_var], accumulation_type
                )
                if output_type.dtype != "float32":
                    return relax.op.astype(result, output_type.dtype)
                return result

        if operation_name == "relax.take" and target_name == EMBEDDING_VARIABLE:
            packed_var, scales_var = self.q4_parameters[target_name]
            ids = updated.args[1]
            output_type = _type_from_call(updated)
            kernel = self._gather_kernel(ids.ty, packed_var)
            self.replaced_gathers += 1
            self._record_replacement(target_name)
            return relax.call_tir(kernel, [packed_var, ids, scales_var], output_type)

        return updated


def rewrite_main_graph(
    module, new_params, q4_parameters, target=None, experimental_schedule=None
):
    module = relax.transform.Normalize()(module)
    main = module["main"]
    builder = relax.BlockBuilder(module)
    mutator = Q4GraphMutator(
        module,
        builder,
        q4_parameters,
        target=target,
        experimental_schedule=experimental_schedule,
    )
    body = mutator.visit_expr(main.body)
    new_main = relax.Function(
        new_params,
        body,
        ret_ty=main.ret_ty,
        is_pure=main.is_pure,
        attrs=main.attrs,
        span=main.span,
    )
    builder.update_func(module.get_global_var("main"), new_main)
    transformed = builder.get()
    transformed = relax.transform.DeadCodeElimination()(transformed)
    transformed = relax.transform.Normalize()(transformed)
    relax.analysis.well_formed(transformed)
    return (
        transformed,
        mutator.replaced_matmuls,
        mutator.replaced_gathers,
        mutator.replaced_weight_counts,
    )


def _validate_parameter_mapping(name, variable):
    expected_name = f"p_{name.replace('.', '_')}"
    if variable.name != expected_name:
        raise ValueError(
            f"Unexpected Relax parameter mapping for {name}: {variable.name}"
        )


def _quantized_entry(name, variable, tensor, cpu_device):
    if not is_q4_weight(name):
        return None
    source = np.asarray(tensor.numpy())
    expected_shape = _static_dimensions(variable.ty)
    if source.shape != expected_shape:
        raise ValueError(
            f"Source and Relax shapes differ for {name}: {source.shape} != {expected_shape}"
        )
    packed, scales = quantize_weight(name, source)
    packed_var = relax.Var(
        f"{variable.name}_q4_packed",
        relax.TensorType(tuple(packed.shape), "uint32"),
    )
    scales_var = relax.Var(
        f"{variable.name}_q4_scales",
        relax.TensorType(tuple(scales.shape), "float16"),
    )
    return {
        "old": variable,
        "packed_var": packed_var,
        "scales_var": scales_var,
        "packed_value": tvm.runtime.tensor(packed, device=cpu_device),
        "scales_value": tvm.runtime.tensor(scales, device=cpu_device),
    }


def _collect_quantized(parameter_names, model_parameters, parameter_values, cpu_device):
    selected = {}
    runtime_by_name = {}
    for name, variable, tensor in zip(
        parameter_names, model_parameters, parameter_values, strict=True
    ):
        _validate_parameter_mapping(name, variable)
        runtime_by_name[name] = tensor
        entry = _quantized_entry(name, variable, tensor, cpu_device)
        if entry is not None:
            selected[name] = entry
    return selected, runtime_by_name


def _validate_inventory(selected):
    if EMBEDDING_WEIGHT not in selected:
        raise ValueError("Pinned Kev embedding weight was not exported.")
    linear_names = [name for name in selected if name != EMBEDDING_WEIGHT]
    if len(linear_names) != 196:
        raise ValueError(
            f"Expected 196 Kev backbone Linear weights, found {len(linear_names)}."
        )
    return linear_names


def _build_runtime_signature(
    main, parameter_names, model_parameters, selected, runtime_by_name
):
    new_params = list(main.params[:2])
    runtime_parameters = []
    q4_parameters = {}
    for name, variable in zip(parameter_names, model_parameters, strict=True):
        item = selected.get(name)
        if item is None:
            new_params.append(variable)
            runtime_parameters.append(runtime_by_name[name])
            continue
        packed_var = item["packed_var"]
        scales_var = item["scales_var"]
        new_params.extend((packed_var, scales_var))
        runtime_parameters.extend((item["packed_value"], item["scales_value"]))
        q4_parameters[variable.name] = (packed_var, scales_var)
    return new_params, runtime_parameters, q4_parameters


def _validate_replacements(
    replaced_matmuls, replaced_gathers, replacement_counts, expected_names
):
    if replaced_matmuls != len(expected_names) - 1 or replaced_gathers != 1:
        raise ValueError(
            "Q4 graph replacement count mismatch: "
            f"matmul={replaced_matmuls}/{len(expected_names) - 1}, "
            f"gather={replaced_gathers}/1"
        )
    if set(replacement_counts) != expected_names or any(
        count != 1 for count in replacement_counts.values()
    ):
        raise ValueError("Q4 graph did not replace every selected weight exactly once.")


def integrate_q4(
    module, parameter_names: list[str], parameter_values: list, cpu_device, target=None
):
    """Quantize selected weights and rewrite Relax, optionally scheduling for a target."""
    main = module["main"]
    model_parameters = main.params[2:]
    if len(model_parameters) != len(parameter_names) or len(parameter_values) != len(
        parameter_names
    ):
        raise ValueError(
            "Relax parameter order differs from the exported source parameter order."
        )
    selected, runtime_by_name = _collect_quantized(
        parameter_names, model_parameters, parameter_values, cpu_device
    )
    linear_names = _validate_inventory(selected)
    new_params, runtime_parameters, q4_parameters = _build_runtime_signature(
        main, parameter_names, model_parameters, selected, runtime_by_name
    )

    transformed, replaced_matmuls, replaced_gathers, replacement_counts = (
        rewrite_main_graph(module, new_params, q4_parameters, target=target)
    )

    expected_names = {f"p_{name.replace('.', '_')}" for name in selected}
    _validate_replacements(
        replaced_matmuls, replaced_gathers, replacement_counts, expected_names
    )

    return (
        transformed,
        runtime_parameters,
        {
            "quantizedParameterCount": len(selected),
            "quantizedLinearCount": len(linear_names),
            "quantizedEmbeddingCount": 1,
            "replacedMatmulCount": replaced_matmuls,
            "replacedEmbeddingGatherCount": replaced_gathers,
            "replacedWeightVariableNames": sorted(replacement_counts),
            "quantizedParameterNames": list(selected),
            "format": "kev-q4-ort-matmul-offset8-signed-gather-v1",
            "bits": 4,
            "blockSize": BLOCK_SIZE,
            "scaleDtype": "float16",
            "matmulEncoding": "uint4-offset-8",
            "matmulScaleFormula": "signed-extremeDivNegative8",
            "matmulRounding": "float32-roundf-half-away-from-zero",
            "embeddingEncoding": "signed-int4-two-complement",
            "embeddingScaleFormula": "signed-extremeDivNegative8",
            "embeddingRounding": "numpy-round-to-even",
            "nibbleOrder": "low-nibble-first",
        },
    )
