"""Verify decoder launch bounds and guard only oversized independent workgroups."""

import tvm
import tvm_ffi
from causal_attention_ir import MAX_QUERY_TILE
from causal_layer_ir import MAX_LENGTH
from tvm import relax, s_tir, sym, tirx

GRID_LIMIT = 65535


def entry_shape_limits(entry, context):
    if entry.endswith("_attention_main"):
        return ((0, 2, MAX_QUERY_TILE), (1, 2, context))
    if entry.endswith("_project_main"):
        return ((0, 1, MAX_LENGTH),)
    if entry.endswith("_tail_main") or entry == "ple_main":
        return ((0, 1, MAX_QUERY_TILE),)
    if entry == "final_main":
        return ((0, 1, 256),)
    if entry == "pointer_main":
        return ((1, 1, 255),)
    raise ValueError(f"Unknown Gemma v1 graph entry point {entry}")


def model_shape_limits(module, context):
    limits = {}
    for name, function in module.functions.items():
        if not isinstance(function, relax.Function):
            continue
        entry = name.name_hint
        for argument, axis, limit in entry_shape_limits(entry, context):
            dim = function.params[argument].ty.shape.values[axis]
            if (
                not isinstance(dim, tirx.Var)
                or limits.setdefault(dim.name, limit) != limit
            ):
                raise ValueError(f"Unrecognized Gemma v1 graph dimension for {entry}")
    return limits


def grid_upper_bound(extent, limits):
    analyzer = sym.Analyzer()
    for var in tirx.analysis.undefined_vars(extent):
        if var.name not in limits:
            raise ValueError(f"Missing Gemma v1 grid bound for {var.name}")
        analyzer.bind(var, tvm.ir.Range(1, limits[var.name] + 1))
    return analyzer.const_int_bound(extent).max_value


def block_axis_guard(loop, limits):
    binding = loop.thread_binding
    if binding is None or not binding.thread_tag.startswith("blockIdx."):
        return None, False
    axis = binding.thread_tag
    extent = loop.extent
    upper = grid_upper_bound(extent, limits)
    if upper <= GRID_LIMIT:
        return axis, False
    if axis != "blockIdx.x" or upper >= 65536 * 256:
        raise ValueError(f"Gemma v1 launch does not fit the WebGPU {axis} bound")
    # The pinned runtime reserves z and folds x by powers of two. 256-aligned
    # launches in this bound need no runtime-added workgroups.
    return axis, not isinstance(extent, tirx.IntImm) or extent.value % 256 != 0


def needs_guard(function, limits):
    axes = set()
    oversized_x = False
    uses_shared = False

    def visit(loop):
        nonlocal oversized_x
        axis, guarded = block_axis_guard(loop, limits)
        if axis is not None:
            axes.add(axis)
        oversized_x |= guarded

    def record(block, _scope):
        nonlocal uses_shared
        uses_shared |= any(
            buffer.scope().startswith("shared") for buffer in block.alloc_buffers
        )

    tvm_ffi.structural_walk(function.body, (tirx.For, visit), (s_tir.SBlock, record))
    if oversized_x and (uses_shared or axes != {"blockIdx.x"}):
        raise ValueError(
            "Cannot guard a Gemma v1 oversized launch with shared memory or multiple grid axes"
        )
    return oversized_x


def verify_and_guard_grids(module, context):
    # This is the same model-local guard used by E2B v2, restricted to workgroup
    # independent kernels. Wrapping a shared softmax in this pass blocks ThreadSync.
    from webgpu_grid import guard_model_grids

    limits = model_shape_limits(module, context)
    selected = {
        name: fn
        for name, fn in module.functions.items()
        if isinstance(fn, tirx.PrimFunc) and needs_guard(fn, limits)
    }
    guarded = guard_model_grids(tvm.IRModule(selected))
    result = dict(module.functions.items())
    result.update(guarded.functions.items())
    return tvm.IRModule(result, attrs=module.attrs)
