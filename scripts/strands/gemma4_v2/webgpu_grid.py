"""Give Gemma's large WebGPU launches a model-side workgroup tail guard."""

import tvm
import tvm_ffi
from tvm import tirx

GRID_ALIGNMENT = 256
GRID_LIMIT = 65535


def grid_axes(body):
    axes = set()

    def record(loop):
        binding = loop.thread_binding
        if binding is not None and binding.thread_tag.startswith("blockIdx."):
            axes.add(binding.thread_tag)

    tvm_ffi.structural_walk(body, (tirx.For, record))
    return axes


def guard_loop(loop, mutator):
    binding = loop.thread_binding
    extent = loop.extent
    if (
        binding is None
        or binding.thread_tag != "blockIdx.x"
        or isinstance(extent, tirx.IntImm)
        and extent.value <= GRID_LIMIT
    ):
        return mutator.default_mutate(loop)
    if (
        not isinstance(loop.min, tirx.IntImm)
        or loop.min.value != 0
        or loop.step is not None
    ):
        raise ValueError("Unexpected Gemma grid origin or step")

    # Let TVM fold the x extent into z without changing its original reduction binding.
    aligned = tirx.ceildiv(extent, GRID_ALIGNMENT) * GRID_ALIGNMENT
    launched = tirx.Select(extent <= GRID_LIMIT, extent, aligned)
    guarded = tirx.IfThenElse(loop.loop_var < extent, loop.body, None)
    return tirx.For(
        loop.loop_var,
        loop.min,
        launched,
        loop.kind,
        guarded,
        binding,
        loop.annotations,
        span=loop.span,
    )


@tvm.transform.module_pass(opt_level=0, name="GemmaWebGPUGrid")
def guard_model_grids(module, _context):
    updated = {}
    for name, function in module.functions.items():
        if isinstance(function, tirx.PrimFunc) and grid_axes(function.body) == {
            "blockIdx.x"
        }:
            body = tvm_ffi.structural_mutate(function.body, (tirx.For, guard_loop))
            function = function.with_body(body)
        updated[name] = function
    return tvm.IRModule(updated, attrs=module.attrs)
