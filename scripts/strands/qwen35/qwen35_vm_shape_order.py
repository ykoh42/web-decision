"""Keep independent Qwen VM shape calculations in a reproducible order."""

from tvm import ir, tirx

ARITHMETIC = (tirx.Add, tirx.Sub, tirx.Mul, tirx.Min, tirx.Max)
DIVISION = (tirx.Div, tirx.Mod, tirx.FloorDiv, tirx.FloorMod)


def _slot(heap, buffer, indices):
    if not buffer.same_as(heap) or len(indices) != 1:
        raise ValueError("Expected an access to the Qwen shape heap")
    index = indices[0]
    if not isinstance(index, tirx.IntImm) or not 0 <= int(index) < int(heap.shape[0]):
        raise ValueError("Expected a constant, in-bounds Qwen shape slot")
    return int(index)


def _read_slots(heap, expr):
    if isinstance(expr, tirx.IntImm):
        return set()
    if isinstance(expr, ir.TensorLoad):
        return {_slot(heap, expr.source, expr.indices)}
    if isinstance(expr, ARITHMETIC):
        return _read_slots(heap, expr.a) | _read_slots(heap, expr.b)
    if isinstance(expr, DIVISION) and isinstance(expr.b, tirx.IntImm) and int(expr.b):
        return _read_slots(heap, expr.a)
    raise ValueError(f"Unexpected Qwen host shape expression {type(expr).__name__}")


def order_vm_shape_stores(module):
    """Sort only the generated host stores that can run in any order."""
    name = module.get_global_var("shape_func")
    function = module[name]
    if (
        not isinstance(function, tirx.PrimFunc)
        or function.attrs is None
        or not function.attrs.get("tirx.is_host_func")
    ):
        raise ValueError("Expected a Qwen host shape function")
    if not isinstance(function.body, tirx.SeqStmt):
        raise ValueError("Expected a sequence of Qwen shape stores")
    stores = list(function.body.seq)
    if not stores or not all(isinstance(store, tirx.BufferStore) for store in stores):
        raise ValueError("Expected a flat sequence of Qwen shape stores")
    heap = stores[0].buffer
    if (
        len(heap.shape) != 1
        or not isinstance(heap.shape[0], tirx.IntImm)
        or heap.strides
        or heap.dtype != "int64"
    ):
        raise ValueError("Expected a fixed-size Qwen shape heap")

    by_slot = {_slot(heap, store.buffer, store.indices): store for store in stores}
    reads = set().union(*(_read_slots(heap, store.value) for store in stores))
    if len(by_slot) != len(stores) or not reads.isdisjoint(by_slot):
        raise ValueError("Qwen shape stores must be independent")
    # VMShapeLower can emit these stores in address-dependent map order.
    body = tirx.SeqStmt([by_slot[index] for index in sorted(by_slot)])
    module.update_func(name, function.with_body(body))
