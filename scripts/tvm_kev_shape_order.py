"""Canonicalize only the verified independent stores in Kev's dynamic shape function."""

import tvm
from tvm import tirx
from tvm.ir import prim


def _is_single_integer_store(store):
    return (
        isinstance(store, tirx.BufferStore)
        and len(store.indices) == 1
        and isinstance(store.indices[0], prim.IntImm)
    )


def _validate_stores(fn):
    if not isinstance(fn.body, tirx.SeqStmt) or len(fn.body.seq) != 12:
        raise ValueError("Unexpected Kev shape function; refusing to reorder.")

    stores = list(fn.body.seq)
    if not all(_is_single_integer_store(store) for store in stores):
        raise ValueError("Unexpected Kev shape stores; refusing to reorder.")

    buffer = stores[0].buffer
    if len(fn.params) != 1 or not buffer.same_as(fn.params[0]):
        raise ValueError("Unexpected shape function parameter; refusing to reorder.")

    indices = [int(store.indices[0]) for store in stores]
    if sorted(indices) != list(range(1, 13)) or any(
        not store.buffer.same_as(buffer) for store in stores
    ):
        raise ValueError("Unexpected shape buffer or indexes; refusing to reorder.")
    return stores, buffer


def _is_supported_expression(expression, buffer):
    if isinstance(expression, prim.IntImm):
        return True
    if isinstance(expression, prim.Mul):
        return _is_supported_expression(
            expression.a, buffer
        ) and _is_supported_expression(expression.b, buffer)
    if isinstance(expression, tvm.ir.expr.TensorLoad):
        return (
            expression.source.same_as(buffer)
            and len(expression.indices) == 1
            and isinstance(expression.indices[0], prim.IntImm)
            and int(expression.indices[0]) == 0
        )
    return False


def normalize(mod):
    """Sort Kev's twelve output writes after proving they are independent."""
    gv = mod.get_global_var("shape_func")
    fn = mod[gv]
    stores, buffer = _validate_stores(fn)
    if not all(_is_supported_expression(store.value, buffer) for store in stores):
        raise ValueError("Shape stores have unknown dependencies; refusing to reorder.")

    normalized = sorted(stores, key=lambda store: int(store.indices[0]))
    mod.update_func(gv, fn.with_body(tirx.SeqStmt(normalized)))
    return mod
