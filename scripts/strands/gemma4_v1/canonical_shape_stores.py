"""Canonicalize provably independent integer stores using public TVM IR."""

import tvm
import tvm_ffi
from tvm import tirx as t


def constant_index(indices, size):
    if len(indices) != 1:
        return None
    index = indices[0]
    if not isinstance(index, t.IntImm) or not 0 <= index.value < size:
        return None
    return index.value


def independent_integer_value(value, buffer, size, writes):
    if isinstance(value, t.IntImm):
        return True
    if isinstance(value, tvm.ir.TensorLoad):
        index = constant_index(value.indices, size)
        return (
            value.source.same_as(buffer) and index is not None and index not in writes
        )
    if isinstance(value, (t.Add, t.Sub, t.Mul)):
        return independent_integer_value(
            value.a, buffer, size, writes
        ) and independent_integer_value(value.b, buffer, size, writes)
    return False


def shape_store_buffer(func):
    attrs = func.attrs
    if not attrs or not attrs.get("tirx.is_host_func"):
        return None
    name = attrs.get("global_symbol", "")
    suffix = str(name).removeprefix("shape_func")
    if not str(name).startswith("shape_func") or (suffix and not suffix.isdigit()):
        return None
    if len(func.params) != 1 or not isinstance(func.body, t.SeqStmt):
        return None
    if not all(isinstance(stmt, t.BufferStore) for stmt in func.body.seq):
        return None
    buffer = func.body.seq[0].buffer
    if (
        not buffer.same_as(func.params[0])
        or str(buffer.dtype) != "int64"
        or len(buffer.shape) != 1
    ):
        return None
    size = buffer.shape[0]
    if not isinstance(size, t.IntImm) or buffer.strides or buffer.ty.allocated_addr:
        return None
    ordinary = t.decl_buffer((size,), "int64", "ordinary")
    if buffer.ty.layout is not None and not tvm_ffi.structural_equal(
        buffer.ty.layout, ordinary.ty.layout
    ):
        return None
    return buffer, size


def independent_stores(func, buffer, size):
    if not all(store.buffer.same_as(buffer) for store in func.body.seq):
        return None
    keyed = {
        constant_index(store.indices, size.value): store for store in func.body.seq
    }
    if None in keyed or len(keyed) != len(func.body.seq):
        return None
    if not all(
        independent_integer_value(store.value, buffer, size.value, keyed)
        for store in keyed.values()
    ):
        return None
    return [keyed[index] for index in sorted(keyed)]


@t.transform.prim_func_pass(opt_level=0)
def canonical_stores(func, mod, ctx):
    matched = shape_store_buffer(func)
    if matched is None:
        return func
    stores = independent_stores(func, *matched)
    if stores is None:
        return func
    return func.with_body(t.SeqStmt(stores))
