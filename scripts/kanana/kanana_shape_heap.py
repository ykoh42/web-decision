"""Reproduce the host shape-heap store order of the published Kanana WASM."""

from dataclasses import dataclass

import tvm
from tvm import tirx


@dataclass(frozen=True)
class Product:
    destination: int
    source: int
    coefficient: int
    degree: int = 1


PUBLISHED_PRODUCTS = {
    "text": (
        Product(16, 0, 10240),
        Product(15, 0, 32768),
        Product(10, 0, 512),
        Product(9, 0, 240),
        Product(6, 0, 256),
        Product(7, 0, 96),
        Product(5, 0, 768),
        Product(14, 0, 128),
        Product(2, 0, 5120),
        Product(13, 0, 4),
        Product(12, 0, 4096),
        Product(11, 0, 16384),
        Product(8, 0, 80),
        Product(3, 0, 10240),
        Product(4, 1, 10240),
    ),
    "vision": (
        Product(20, 1, 10240),
        Product(19, 1, 18432),
        Product(18, 1, 68864),
        Product(12, 1, 18432),
        Product(9, 1, 73728),
        Product(8, 1, 1152),
        Product(11, 1, 18432),
        Product(7, 1, 288),
        Product(17, 1, 18432),
        Product(6, 1, 576),
        Product(16, 1, 18432),
        Product(10, 1, 36864),
        Product(5, 1, 73728),
        Product(13, 1, 18432),
        Product(4, 1, 16),
        Product(3, 1, 18432),
        Product(15, 1, 1024, 2),
        Product(14, 1, 1024, 2),
        Product(0, 1, 4),
        Product(2, 1, 4),
    ),
}


# The pinned emcc toolchain can move independent stores after TVM emits LLVM IR.
# These TIR input orders reproduce the published WASM store bodies in that toolchain.
TIR_STORE_ORDER = {
    "text": (16, 15, 10, 9, 6, 5, 7, 2, 14, 12, 13, 11, 8, 3, 4),
    "vision": (20, 19, 18, 12, 9, 8, 7, 11, 6, 17, 10, 16, 5, 4, 13, 3, 15, 14, 2, 0),
}


def slot(indices):
    if len(indices) != 1 or not isinstance(indices[0], tirx.IntImm):
        raise ValueError("Kanana shape heap requires a constant scalar index.")
    return int(indices[0])


def monomial(expr, buffer):
    if isinstance(expr, tirx.IntImm):
        return int(expr), ()
    if isinstance(expr, tvm.ir.TensorLoad) and expr.source.same_as(buffer):
        return 1, (slot(expr.indices),)
    if isinstance(expr, tirx.Mul):
        left_factor, left_slots = monomial(expr.a, buffer)
        right_factor, right_slots = monomial(expr.b, buffer)
        return left_factor * right_factor, left_slots + right_slots
    if isinstance(expr, tirx.LShift) and isinstance(expr.b, tirx.IntImm):
        factor, slots = monomial(expr.a, buffer)
        return factor << int(expr.b), slots
    raise ValueError("Unexpected Kanana shape heap expression.")


def ordered_shape_heap(graph, func):
    products = PUBLISHED_PRODUCTS[graph]
    body = func.body
    if not isinstance(body, tirx.SeqStmt) or len(body.seq) != len(products):
        raise ValueError(f"Unexpected Kanana {graph} shape heap body.")
    if not all(isinstance(store, tirx.BufferStore) for store in body.seq):
        raise ValueError(f"Kanana {graph} shape heap requires plain buffer stores.")
    buffer = body.seq[0].buffer
    if str(buffer.dtype) != "int64" or len(buffer.shape) != 1:
        raise ValueError(f"Unexpected Kanana {graph} shape heap buffer.")
    if not all(store.buffer.same_as(buffer) for store in body.seq):
        raise ValueError(f"Kanana {graph} shape heap stores must share one buffer.")
    stores = {slot(store.indices): store for store in body.seq}
    if set(stores) != {product.destination for product in products}:
        raise ValueError(f"Unexpected Kanana {graph} shape heap destinations.")

    for product in products:
        store = stores[product.destination]
        if product.source in stores:
            raise ValueError(f"Kanana {graph} shape heap product reads a destination.")
        actual = monomial(store.value, buffer)
        expected = product.coefficient, (product.source,) * product.degree
        if actual != expected:
            raise ValueError(
                f"Unexpected Kanana {graph} shape heap product at slot {product.destination}: "
                f"expected {expected}, got {actual}."
            )
    order = [stores[destination] for destination in TIR_STORE_ORDER[graph]]
    return func.with_body(tirx.SeqStmt(order, body.span), func.span)


def canonicalize_shape_heap(graph):
    @tvm.transform.module_pass(opt_level=0, name="KananaPublishedShapeHeapOrder")
    def transform(mod, _ctx):
        for symbol, func in mod.functions.items():
            if symbol.name_hint == "shape_func" and isinstance(func, tirx.PrimFunc):
                if func.attrs.get("tirx.is_host_func") != 1:
                    break
                mod.update_func(symbol, ordered_shape_heap(graph, func))
                return mod
        raise ValueError(f"Cannot find Kanana {graph} host shape_func.")

    return transform
