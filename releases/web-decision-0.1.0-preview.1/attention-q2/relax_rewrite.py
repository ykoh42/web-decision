from __future__ import annotations

from tvm import relax, tirx

from online_tir import attention_primfunc


def _bound_calls(function):
    bindings = {}
    for block in function.body.blocks:
        for binding in block.bindings:
            bindings[str(binding.var)] = binding.value
    return bindings


def _op_name(expression) -> str | None:
    return getattr(expression.op, "name", None)


def _is_variable(expression) -> bool:
    return isinstance(expression, relax.Var)


def _call_for(bindings, expression):
    if not _is_variable(expression):
        return None
    return bindings.get(str(expression))


def _attention_inputs(bindings, output_call):
    if _op_name(output_call) != "relax.matmul" or len(output_call.args) != 2:
        return None
    softmax = _call_for(bindings, output_call.args[0])
    if softmax is None or _op_name(softmax) != "relax.nn.softmax":
        return None
    if len(softmax.args) != 1:
        return None

    add = _call_for(bindings, softmax.args[0])
    if add is None or _op_name(add) != "relax.add" or len(add.args) != 2:
        return None
    scaled = _call_for(bindings, add.args[0])
    bias = add.args[1]
    if scaled is None or _op_name(scaled) != "relax.multiply" or len(scaled.args) != 2:
        return None
    qk = _call_for(bindings, scaled.args[0])
    if qk is None or _op_name(qk) != "relax.matmul" or len(qk.args) != 2:
        return None
    scale = scaled.args[1]
    if str(scale) != 'R.const(0.08837890625, "float16")':
        return None

    query, key = qk.args
    value = output_call.args[1]
    if not all(_is_variable(item) for item in (query, key, bias, value)):
        return None
    if str(query.ty) != 'R.Tensor((1, 16, sequence_length, 128), dtype="float16")':
        return None
    if str(key.ty) != 'R.Tensor((1, 16, 128, sequence_length), dtype="float16")':
        return None
    if str(bias.ty) != 'R.Tensor((1, 1, sequence_length, sequence_length), dtype="float16")':
        return None
    if str(value.ty) != 'R.Tensor((1, 16, sequence_length, 128), dtype="float16")':
        return None
    return query, key, bias, value


def rewrite_attention(mod):
    """Replace each exact Kev QK-scale-bias-softmax-V chain with one call_tir."""
    main = mod["main"]
    bindings = _bound_calls(main)
    builder = relax.BlockBuilder(mod)
    kernel = builder.add_func(attention_primfunc(), "kev_attention_query_tile")
    mod = builder.get()

    @relax.expr_functor.mutator
    class AttentionRewriter(relax.PyExprMutator):
        def __init__(self, current):
            super().__init__(current)
            self.count = 0

        def visit_call_(self, call):
            updated = super().visit_call_(call)
            inputs = _attention_inputs(bindings, updated)
            if inputs is None:
                return updated
            self.count += 1
            query, key, bias, value = inputs
            sequence_length = relax.prim_value(
                tirx.Cast("int32", query.ty.shape[2])
            )
            return relax.call_tir(
                kernel,
                relax.Tuple([query, key, value, bias, sequence_length]),
                updated.ty,
            )

    rewriter = AttentionRewriter(mod)
    rewritten_main = rewriter.visit_expr(mod["main"])
    if rewriter.count != 28:
        raise ValueError(f"Expected 28 exact attention chains, rewrote {rewriter.count}")
    mod.update_func(mod.get_global_var("main"), rewritten_main)
    calls = 0
    for block in mod["main"].body.blocks:
        for binding in block.bindings:
            call = binding.value
            if (
                _op_name(call) == "relax.call_tir"
                and call.args[0].name_hint == "kev_attention_query_tile"
            ):
                calls += 1
    if calls != 28:
        raise ValueError(f"Expected 28 query-row call_tir nodes, found {calls}")
    relax.analysis.well_formed(mod)
    return mod, rewriter.count
