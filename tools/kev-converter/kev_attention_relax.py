"""Public Relax rewrite for replacing Kev's expanded dense attention graph."""

from dataclasses import dataclass

from kev_attention_grid import ATTENTION_SCALE, attention_grid
from kev_preprocess import (
    PREPROCESS_FIELDS,
    PREPROCESS_NEXT_DECISION,
    PREPROCESS_POSITION_IDS,
    PREPROCESS_SEGMENTS,
    PREPROCESS_VALID,
    build_kev_preprocessor_module,
)
from tvm import relax

ATTENTION_FUNCTION_NAME = "kev_attention_grid"
DEFAULT_METADATA_BINDINGS = {
    "lv5": PREPROCESS_VALID,
    "lv23": PREPROCESS_SEGMENTS,
    "lv59": PREPROCESS_POSITION_IDS,
    "lv3273": PREPROCESS_NEXT_DECISION,
}


@dataclass(frozen=True)
class KevAttentionRewriteReport:
    """Counts and metadata about one full-module attention rewrite."""

    attention_blocks: int
    metadata_bindings: tuple[str, ...]
    preprocess_calls: int
    fused_function: str = ATTENTION_FUNCTION_NAME


def _resolve_call(expression, operation_name, lookup_binding):
    """Resolve one bound call and verify its Relax operator name."""
    if isinstance(expression, relax.Var):
        expression = lookup_binding(expression)
    if isinstance(expression, relax.Call) and expression.op.name == operation_name:
        return expression
    return None


def _type_of(expression):
    """Return the pinned Relax TensorType for an expression."""
    return expression.ty


def _scalar_constant(expression):
    """Read a scalar Relax/TIR constant across pinned FFI representations."""
    value = getattr(expression, "value", None)
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _trace_call_chain(expression, operation_names, lookup_binding):
    """Trace a first-argument Relax chain, returning each matched call."""
    calls = []
    current = expression
    for operation_name in operation_names:
        call = _resolve_call(current, operation_name, lookup_binding)
        if call is None:
            return None
        calls.append(call)
        current = call.args[0]
    return calls


def _attention_operands(pv_call, lookup_binding):
    """Extract Q, K, and V only from the exported FP16-boundary chain."""
    chain = _trace_call_chain(
        pv_call.args[0],
        (
            "relax.astype",
            "relax.reshape",
            "relax.broadcast_to",
            "relax.astype",
            "relax.nn.softmax",
            "relax.astype",
            "relax.add",
            "relax.multiply",
            "relax.reshape",
            "relax.matmul",
        ),
        lookup_binding,
    )
    if chain is None:
        return None

    add_call = chain[6]
    multiply_call = chain[7]
    mask_cast = _resolve_call(add_call.args[1], "relax.astype", lookup_binding)
    if mask_cast is None or _type_of(mask_cast.args[0]).dtype != "float16":
        return None
    score_cast = chain[5]
    if _type_of(score_cast.args[0]).dtype != "float32":
        return None
    qk_call = chain[9]
    scale_value = _scalar_constant(multiply_call.args[1])
    if scale_value is None or abs(scale_value - ATTENTION_SCALE) >= 1e-8:
        return None
    return qk_call.args[0], qk_call.args[1], pv_call.args[1]


def _is_supported_attention_type(query, key, value, output_type):
    """Check the static head/dimension contract and emitted dtype boundaries."""
    query_type = _type_of(query)
    key_type = _type_of(key)
    value_type = _type_of(value)
    if not all(
        isinstance(info, relax.TensorType)
        for info in (query_type, key_type, value_type, output_type)
    ):
        return False
    if (
        len(query_type.shape) != 3
        or len(key_type.shape) != 3
        or len(value_type.shape) != 3
        or len(output_type.shape) != 3
    ):
        return False
    return (
        query_type.dtype == "float16"
        and key_type.dtype == "float16"
        and value_type.dtype == "float32"
        and output_type.dtype == "float32"
        and query_type.shape[0] == 16
        and query_type.shape[2] == 128
        and key_type.shape[0] == 16
        and key_type.shape[1] == 128
        and value_type.shape[0] == 16
        and value_type.shape[2] == 128
    )


@relax.expr_functor.mutator
class KevAttentionGraphMutator(relax.PyExprMutator):
    """Replace expanded QK/mask/softmax/PV chains and dense wrapper metadata."""

    def __init__(
        self,
        module,
        fused_gv,
        preprocess_var,
        sequence_length,
        metadata_bindings=None,
    ):
        super().__init__(module)
        self.fused_gv = fused_gv
        self.preprocess_var = preprocess_var
        self.sequence_length = sequence_length
        self.metadata_bindings = dict(
            DEFAULT_METADATA_BINDINGS
            if metadata_bindings is None
            else metadata_bindings
        )
        self.required_metadata = set(self.metadata_bindings)
        self.valid_binding_name = next(
            name
            for name, field in self.metadata_bindings.items()
            if field == PREPROCESS_VALID
        )
        self.segment_binding_name = next(
            name
            for name, field in self.metadata_bindings.items()
            if field == PREPROCESS_SEGMENTS
        )
        self.seen_metadata = set()
        self.valid_var = None
        self.segment_var = None
        self.attention_count = 0

    def _metadata_field(self, field_index):
        """Slice one compact field into the wrapper's `(1, S)` shape."""
        selected = relax.op.strided_slice(
            self.preprocess_var,
            axes=[0, 2],
            begin=[field_index, 0],
            end=[field_index + 1, self.sequence_length],
        )
        return relax.op.reshape(selected, (1, self.sequence_length))

    def visit_var_binding_(self, binding):
        """Replace selected wrapper fields before their dense consumers run."""
        name = str(binding.var.name)
        if name not in self.metadata_bindings:
            return super().visit_var_binding_(binding)
        field_index = self.metadata_bindings[name]
        replacement = self._metadata_field(field_index)
        self.seen_metadata.add(name)
        if name == self.valid_binding_name:
            self.valid_var = binding.var
        elif name == self.segment_binding_name:
            self.segment_var = binding.var
        return super().visit_var_binding_(relax.VarBinding(binding.var, replacement))

    def visit_call_(self, call):
        """Replace a complete dense attention block with one TIRX call."""
        updated = super().visit_call_(call)
        if call.op.name != "relax.matmul":
            return updated
        if self.valid_var is None or self.segment_var is None:
            return updated
        operands = _attention_operands(updated, self.lookup_binding)
        if operands is None:
            return updated
        query, key, value = operands
        if not _is_supported_attention_type(query, key, value, updated.ty):
            return updated
        valid = self.visit_expr(self.valid_var)
        segments = self.visit_expr(self.segment_var)
        result = relax.call_tir(
            self.fused_gv,
            [query, key, value, segments, valid, self.sequence_length],
            updated.ty,
        )
        self.attention_count += 1
        return result


def rewrite_kev_attention(
    module,
    question_id,
    decide_id,
    expected_attention_blocks=28,
    metadata_bindings=None,
    target=None,
):
    """Fuse all Kev attention blocks and replace quadratic wrapper metadata.

    The source graph contract is the pinned Qwen3 attention form: 16 expanded
    query/key/value heads, 128 channels, FP16 Q/K, FP32 V, and FP16 probabilities
    between FP32 softmax and FP32 PV accumulation. `metadata_bindings` can map a
    changed export's Relax variable names to fields emitted by `kev_preprocess`.
    """
    module = relax.transform.Normalize()(module)
    main = module["main"]
    if len(main.params) < 2:
        raise ValueError("Kev attention rewrite needs input_ids and attention_mask.")
    input_ids, attention_mask = main.params[:2]
    input_shape = input_ids.ty.shape
    if len(input_shape) != 2 or int(input_shape[0]) != 1:
        raise ValueError("Kev attention rewrite supports the pinned batch size of one.")

    setup_builder = relax.BlockBuilder(module)
    fused_gv = setup_builder.add_func(attention_grid, ATTENTION_FUNCTION_NAME)
    preprocess_module = build_kev_preprocessor_module(
        question_id, decide_id, target=target
    )
    preprocess_gv = setup_builder.add_func(
        preprocess_module["kev_preprocess"], "kev_preprocess"
    )

    input_shape_expr = relax.get_shape_of(input_ids)
    batch_size = input_shape_expr[0]
    sequence_length = input_shape_expr[1]
    preprocess_type = relax.TensorType(
        (PREPROCESS_FIELDS, input_shape[0], sequence_length + 1),
        "int32",
    )
    preprocess_var = relax.Var("kev_preprocess", preprocess_type)
    preprocess_call = relax.call_tir(
        preprocess_gv,
        [
            input_ids,
            attention_mask,
            batch_size,
            sequence_length,
            relax.prim_value(question_id, dtype="int32"),
            relax.prim_value(decide_id, dtype="int32"),
        ],
        preprocess_type,
    )
    preprocess_block = relax.BindingBlock(
        [relax.VarBinding(preprocess_var, preprocess_call)]
    )
    if isinstance(main.body, relax.SeqExpr):
        prepared_body = relax.SeqExpr(
            [preprocess_block, *main.body.blocks], main.body.body, main.body.span
        )
    else:
        prepared_body = relax.SeqExpr([preprocess_block], main.body)
    prepared_main = relax.Function(
        main.params,
        prepared_body,
        ret_ty=main.ret_ty,
        is_pure=main.is_pure,
        attrs=main.attrs,
        span=main.span,
    )
    prepared_module = setup_builder.get()
    main_gv = prepared_module.get_global_var("main")
    prepared_module.update_func(main_gv, prepared_main)

    mutator = KevAttentionGraphMutator(
        prepared_module,
        fused_gv,
        preprocess_var,
        sequence_length,
        metadata_bindings=metadata_bindings,
    )
    main = prepared_module["main"]
    body = mutator.visit_expr(main.body)
    updated_main = relax.Function(
        main.params,
        body,
        ret_ty=main.ret_ty,
        is_pure=main.is_pure,
        attrs=main.attrs,
        span=main.span,
    )
    builder = relax.BlockBuilder(prepared_module)
    builder.update_func(main_gv, updated_main)
    rewritten = builder.get()
    rewritten = relax.transform.DeadCodeElimination()(rewritten)
    rewritten = relax.transform.Normalize()(rewritten)
    relax.analysis.well_formed(rewritten)
    missing_metadata = mutator.required_metadata - mutator.seen_metadata
    if missing_metadata:
        raise ValueError(
            "Kev attention rewrite could not find metadata bindings: "
            + ", ".join(sorted(missing_metadata))
        )
    if mutator.attention_count != expected_attention_blocks:
        raise ValueError(
            "Kev attention replacement count mismatch: "
            f"{mutator.attention_count}/{expected_attention_blocks}"
        )
    return rewritten, KevAttentionRewriteReport(
        attention_blocks=mutator.attention_count,
        metadata_bindings=tuple(sorted(mutator.seen_metadata)),
        preprocess_calls=1,
    )
