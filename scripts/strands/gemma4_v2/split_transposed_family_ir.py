"""Split a Gemma decoder family into attention and a reusable FFN tail."""

import gc

import torch
from decoder_family_ir import convert_attention, load_layer
from packed_parameters import pack_parameters
from split_layers import AttentionPart, Tail
from transpose_reuse_ir import tail as build_optimized_tail
from transpose_reuse_ir import unpack_transpose
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program


def build(checkpoint, index):
    _, block, scale, shared = load_layer(checkpoint, index)
    n = 7
    width = block.self_attn.head_dim
    hidden = torch.zeros(1, n, 1536)
    per_layer = torch.zeros(1, n, 256)
    cosine = torch.ones(1, n, width)
    sine = torch.zeros(1, n, width)
    mask = torch.tensor([3], dtype=torch.int32)
    valid = torch.ones(1, n, dtype=torch.int32)
    cache = (torch.zeros(1, 1, n, width), torch.zeros(1, 1, n, width)) if shared else ()
    length = torch.export.Dim("length", min=2)
    dyn = (
        {1: length},
        {1: length},
        {1: length},
        None,
        {1: length},
        *((({2: length}, {2: length}),) if shared else ()),
    )
    program = torch.export.export(
        AttentionPart(block, shared).eval(),
        (hidden, cosine, sine, mask, valid, *cache),
        dynamic_shapes=dyn,
        strict=False,
    )
    names_a = [name for name, _ in program.named_parameters()]
    mod = from_exported_program(
        program,
        unwrap_unit_return_tuple=True,
        keep_params_as_input=True,
        custom_convert_map={"prefix_attention.default": convert_attention},
    )
    mod, params = relax.frontend.detach_params(mod)
    mod_a, _ = pack_parameters(mod, params["main"], 7 if shared else 5)
    del program, mod, params
    length = torch.export.Dim("chunk", min=1)
    program = torch.export.export(
        Tail(block).eval(),
        (hidden, per_layer, scale),
        dynamic_shapes=({1: length}, {1: length}, None),
        strict=False,
    )
    names_t = [name for name, _ in program.named_parameters()]
    mod = from_exported_program(
        program, unwrap_unit_return_tuple=True, keep_params_as_input=True
    )
    mod, params = relax.frontend.detach_params(mod)
    mod_u = unpack_transpose(params["main"])
    mod_t, _ = build_optimized_tail(block, hidden, per_layer, scale)
    print("split weight order", index, names_a, names_t, flush=True)
    del program, mod, params, block
    gc.collect()
    return mod_a, mod_u, mod_t
