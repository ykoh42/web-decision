"""Preserve the exported graph's BF16 storage and rounding in portable TVM IR."""

from copy import deepcopy
from math import prod

import numpy as np
import torch
from tvm import te, tirx
from tvm.relax.frontend.torch import from_exported_program

NORM_BLOCK_WIDTH = 32


@torch.library.custom_op("webdecision::round_bf16_v1", mutates_args=())
def round_bf16(value: torch.Tensor) -> torch.Tensor:
    return value.bfloat16().float()


@round_bf16.register_fake
def _round_fake(value):
    return torch.empty_like(value)


@torch.library.custom_op("webdecision::unpack_bf16_v1", mutates_args=())
def unpack_bf16(words: torch.Tensor, shape: list[int]) -> torch.Tensor:
    return words.view(torch.bfloat16)[: prod(shape)].reshape(shape).float()


@unpack_bf16.register_fake
def _unpack_fake(words, shape):
    return torch.empty(shape, dtype=torch.float32, device=words.device)


def pack_parameter(root, target):
    location, _, name = target.rpartition(".")
    parent = root.get_submodule(location)
    parameter = getattr(parent, name)
    halves = parameter.detach().view(torch.uint16).numpy().reshape(-1)
    words = np.pad(halves, (0, len(halves) % 2)).view(np.uint32)
    # Export the layer's BF16 scalar buffer as an external parameter as well.
    setattr(
        parent, name, torch.nn.Parameter(torch.from_numpy(words), requires_grad=False)
    )
    return list(parameter.shape)


def transport_dtype(value):
    if value is torch.bfloat16:
        return torch.float32
    if isinstance(value, tuple):
        return tuple(map(transport_dtype, value))
    if isinstance(value, list):
        return list(map(transport_dtype, value))
    if isinstance(value, dict):
        return {name: transport_dtype(item) for name, item in value.items()}
    return value


def preserve_bf16(exported):
    original = exported.module()
    root = deepcopy(original)
    graph = torch.fx.Graph()
    values = {}
    for source in original.graph.nodes:
        node = graph.node_copy(source, values.__getitem__)
        node.args = transport_dtype(node.args)
        node.kwargs = transport_dtype(node.kwargs)
        if node.target == torch.ops.aten.mul_.Tensor:
            # Gemma multiplies its final residual by layer_scalar in place.
            if len(source.args[0].users) != 1:
                raise ValueError("The Gemma layer scale must have no other users")
            node.target = torch.ops.aten.mul.Tensor
        if getattr(source.meta.get("val"), "dtype", None) == torch.bfloat16:
            node = transport_bf16_node(graph, root, source, node)
        values[source] = node
    return torch.fx.GraphModule(root, graph)


def transport_bf16_node(graph, root, source, node):
    if source.op == "get_attr":
        shape = pack_parameter(root, source.target)
        return graph.call_function(
            torch.ops.webdecision.unpack_bf16_v1.default, (node, shape)
        )
    if source.op == "call_function":
        # Export records the original result dtype, including PEFT's final cast.
        return graph.call_function(torch.ops.webdecision.round_bf16_v1.default, (node,))
    return node


def uint32(value):
    return tirx.const(value, "uint32")


def round_compute(value):
    def at(*indices):
        bits = tirx.reinterpret("uint32", value[indices])
        rounded = (bits + uint32(0x7FFF) + ((bits >> uint32(16)) & uint32(1))) & uint32(
            0xFFFF0000
        )
        nan = ((bits & uint32(0x7F800000)) == uint32(0x7F800000)) & (
            (bits & uint32(0x7FFFFF)) != uint32(0)
        )
        return tirx.reinterpret("float32", tirx.if_then_else(nan, bits, rounded))

    return te.compute(value.shape, at, name="round_bf16")


def unpack_compute(words, shape):
    def at(*indices):
        flat = 0
        for index, extent in zip(indices, shape, strict=True):
            flat = flat * extent + index
        half = words[flat // 2] >> tirx.Cast("uint32", (flat % 2) * 16)
        return tirx.reinterpret("float32", half << uint32(16))

    return te.compute(shape, at, name="unpack_bf16")


def convert_round(node, importer):
    return importer.block_builder.emit_te(round_compute, importer.env[node.args[0]])


def convert_unpack(node, importer):
    return importer.block_builder.emit_te(
        unpack_compute, importer.env[node.args[0]], node.args[1]
    )


def norm_mean_compute(value):
    width = int(value.shape[-1])
    if width % NORM_BLOCK_WIDTH:
        raise ValueError("Expected a pinned Gemma RMSNorm width")
    # Bounded FP32 sums limit error before Gemma rounds the normalized values.
    block = te.reduce_axis((0, NORM_BLOCK_WIDTH), "column")
    groups = width // NORM_BLOCK_WIDTH
    partial = te.compute(
        (*value.shape[:-1], groups),
        lambda *indices: te.sum(
            value[(*indices[:-1], indices[-1] * NORM_BLOCK_WIDTH + block)], axis=block
        ),
        name="norm_block_sum",
    )
    group = te.reduce_axis((0, groups), "group")
    total = te.compute(
        (*value.shape[:-1], 1),
        lambda *indices: te.sum(partial[(*indices[:-1], group)], axis=group),
        name="norm_total_sum",
    )
    scale = tirx.const(1 / width, "float32")
    return te.compute(
        total.shape, lambda *indices: total[indices] * scale, name="norm_mean"
    )


def convert_norm_mean(node, importer):
    if node.args[1:] != ([-1], True) or node.kwargs:
        raise ValueError("Expected Gemma RMSNorm to reduce the final axis in FP32")
    return importer.block_builder.emit_te(norm_mean_compute, importer.env[node.args[0]])


def lower_bf16_program(program, example, shapes):
    converted = torch.export.export(
        preserve_bf16(program),
        tuple(value.float() for value in example),
        dynamic_shapes=shapes,
        strict=False,
    )
    return from_exported_program(
        converted,
        unwrap_unit_return_tuple=True,
        keep_params_as_input=True,
        custom_convert_map={
            "round_bf16_v1.default": convert_round,
            "unpack_bf16_v1.default": convert_unpack,
            "mean.dim": convert_norm_mean,
        },
    )
