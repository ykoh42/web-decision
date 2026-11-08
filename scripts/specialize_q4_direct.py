#!/usr/bin/env python3
"""Specialize Kev q4 ONNX without ever expanding checkpoint weights.

During shape-only simplification, quantized operators are represented by
shape-only FP16 graph inputs. They have no tensor contents and are never
allocated. After simplification, the original packed operators and external
q4 bytes/scales are restored. A second graph views the same bytes as uint32 for
TVM's WebGPU kernels.
"""

import argparse
import copy
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxsim
from onnx import TensorProto, external_data_helper, helper, numpy_helper
from safe_output import atomic_write, atomic_write_serialized_message, open_directory

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ShapeOnlyState:
    model: onnx.ModelProto
    initializers: dict[str, onnx.TensorProto]
    operators: dict[str, onnx.NodeProto]
    placeholder_names: set[str]
    counts: dict[str, int]


def rewrite_matmul_nbits(node, index):
    attrs = {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }
    n, k, block = attrs["N"], attrs["K"], attrs["block_size"]
    assert len(node.input) == 3 and attrs["bits"] == 4 and k % block == 0
    name = f"__expanded_weight_{index}"
    placeholder = helper.make_tensor_value_info(name, TensorProto.FLOAT16, [k, n])
    expanded = helper.make_node(
        "MatMul", [node.input[0], name], list(node.output), name=node.name + "_expanded"
    )
    return [expanded], placeholder, tuple(node.input[1:]), node


def rewrite_gather_block_quantized(node, index, initializers):
    attrs = {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }
    assert (
        len(node.input) == 3
        and attrs["block_size"] == 32
        and attrs["gather_axis"] == 0
        and attrs["quantize_axis"] == 1
    )
    rows, cols = initializers[node.input[0]].dims
    name = f"__expanded_embedding_{index}"
    placeholder = helper.make_tensor_value_info(name, TensorProto.FLOAT16, [rows, cols])
    expanded = helper.make_node(
        "Gather",
        [name, node.input[1]],
        list(node.output),
        name=node.name + "_expanded",
        axis=0,
    )
    return [expanded], placeholder, (node.input[0], node.input[2]), node


def rewrite_layer_norm(node, index):
    attrs = {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }
    assert (
        attrs.get("axis", -1) == -1
        and abs(attrs.get("epsilon", 1e-5) - 1e-6) < 1e-10
        and len(node.output) == 1
    )
    prefix = f"__expanded_norm_{index}"
    x, weight = node.input
    nodes = [
        helper.make_node("Cast", [x], [prefix + "_x"], to=TensorProto.FLOAT),
        helper.make_node("Cast", [weight], [prefix + "_w"], to=TensorProto.FLOAT),
        helper.make_node("Mul", [prefix + "_x", prefix + "_x"], [prefix + "_square"]),
        helper.make_node(
            "ReduceMean",
            [prefix + "_square", "__expanded_norm_axis"],
            [prefix + "_mean"],
            keepdims=1,
        ),
        helper.make_node(
            "Add", [prefix + "_mean", "__expanded_norm_epsilon"], [prefix + "_variance"]
        ),
        helper.make_node("Sqrt", [prefix + "_variance"], [prefix + "_root"]),
        helper.make_node(
            "Div", [prefix + "_x", prefix + "_root"], [prefix + "_normalized"]
        ),
        helper.make_node(
            "Mul", [prefix + "_normalized", prefix + "_w"], [prefix + "_scaled"]
        ),
        helper.make_node(
            "Cast", [prefix + "_scaled"], list(node.output), to=TensorProto.FLOAT16
        ),
    ]
    return nodes, None, (), None


def rewrite_node(node, index, initializers):
    if node.op_type == "MatMulNBits" and node.domain == "com.microsoft":
        return rewrite_matmul_nbits(node, index)
    if node.op_type == "GatherBlockQuantized" and node.domain == "com.microsoft":
        return rewrite_gather_block_quantized(node, index, initializers)
    if node.op_type == "SimplifiedLayerNormalization":
        return rewrite_layer_norm(node, index)
    return [copy.deepcopy(node)], None, (), None


def retain_initializers(original, folder, removed):
    kept = []
    for tensor in original.graph.initializer:
        if tensor.name in removed:
            continue
        tensor = copy.deepcopy(tensor)
        if tensor.data_location == TensorProto.EXTERNAL:
            info = {entry.key: entry.value for entry in tensor.external_data}
            assert int(info["length"]) < 1024 * 1024
            external_data_helper.load_external_data_for_tensor(tensor, str(folder))
            tensor.ClearField("external_data")
            tensor.data_location = TensorProto.DEFAULT
        kept.append(tensor)
    return kept


def set_input_shapes(model, seq, dynamic):
    for graph_input in model.graph.input:
        if graph_input.name not in ("input_ids", "attention_mask"):
            continue
        batch_dim, sequence_dim = graph_input.type.tensor_type.shape.dim
        batch_dim.ClearField("dim_param")
        batch_dim.dim_value = 1
        if dynamic:
            sequence_dim.ClearField("dim_value")
            sequence_dim.dim_param = "sequence_length"
        else:
            sequence_dim.ClearField("dim_param")
            sequence_dim.dim_value = seq


def make_shape_only_model(original, folder, seq, dynamic):
    initializers = {tensor.name: tensor for tensor in original.graph.initializer}
    operators = {}
    removed = set()
    placeholders = []
    nodes = []
    counts = {}
    for index, node in enumerate(original.graph.node):
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
        replacements, placeholder, removed_inputs, operator = rewrite_node(
            node, index, initializers
        )
        nodes.extend(replacements)
        if placeholder is not None:
            placeholders.append(placeholder)
        removed.update(removed_inputs)
        if operator is not None:
            operators[operator.name] = operator
    assert len(placeholders) == 197 and len(operators) == 197

    model = copy.deepcopy(original)
    constants = [
        numpy_helper.from_array(np.array([-1], np.int64), "__expanded_norm_axis"),
        numpy_helper.from_array(np.array(1e-6, np.float32), "__expanded_norm_epsilon"),
    ]
    kept = retain_initializers(original, folder, removed)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept + constants)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.input.extend(placeholders)
    set_input_shapes(model, seq, dynamic)
    return ShapeOnlyState(
        model=model,
        initializers=initializers,
        operators=operators,
        placeholder_names={value.name for value in placeholders},
        counts=counts,
    )


def simplify_shapes(state):
    print(
        f"Simplifying shapes using {len(state.placeholder_names)} content-free placeholders; no dense weights are loaded.",
        flush=True,
    )
    simplified, checked = onnxsim.simplify(state.model, check_n=0, skip_fuse_bn=True)
    assert checked
    inputs = {value.name for value in simplified.graph.input}
    assert state.placeholder_names <= inputs
    selected = [
        node for node in simplified.graph.node if node.name.endswith("_Q4_expanded")
    ]
    assert len(selected) == 197
    return simplified


def restore_q4_node(node, source_node, rename, counter):
    restored = copy.deepcopy(source_node)
    if source_node.op_type == "MatMulNBits":
        restored.input[0] = node.input[0]
        restored.output[0] = node.output[0]
        indexes = [1, 2]
    else:
        restored.input[1] = node.input[1]
        restored.output[0] = node.output[0]
        indexes = [0, 2]
    for index in indexes:
        old_name = source_node.input[index]
        packed_index = 1 if source_node.op_type == "MatMulNBits" else 0
        role = "packed" if index == packed_index else "scale"
        if old_name not in rename:
            rename[old_name] = f"q4_{role}_{len(rename):03d}"
        restored.input[index] = rename[old_name]
    restored.name = f"q4_{source_node.op_type}_{counter:03d}"
    return restored, counter + 1


def restore_expanded_nodes(nodes, operators):
    restored_nodes = []
    names = set()
    rename = {}
    counter = 0
    for node in nodes:
        if not node.name.endswith("_Q4_expanded"):
            restored_nodes.append(node)
            continue
        source_node = operators[node.name[:-9]]
        restored, counter = restore_q4_node(node, source_node, rename, counter)
        for index in (1, 2) if source_node.op_type == "MatMulNBits" else (0, 2):
            names.add(source_node.input[index])
        restored_nodes.append(restored)
    return restored_nodes, names, rename


def convert_packed_initializer_to_u32(tensor):
    external = {entry.key: entry.value for entry in tensor.external_data}
    length = int(external["length"])
    assert length % 4 == 0
    if tensor.data_type == TensorProto.INT4:
        rows, columns = tensor.dims
        assert columns % 8 == 0
        tensor.dims[:] = [rows, columns // 8]
    else:
        assert tensor.data_type == TensorProto.UINT8
        rows, groups, size = tensor.dims
        assert (groups * size) % 4 == 0
        tensor.dims[:] = [rows, groups * size // 4]
    tensor.data_type = TensorProto.UINT32
    assert math.prod(tensor.dims) * 4 == length


def restore_initializers(model, initializers, names, rename, as_u32):
    kept = list(model.graph.initializer)
    for name in sorted(names, key=lambda item: rename[item]):
        tensor = copy.deepcopy(initializers[name])
        tensor.name = rename[name]
        if as_u32 and tensor.name.startswith("q4_packed_"):
            convert_packed_initializer_to_u32(tensor)
        kept.append(tensor)
    return kept


def remove_shape_placeholders(model, placeholder_names):
    inputs = [
        value for value in model.graph.input if value.name not in placeholder_names
    ]
    del model.graph.input[:]
    model.graph.input.extend(inputs)
    used = {name for node in model.graph.node for name in node.input}
    assert not (used & placeholder_names)


def restore_model(simplified, operators, initializers, placeholder_names, as_u32):
    model = copy.deepcopy(simplified)
    nodes, names, rename = restore_expanded_nodes(model.graph.node, operators)
    kept = restore_initializers(model, initializers, names, rename, as_u32)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    remove_shape_placeholders(model, placeholder_names)
    return model


def output_graph_name(suffix, as_u32):
    if as_u32:
        return f"model_q4f16_direct_{suffix}_u32.onnx"
    return f"model_q4f16_direct_{suffix}.onnx"


def restore(
    simplified,
    operators,
    initializers,
    placeholder_names,
    model_output_fd,
    suffix,
    as_u32,
):
    model = restore_model(
        simplified, operators, initializers, placeholder_names, as_u32
    )
    name = output_graph_name(suffix, as_u32)
    atomic_write_serialized_message(model_output_fd, name, model)
    return name


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=69)
    parser.add_argument("--dynamic", action="store_true")
    return parser.parse_args(argv)


def build_report(root, source, state, simplified, ort, tvm, args, started):
    return {
        "kind": "q4-direct-specialization",
        "source": str(source.relative_to(root)),
        "onnxGraph": ort,
        "tvmGraph": tvm,
        "inputShape": [1, "sequence_length" if args.dynamic else args.seq_len],
        "quantizedOperators": len(state.operators),
        "rmsnormOperatorsExpanded": state.counts["SimplifiedLayerNormalization"],
        "placeholderCount": len(state.placeholder_names),
        "simplifiedNodeCount": len(simplified.graph.node),
        "denseWeightFilesCreated": 0,
        "checkpointWeightBytesReadByConverter": 0,
        "elapsedSeconds": time.perf_counter() - started,
        "note": "Shape-only graph inputs carry no values. Both outputs reuse the original packed external weights and scales.",
    }


def report_name(args):
    if args.dynamic:
        return "q4-direct-dynamic.json"
    if args.seq_len == 69:
        return "q4-direct-specialization.json"
    return f"q4-direct-specialization-seq{args.seq_len}.json"


def run_specialization(root, args, started):
    folder = root / "models/kev-0.6b-onnx/onnx"
    source = folder / "model_q4f16.onnx"
    model_output_fd = open_directory(root, "models/kev-0.6b-onnx/onnx")
    report_output_fd = None
    try:
        report_output_fd = open_directory(root, "reports")
        original = onnx.load(source, load_external_data=False)
        state = make_shape_only_model(original, folder, args.seq_len, args.dynamic)
        simplified = simplify_shapes(state)
        suffix = "dynamic" if args.dynamic else f"seq{args.seq_len}"
        ort = restore(
            simplified,
            state.operators,
            state.initializers,
            state.placeholder_names,
            model_output_fd,
            suffix,
            False,
        )
        tvm = restore(
            simplified,
            state.operators,
            state.initializers,
            state.placeholder_names,
            model_output_fd,
            suffix,
            True,
        )
        report = build_report(root, source, state, simplified, ort, tvm, args, started)
        text = json.dumps(report, indent=2) + "\n"
        atomic_write(
            report_output_fd,
            report_name(args),
            lambda output: output.write(text.encode()),
        )
        return report
    finally:
        if report_output_fd is not None:
            os.close(report_output_fd)
        os.close(model_output_fd)


def main(argv=None):
    started = time.perf_counter()
    args = parse_arguments(argv)
    if not 1 <= args.seq_len <= 1024:
        raise SystemExit("seq-len must be in 1..1024 for this prototype")
    report = run_specialization(ROOT, args, started)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
