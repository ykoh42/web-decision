#!/usr/bin/env python3
"""Keep external q4 weight references intact when atomically saving ONNX."""

import os
import tempfile
import unittest
from pathlib import Path

import onnx
import specialize_q4_direct as specializer
from onnx import TensorProto, helper
from safe_output import atomic_write_serialized_message, open_directory


class DirectQ4SerializationTests(unittest.TestCase):
    def test_preserves_external_weight_reference_without_copying_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "models"
            output_dir.mkdir()
            weights = b"packed q4 bytes"
            weights_path = output_dir / "weights.bin"
            weights_path.write_bytes(weights)

            packed = TensorProto()
            packed.name = "q4_packed_000"
            packed.data_type = TensorProto.UINT8
            packed.dims.append(len(weights))
            packed.data_location = TensorProto.EXTERNAL
            packed.external_data.add(key="location", value="weights.bin")
            packed.external_data.add(key="offset", value="0")
            packed.external_data.add(key="length", value=str(len(weights)))
            graph = helper.make_graph([], "q4", [], [], initializer=[packed])
            model = helper.make_model(graph)

            directory_fd = open_directory(root, "models")
            try:
                atomic_write_serialized_message(directory_fd, "model.onnx", model)
            finally:
                os.close(directory_fd)

            saved = onnx.load_model(output_dir / "model.onnx", load_external_data=False)
            self.assertEqual(len(saved.graph.initializer), 1)
            saved_tensor = saved.graph.initializer[0]
            self.assertEqual(saved_tensor.data_location, TensorProto.EXTERNAL)
            self.assertEqual(
                {entry.key: entry.value for entry in saved_tensor.external_data},
                {"location": "weights.bin", "offset": "0", "length": str(len(weights))},
            )
            self.assertEqual(weights_path.read_bytes(), weights)
            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                ["model.onnx", "weights.bin"],
            )

    def test_restores_matmul_and_embedding_q4_operands_in_role_order(self):
        matmul = helper.make_node(
            "MatMulNBits",
            ["hidden", "packed_matmul", "scale_matmul"],
            ["matmul_output"],
            name="decoder_Q4",
            domain="com.microsoft",
            bits=4,
        )
        gather = helper.make_node(
            "GatherBlockQuantized",
            ["packed_embedding", "indices", "scale_embedding"],
            ["embedding_output"],
            name="embedding_Q4",
            domain="com.microsoft",
            block_size=32,
            gather_axis=0,
            quantize_axis=1,
        )
        expanded = [
            helper.make_node(
                "MatMul",
                ["hidden", "__expanded_weight_0"],
                ["matmul_output"],
                name="decoder_Q4_expanded",
            ),
            helper.make_node(
                "Gather",
                ["__expanded_embedding_1", "indices"],
                ["embedding_output"],
                name="embedding_Q4_expanded",
                axis=0,
            ),
        ]

        nodes, names, rename = specializer.restore_expanded_nodes(
            expanded, {matmul.name: matmul, gather.name: gather}
        )

        self.assertEqual(nodes[0].op_type, "MatMulNBits")
        self.assertEqual(nodes[0].domain, "com.microsoft")
        self.assertEqual(
            list(nodes[0].input), ["hidden", "q4_packed_000", "q4_scale_001"]
        )
        self.assertEqual(nodes[0].name, "q4_MatMulNBits_000")
        self.assertEqual(nodes[1].op_type, "GatherBlockQuantized")
        self.assertEqual(
            list(nodes[1].input),
            ["q4_packed_002", "indices", "q4_scale_003"],
        )
        self.assertEqual(nodes[1].name, "q4_GatherBlockQuantized_001")
        self.assertEqual(
            names,
            {"packed_matmul", "scale_matmul", "packed_embedding", "scale_embedding"},
        )
        self.assertEqual(
            rename,
            {
                "packed_matmul": "q4_packed_000",
                "scale_matmul": "q4_scale_001",
                "packed_embedding": "q4_packed_002",
                "scale_embedding": "q4_scale_003",
            },
        )

    def test_restored_u32_graph_keeps_external_weights_and_removes_placeholders(self):
        source_node = helper.make_node(
            "MatMulNBits",
            ["hidden", "packed", "scale"],
            ["output"],
            name="decoder_Q4",
            domain="com.microsoft",
        )
        expanded_node = helper.make_node(
            "MatMul",
            ["hidden", "__expanded_weight_7"],
            ["output"],
            name="decoder_Q4_expanded",
        )
        packed = external_tensor("packed", TensorProto.INT4, [2, 16], 16)
        scale = external_tensor("scale", TensorProto.FLOAT16, [2], 4)
        simplified = helper.make_model(
            helper.make_graph(
                [expanded_node],
                "q4-shape-only",
                [
                    helper.make_tensor_value_info(
                        "hidden", TensorProto.FLOAT16, [1, 16]
                    ),
                    helper.make_tensor_value_info(
                        "__expanded_weight_7", TensorProto.FLOAT16, [16, 2]
                    ),
                ],
                [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 2])],
            )
        )

        restored = specializer.restore_model(
            simplified,
            {source_node.name: source_node},
            {"packed": packed, "scale": scale},
            {"__expanded_weight_7"},
            as_u32=True,
        )

        self.assertEqual([value.name for value in restored.graph.input], ["hidden"])
        self.assertEqual(restored.graph.node[0].op_type, "MatMulNBits")
        self.assertEqual(
            list(restored.graph.node[0].input),
            ["hidden", "q4_packed_000", "q4_scale_001"],
        )
        tensors = {tensor.name: tensor for tensor in restored.graph.initializer}
        packed_result = tensors["q4_packed_000"]
        self.assertEqual(packed_result.data_type, TensorProto.UINT32)
        self.assertEqual(list(packed_result.dims), [2, 2])
        self.assertEqual(external_data(packed_result), external_data(packed))
        self.assertEqual(tensors["q4_scale_001"].data_type, TensorProto.FLOAT16)
        self.assertEqual(list(tensors["q4_scale_001"].dims), [2])
        self.assertEqual(list(packed.dims), [2, 16])
        self.assertEqual(packed.data_type, TensorProto.INT4)

    def test_u32_packed_view_handles_uint8_and_rejects_bad_external_length(self):
        uint8_tensor = external_tensor(
            "q4_packed_uint8", TensorProto.UINT8, [2, 2, 4], 16
        )
        specializer.convert_packed_initializer_to_u32(uint8_tensor)
        self.assertEqual(uint8_tensor.data_type, TensorProto.UINT32)
        self.assertEqual(list(uint8_tensor.dims), [2, 2])

        invalid = external_tensor("q4_packed_bad", TensorProto.INT4, [2, 16], 14)
        with self.assertRaises(AssertionError):
            specializer.convert_packed_initializer_to_u32(invalid)


def external_tensor(name, data_type, dimensions, length):
    tensor = onnx.TensorProto()
    tensor.name = name
    tensor.data_type = data_type
    tensor.dims.extend(dimensions)
    tensor.data_location = TensorProto.EXTERNAL
    tensor.external_data.add(key="location", value="weights.bin")
    tensor.external_data.add(key="offset", value="0")
    tensor.external_data.add(key="length", value=str(length))
    return tensor


def external_data(tensor):
    return {entry.key: entry.value for entry in tensor.external_data}


if __name__ == "__main__":
    unittest.main()
