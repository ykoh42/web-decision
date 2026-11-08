#!/usr/bin/env python3
"""Keep external q4 weight references intact when atomically saving ONNX."""

import os
import tempfile
import unittest
from pathlib import Path

import onnx
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


if __name__ == "__main__":
    unittest.main()
