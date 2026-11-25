"""Run the native TVM lowering against the pinned FP32 pointer head."""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import tvm
from pointer_head_ir import (
    HIDDEN_SIZE,
    MAX_OPTIONS,
    import_pointer_head,
    load_pointer_head,
)
from tvm import relax


class PointerHeadTests(unittest.TestCase):
    def test_rejects_wrong_weight_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            weights = Path(folder) / "head.safetensors"
            weights.write_bytes(b"not the pinned weights")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_pointer_head(weights)

    @unittest.skipUnless(
        os.environ.get("WEB_DECISION_GEMMA_V1_HEAD"),
        "Set WEB_DECISION_GEMMA_V1_HEAD to the pinned local head.safetensors",
    )
    def test_matches_upstream_head_and_compiles_webgpu(self):
        torch.set_num_threads(1)
        head = load_pointer_head(Path(os.environ["WEB_DECISION_GEMMA_V1_HEAD"]))
        module = import_pointer_head(head)
        device = tvm.cpu()
        executable = relax.build(module, target="llvm")
        vm = relax.VirtualMachine(executable, device)
        generator = torch.Generator(device="cpu").manual_seed(983)
        answer = torch.randn((1, HIDDEN_SIZE), generator=generator)
        for count in (2, 3, 8, MAX_OPTIONS):
            with self.subTest(option_count=count):
                options = torch.randn((1, count, HIDDEN_SIZE), generator=generator)
                with torch.no_grad():
                    expected = head(answer, options).numpy()
                actual = vm["main"](
                    tvm.runtime.tensor(answer.numpy(), device=device),
                    tvm.runtime.tensor(options.numpy(), device=device),
                ).numpy()
                np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-6)
                self.assertEqual(int(actual.argmax()), int(expected.argmax()))
        # Compile the browser target without running a WebGPU device.
        target = tvm.target.Target(
            "webgpu", host={"kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"}
        )
        relax.build(module, target=target)


if __name__ == "__main__":
    unittest.main()
