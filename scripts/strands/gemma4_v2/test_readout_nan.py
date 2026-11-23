"""CPU regression checks for the host BF16 readout."""

import unittest

import numpy as np
import tvm
from readout_ir import accumulate_readout
from tvm import tirx


class ReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module = tirx.build(
            accumulate_readout.with_attr("global_symbol", "readout"), target="llvm"
        )
        cls.kernel = module["readout"]
        cls.weights = tvm.runtime.tensor(np.full(195840, 0x3F003F00, dtype=np.uint32))

    def run_readout(self, hidden):
        output = tvm.runtime.empty((1, 255), "float32")
        self.kernel(tvm.runtime.tensor(hidden), self.weights, output)
        return output.numpy()

    def test_negative_nan_payloads_do_not_wrap_to_zero(self):
        for bits in (0xFFFF8000, 0xFFFFFFFF, 0x7F800001):
            with self.subTest(bits=hex(bits)):
                hidden = np.zeros((1, 1536), dtype=np.float32)
                hidden.view(np.uint32)[0, 0] = bits
                self.assertTrue(np.isnan(self.run_readout(hidden)).all())

    def test_finite_readout_unchanged(self):
        hidden = np.full((1, 1536), 0.125, dtype=np.float32)
        np.testing.assert_array_equal(
            self.run_readout(hidden), np.full((1, 255), 96, dtype=np.float32)
        )


if __name__ == "__main__":
    unittest.main()
