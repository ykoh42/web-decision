"""Check that extracted graph construction preserves the existing model IR."""

import unittest

import tvm
from packed_parameters import make_unpack as new_unpack
from probe_dynamic_rope import build as old_rotary
from probe_dynamic_vision_pool import Pool as old_pool
from probe_dynamic_word import build as old_word
from probe_layer0 import make_unpack as old_unpack
from rotary_ir import build as new_rotary
from vision_pool_ir import Pool as new_pool
from word_embedding_ir import build as new_word


class ExtractedIrTest(unittest.TestCase):
    def test_rotary_word_and_pool(self):
        for original, extracted in (
            (old_rotary(), new_rotary()),
            (old_word(), new_word()),
            (old_pool, new_pool),
        ):
            tvm.ir.assert_structural_equal(original, extracted, map_free_vars=True)

    def test_unpack_shapes(self):
        for shape in ((256,), (1536,), (256, 1536), (1536, 6144)):
            tvm.ir.assert_structural_equal(
                old_unpack(shape), new_unpack(shape), map_free_vars=True
            )


if __name__ == "__main__":
    unittest.main()
