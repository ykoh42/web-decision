"""Public Relax rewrite tests for the full Kev attention graph."""

import ast
import unittest
from pathlib import Path

import tvm
from kev_attention_relax import rewrite_kev_attention
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T


def _attention_graph_source(block_count):
    """Generate a dynamic symbolic fixture with the pinned Qwen3 node chain."""
    blocks = []
    outputs = []
    for index in range(block_count):
        blocks.extend(
            (
                f'            qk_{index} = R.matmul(query, key, out_dtype="float32")',
                f"            qk4_{index} = R.reshape(qk_{index}, (1, 16, sequence_length, sequence_length))",
                f'            scaled_{index} = R.multiply(qk4_{index}, R.const(0.088388346135616302, "float32"))',
                f'            added_{index} = R.add(scaled_{index}, R.astype(mask, dtype="float32"))',
                f'            scores_{index} = R.astype(added_{index}, dtype="float32")',
                f"            softmax_{index} = R.nn.softmax(scores_{index}, axis=-1)",
                f'            probabilities16_{index} = R.astype(softmax_{index}, dtype="float16")',
                f"            broadcast_{index} = R.broadcast_to(probabilities16_{index}, (1, 16, sequence_length, sequence_length))",
                f"            flattened_{index} = R.reshape(broadcast_{index}, (16, sequence_length, sequence_length))",
                f'            probabilities32_{index} = R.astype(flattened_{index}, dtype="float32")',
                f'            output_{index} = R.matmul(probabilities32_{index}, value, out_dtype="float32")',
            )
        )
        outputs.append(f"output_{index}")
    outputs.extend(("lv5", "lv23", "lv59", "lv3273"))
    result = ", ".join(outputs)
    return f"""\
@I.ir_module
class DenseAttentionFixture:
    @R.function
    def main[sequence_length](
        input_ids: R.Tensor((1, sequence_length), "int32"),
        attention_mask: R.Tensor((1, sequence_length), "int32"),
        query: R.Tensor((16, sequence_length, 128), "float16"),
        key: R.Tensor((16, 128, sequence_length), "float16"),
        value: R.Tensor((16, sequence_length, 128), "float32"),
    ):
        with R.dataflow():
            lv5 = R.astype(attention_mask, dtype="int32")
            lv23 = R.full((1, sequence_length), R.const(0, "int32"), dtype="int32")
            lv59 = R.full((1, sequence_length), R.const(0, "int32"), dtype="int32")
            lv3273 = R.full((1, sequence_length), R.const(0, "int32"), dtype="int32")
            mask = R.full((1, 1, sequence_length, sequence_length), R.const(-65504, "float16"), dtype="float16")
{chr(10).join(blocks)}
            result = ({result})
            R.output(result)
        return result
"""


def _attention_graph(block_count):
    """Parse a pinned-shape Relax fixture with a symbolic sequence dimension."""
    source = _attention_graph_source(block_count)
    return tvm.script.from_source(
        source,
        extra_vars={"I": I, "R": R, "T": T},
        check_well_formed=False,
    )


class KevAttentionRelaxTests(unittest.TestCase):
    def test_rewriter_does_not_access_the_private_builder_field(self):
        source_path = Path(__file__).with_name("kev_attention_relax.py")
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        private_builder_accesses = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "builder_"
        ]
        self.assertEqual(private_builder_accesses, [])

    def test_all_28_blocks_are_fused_and_square_graph_is_dead(self):
        original = _attention_graph(28)
        expected_return_type = tvm.relax.transform.Normalize()(original)["main"].ret_ty
        rewritten, report = rewrite_kev_attention(
            original,
            question_id=151660,
            decide_id=151661,
        )
        script = rewritten.script()
        tvm.ir.assert_structural_equal(expected_return_type, rewritten["main"].ret_ty)
        self.assertEqual(report.attention_blocks, 28)
        self.assertEqual(
            report.metadata_bindings,
            ("lv23", "lv3273", "lv5", "lv59"),
        )
        self.assertEqual(report.preprocess_calls, 1)
        self.assertEqual(script.count("R.call_tir(cls.kev_attention_grid"), 28)
        self.assertEqual(script.count("R.call_tir(cls.kev_preprocess"), 1)
        self.assertNotIn("R.nn.softmax", script)
        self.assertNotIn("R.matmul(", script)
        self.assertNotRegex(
            script,
            r"(?:Tensor|Buffer)\(\([^)]*sequence_length\s*,\s*sequence_length",
        )

    def test_rewriter_preserves_a_dynamic_sequence_type(self):
        rewritten, _ = rewrite_kev_attention(
            _attention_graph(1),
            question_id=151660,
            decide_id=151661,
            expected_attention_blocks=1,
        )
        script = rewritten.script()
        self.assertIn("def main[sequence_length]", script)
        self.assertIn("sequence_length", script)
        self.assertIn('R.Tensor((16, sequence_length, 128), dtype="float32")', script)

    def test_replacement_count_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "replacement count mismatch"):
            rewrite_kev_attention(
                _attention_graph(1),
                question_id=151660,
                decide_id=151661,
                expected_attention_blocks=28,
            )


if __name__ == "__main__":
    unittest.main()
