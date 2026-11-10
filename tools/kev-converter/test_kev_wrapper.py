"""Checks the direct graph wrapper's packed-branch contract."""

import unittest
from types import SimpleNamespace

import torch
from kev_wrapper import KevDelimiters, KevGraph, prepare_matmul_weights


class CapturingBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones((), dtype=torch.float32))

    def forward(self, input_ids, position_ids, attention_mask, **_kwargs):
        self.input_ids_dtype = input_ids.dtype
        self.position_ids = position_ids.detach().clone()
        self.full_attention = attention_mask["full_attention"].detach().clone()
        ids = input_ids.to(torch.float32) * self.anchor
        positions = position_ids.to(torch.float32) * self.anchor
        return SimpleNamespace(last_hidden_state=torch.stack((ids, positions), dim=-1))


class IdentityPointerHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Identity()
        self.k = torch.nn.Identity()
        self.scale = 1.0


class KevWrapperTests(unittest.TestCase):
    def setUp(self):
        self.backbone = CapturingBackbone()
        self.graph = KevGraph(
            self.backbone,
            IdentityPointerHead(),
            KevDelimiters(question=9, option_end=7, decide=8),
        )

    def test_multiple_branches_and_right_padding(self):
        input_ids = torch.tensor(
            [
                [100, 101, 9, 11, 20, 7, 21, 7, 8, 9, 12, 22, 7, 23, 7, 8],
                [100, 9, 13, 30, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.int32,
        )
        attention_mask = torch.tensor(
            [[1] * 16, [1, 1, 1, 1, 1, 1, *([0] * 10)]], dtype=torch.int32
        )

        scores = self.graph(input_ids, attention_mask)

        self.assertEqual(scores.dtype, torch.float16)
        self.assertEqual(
            self.backbone.position_ids.tolist(),
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 4, 5, 6, 7, 8],
                [0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        )
        self.assertEqual(self.backbone.input_ids_dtype, torch.int32)
        self.assertEqual(self.backbone.position_ids.dtype, torch.int32)
        allowed = self.backbone.full_attention[:, 0].eq(0)
        self.assertEqual(torch.where(allowed[0, 5])[0].tolist(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(torch.where(allowed[0, 9])[0].tolist(), [0, 1, 9])
        self.assertEqual(
            torch.where(allowed[0, 15])[0].tolist(), [0, 1, 9, 10, 11, 12, 13, 14, 15]
        )
        self.assertEqual(torch.where(allowed[1, 4])[0].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(torch.where(allowed[1, 6])[0].tolist(), [0])
        self.assertFalse(allowed[1, 6:, 1:].any().item())
        self.assertEqual(scores[0, [5, 7, 12, 14]].tolist(), [96.0, 112.0, 96.0, 112.0])
        self.assertEqual(scores[1, 4].item(), 76.0)
        self.assertEqual(scores[1, 6:].tolist(), [0.0] * 10)

    def test_linear_rewrite_preserves_batched_values_with_or_without_bias(self):
        for bias in (False, True):
            with self.subTest(bias=bias):
                original = torch.nn.Linear(4, 3, bias=bias)
                inputs = torch.randn(2, 5, 4)
                expected = original(inputs)
                rewritten = torch.nn.Sequential(original)

                prepare_matmul_weights(rewritten)

                torch.testing.assert_close(rewritten(inputs), expected)


if __name__ == "__main__":
    unittest.main()
