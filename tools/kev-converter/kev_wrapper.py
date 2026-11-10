"""Torch graph wrapper for the pinned Kev packed-input inference contract."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class KevDelimiters:
    question: int
    option_end: int
    decide: int


def _indicator(condition: torch.Tensor) -> torch.Tensor:
    """Materialize a comparison as an int32 mask for WebGPU portability."""
    return torch.where(
        condition,
        torch.ones_like(condition, dtype=torch.int32),
        torch.zeros_like(condition, dtype=torch.int32),
    )


class TransposedMatMul(nn.Module):
    """Expose a Linear weight as the right-hand MatMul operand for TVM."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight = nn.Parameter(
            linear.weight.detach().T.contiguous(), requires_grad=False
        )
        self.bias = (
            nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
            if linear.bias is not None
            else None
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = torch.matmul(inputs, self.weight)
        return outputs if self.bias is None else outputs + self.bias


def prepare_matmul_weights(module: nn.Module) -> None:
    """Rewrite Linear layers so TVM can quantize right-hand MatMul weights."""
    for name, child in tuple(module.named_children()):
        if isinstance(child, nn.Linear):
            module._modules[name] = TransposedMatMul(child)
        else:
            prepare_matmul_weights(child)


class KevGraph(nn.Module):
    """Convert packed token IDs into Kev's branch mask, positions, and scores."""

    def __init__(
        self, backbone: nn.Module, pointer_head: nn.Module, delimiters: KevDelimiters
    ):
        super().__init__()
        self.backbone = backbone
        self.pointer_head = pointer_head
        self.delimiters = delimiters

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        valid = _indicator(attention_mask != 0)
        length = input_ids.shape[1]
        row_positions = torch.arange(length, dtype=torch.int32, device=input_ids.device)
        positions_in_row = row_positions.unsqueeze(0).expand_as(input_ids)
        causal = _indicator(row_positions[None, :] <= row_positions[:, None])

        question_markers = _indicator(input_ids == self.delimiters.question) * valid
        segments = question_markers.cumsum(dim=1, dtype=torch.int32)
        segments = segments * valid + (valid - 1)
        query_segments = segments[:, :, None]
        key_segments = segments[:, None, :]
        same_branch = torch.maximum(
            _indicator(key_segments == query_segments), _indicator(key_segments == 0)
        )
        question_candidates = same_branch * question_markers[:, None, :]
        branch_starts = (question_candidates * positions_in_row[:, None, :]).sum(
            dim=2, dtype=torch.int32
        )
        first_question = (
            question_markers * _indicator(segments == 1) * positions_in_row
        ).sum(dim=1, keepdim=True, dtype=torch.int32)
        initial_segment = _indicator(segments == 0)
        position_ids = initial_segment * positions_in_row + (1 - initial_segment) * (
            first_question + positions_in_row - branch_starts
        )
        position_ids = valid * position_ids

        allowed = causal[None, :, :] * same_branch * valid[:, None, :]
        dtype = next(self.backbone.parameters()).dtype
        mask = ((1 - allowed) * torch.finfo(dtype).min).to(dtype)[:, None, :, :]

        hidden = self.backbone(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask={"full_attention": mask},
            use_cache=False,
            return_dict=True,
        ).last_hidden_state.float()

        decision_markers = _indicator(input_ids == self.delimiters.decide) * valid
        decision_candidates = same_branch * decision_markers[:, None, :]
        next_decision = (decision_candidates * positions_in_row[:, None, :]).sum(
            dim=2, dtype=torch.int32
        )
        gather_indices = next_decision.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        decision_hidden = hidden.gather(dim=1, index=gather_indices)

        queries = self.pointer_head.q(decision_hidden)
        keys = self.pointer_head.k(hidden)
        scale = self.pointer_head.scale
        scores = (queries * keys).sum(dim=-1) * scale
        option_ends = _indicator(input_ids == self.delimiters.option_end) * valid
        return (option_ends * scores).to(torch.float16)
