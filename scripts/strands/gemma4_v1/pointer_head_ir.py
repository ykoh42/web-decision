"""Import the pinned Gemma 4 E2B v1 pointer head into TVM Relax."""

from hashlib import sha256
from pathlib import Path

import torch
from safetensors.torch import load
from strands_decider.modeling import PointerHead
from tvm.relax.frontend.torch import from_exported_program

HEAD_SHA256 = "31220ff165a8bb6e48b87a85de11cbfc8660b7047049ea2f91872e1b6ddb45a9"
HEAD_BYTES = 9_456_104
HIDDEN_SIZE = 1536
POINTER_SIZE = 768
MAX_OPTIONS = 255


def load_pointer_head(weights: Path) -> PointerHead:
    # Load the exact bytes that passed the pin check.
    with weights.open("rb") as source:
        checkpoint = source.read(HEAD_BYTES + 1)
    if len(checkpoint) != HEAD_BYTES or sha256(checkpoint).hexdigest() != HEAD_SHA256:
        raise ValueError("Gemma 4 E2B v1 pointer-head SHA-256 mismatch")
    head = PointerHead(HIDDEN_SIZE, POINTER_SIZE, dropout=0.05).float().eval()
    head.load_state_dict(load(checkpoint))
    return head


def import_pointer_head(head: PointerHead):
    answer = torch.zeros((1, HIDDEN_SIZE), dtype=torch.float32, device="cpu")
    options = torch.zeros((1, 8, HIDDEN_SIZE), dtype=torch.float32, device="cpu")
    program = torch.export.export(
        head,
        (answer, options),
        dynamic_shapes=(
            None,
            {1: torch.export.Dim("option_count", min=2, max=MAX_OPTIONS)},
        ),
        strict=False,
    )
    return from_exported_program(program, unwrap_unit_return_tuple=True)
