"""Check upstream rendering and RoPE beyond config bounds without loading model weights.

This checks input validation and rotary arithmetic, not full transformer inference.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

parser = argparse.ArgumentParser()
parser.add_argument("upstream", type=Path)
parser.add_argument("checkpoint", type=Path)
args = parser.parse_args()
sys.path.insert(0, str(args.upstream / "src"))
render = importlib.import_module("strands_decider.core.prompt").render
Request = importlib.import_module("strands_decider.core.request").Request


class CharacterTokenizer:
    encode_special_tokens = True

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return SimpleNamespace(ids=[1000 + value for value in text.encode("utf8")])


tokenizer = CharacterTokenizer()


def request(size):
    return Request.model_validate(
        {
            "state": "x" * size,
            "questions": {"q": {"type": "noul", "instructions": "yes?"}},
        }
    )


base = render(request(1), tokenizer)
overhead = len(base.state_ids) + len(base.questions["q"].question_ids) - 1
for limit in (32768, 131072, 131073):
    for length in (limit - 1, limit, limit + 1):
        try:
            result = render(request(length - overhead), tokenizer, max_length=limit)
            assert (
                len(result.state_ids) + len(result.questions["q"].question_ids)
                == length
            )
            accepted = True
        except ValueError as error:
            assert "exceeds the limit" in str(error)
            accepted = False
        assert accepted == (length <= limit)
        print(json.dumps({"limit": limit, "tokens": length, "accepted": accepted}))

config = transformers.AutoConfig.from_pretrained(str(args.checkpoint)).text_config
assert config.max_position_embeddings == 131072
rotary = Gemma4TextRotaryEmbedding(config)
position_ids = torch.tensor([[131071, 131072]], dtype=torch.long)
for layer_type in ("sliding_attention", "full_attention"):
    cosine, sine = rotary(torch.zeros(1, 2, 1), position_ids, layer_type)
    assert torch.isfinite(cosine).all() and torch.isfinite(sine).all()
    assert cosine.shape[1] == 2
    print(
        json.dumps(
            {
                "layer_type": layer_type,
                "positions": position_ids.tolist()[0],
                "finite": True,
            }
        )
    )
