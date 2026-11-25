"""Compare the complete Gemma v1 TVM decoder with the official BF16 model."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import tvm
from causal_layer_ir import (
    ADAPTER_FILES,
    BASE_FILES,
    MAX_LENGTH,
    CausalLayerSource,
    verify_files,
)
from huggingface_hub import try_to_load_from_cache
from pointer_head_ir import HEAD_SHA256, import_pointer_head, load_pointer_head
from strands_decider.infer import _option_token_index
from strands_decider.modeling import StrandsDeciderModel
from strands_decider.prompting import render_question, render_state
from strands_decider.schema import ChoiceQuestion
from test_causal_layer_ir import BASE, CHECKPOINT, TOKENIZER_FILES
from tiled_causal_decoder import TiledCausalDecoder
from tiled_causal_layer import to_tvm
from tvm import relax

BASE_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
CHECKPOINT_FILES = {
    **TOKENIZER_FILES,
    "head.safetensors": HEAD_SHA256,
    "provenance.json": "214928b49e4ebb43833c345882662ef5c60917c9fafd7e99a1144ee3d9c5582b",
    "strands_decider_config.json": "d323b2b07fdbcb911df601cdf50457ba62fc51c7d58538a4c457b34a290efbde",
}


def official_tokens(model):
    tokenizer = model.tokenizer.backend_tokenizer
    state = render_state("The customer was billed twice for the same order.")
    rendered = render_question(
        ChoiceQuestion(
            instructions="Which team should review this?",
            criteria={"billing": "Payment issues", "support": "Other issues"},
        )
    )
    question = tokenizer.encode(rendered.text, add_special_tokens=False)
    option_positions = _option_token_index(question.offsets, rendered.option_spans, 0)
    return (
        torch.tensor([tokenizer.encode(state, add_special_tokens=True).ids]),
        torch.tensor([question.ids]),
        option_positions,
    )


@torch.inference_mode()
def capture_reference(base, checkpoint, output):
    torch.set_num_threads(2)
    cached_config = try_to_load_from_cache(
        "google/gemma-4-E2B-it", "config.json", revision=BASE_REVISION
    )
    if not isinstance(cached_config, str):
        raise FileNotFoundError("The pinned Gemma base must be in the local Hub cache")
    cached_base = Path(cached_config).parent
    verify_files(cached_base, BASE_FILES)
    if cached_base.resolve() != base.resolve():
        verify_files(base, BASE_FILES)
    verify_files(checkpoint, CHECKPOINT_FILES)
    verify_files(checkpoint / "lora", ADAPTER_FILES)
    model = StrandsDeciderModel.load(
        str(checkpoint), device_map="cpu", attn_implementation="eager"
    ).eval()
    if model.config.base_revision != BASE_REVISION:
        raise ValueError("The official Strands model loaded the wrong Gemma base")
    state, question, options = official_tokens(model)
    tokens = torch.cat((state, question), dim=1)
    mask = torch.ones_like(tokens)
    language_model = model.torso.get_base_model()
    hidden = language_model.embed_tokens(tokens)
    identity = language_model.get_per_layer_inputs(tokens, hidden)
    per_layer = language_model.project_per_layer_inputs(hidden, identity)
    full = model.torso(input_ids=tokens, attention_mask=mask, use_cache=True)
    prefix = model.torso(
        input_ids=state, attention_mask=torch.ones_like(state), use_cache=True
    )
    suffix = model.torso(
        input_ids=question,
        attention_mask=mask,
        past_key_values=prefix.past_key_values,
        use_cache=True,
    )
    cut = state.shape[1]
    full_hidden = full.last_hidden_state.float()
    suffix_hidden = suffix.last_hidden_state.float()
    np.savez(
        output,
        state=state[0].numpy(),
        question=question[0].numpy(),
        options=options,
        hidden=hidden.float().numpy(),
        per_layer=per_layer.float().numpy(),
        full=full_hidden.numpy(),
        prefix=prefix.last_hidden_state.float().numpy(),
        suffix=suffix_hidden.numpy(),
        full_choice=model.head(
            full_hidden[:, -1], full_hidden[:, [cut + index for index in options]]
        ).numpy(),
        suffix_choice=model.head(
            suffix_hidden[:, -1], suffix_hidden[:, options]
        ).numpy(),
    )


@unittest.skipUnless(
    BASE and CHECKPOINT, "Set the two pinned local Gemma v1 snapshot paths"
)
class TiledCausalDecoderTests(unittest.TestCase):
    def test_all_layers_share_kv_and_reuse_official_state(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "reference.npz"
            subprocess.run(
                [sys.executable, __file__, "--reference", BASE, CHECKPOINT, output],
                check=True,
                env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            )
            with np.load(output) as reference:
                self.compare_decoder(reference)

    def compare_decoder(self, reference):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)
        model = TiledCausalDecoder(CausalLayerSource(Path(BASE), Path(CHECKPOINT)))
        with self.assertRaisesRegex(ValueError, "outside the pinned Gemma vocabulary"):
            model([-2])
        with self.assertRaisesRegex(ValueError, "positions must fit"):
            model([0] * (MAX_LENGTH + 1))
        self.assertEqual(len(model.layers), 35)
        full_readers = {19, 24, 29, 34}
        self.assertEqual(
            model.shared_producers,
            {index: 14 if index in full_readers else 13 for index in range(15, 35)},
        )
        print("Imported and built all 35 pinned decoder layers", flush=True)
        state, question = reference["state"].tolist(), reference["question"].tolist()
        cut = len(state)
        with torch.inference_mode():
            hidden, per_layer = model.embeddings(state + question)
        np.testing.assert_array_equal(hidden.float().numpy(), reference["hidden"])
        np.testing.assert_array_equal(per_layer.float().numpy(), reference["per_layer"])
        self.record_delta(
            "official split", reference["full"][:, cut:], reference["suffix"]
        )
        whole, _ = model(state + question)
        self.record_delta("full", whole.float().numpy(), reference["full"])
        prefix, cache = model(state)
        self.record_delta("prefix", prefix.float().numpy(), reference["prefix"])
        before = self.copy_owner_kv(cache)
        suffix, after = model(question, prefix=cache)
        self.record_delta("suffix", suffix.float().numpy(), reference["suffix"])
        first, _ = model(question[:1], prefix=cache)
        repeated, _ = model(question[:1], prefix=cache)
        torch.testing.assert_close(first, repeated, rtol=0, atol=0)
        self.assert_prefix(cache, cut)
        self.assert_prefix(after, cut + len(question))
        for (key, value), layer in zip(before, cache.layers[:15], strict=True):
            self.assertTrue(torch.equal(key, layer.key))
            self.assertTrue(torch.equal(value, layer.value))
        for actual in (whole, prefix, suffix):
            self.assertEqual(actual.dtype, torch.bfloat16)
        self.compare_pointer(whole, suffix, cut, reference)

    def assert_prefix(self, prefix, end):
        self.assertEqual(prefix.end, end)
        for layer in prefix.layers[:15]:
            self.assertEqual((layer.start, layer.end), (0, end))
            self.assertEqual(layer.key.dtype, torch.bfloat16)
            self.assertEqual(layer.value.dtype, torch.bfloat16)
        self.assertEqual(prefix.layers[15:], (None,) * 20)

    def copy_owner_kv(self, prefix):
        return [
            (layer.key.clone(), layer.value.clone()) for layer in prefix.layers[:15]
        ]

    def compare_pointer(self, whole, suffix, cut, reference):
        head = import_pointer_head(
            load_pointer_head(Path(CHECKPOINT) / "head.safetensors")
        )
        vm = relax.VirtualMachine(relax.build(head, target="llvm"), tvm.cpu())
        options = reference["options"]
        for name, hidden, positions in (
            ("full_choice", whole, options + cut),
            ("suffix_choice", suffix, options),
        ):
            actual = vm["main"](
                to_tvm(hidden[:, -1]), to_tvm(hidden[:, positions])
            ).numpy()
            self.record_delta(name, actual, reference[name])
            self.assertEqual(actual.argmax(), reference[name].argmax())
            print(f"{name} logits {actual.tolist()}", flush=True)

    def record_delta(self, stage, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.isfinite(actual).all())
        delta = np.abs(actual - expected)
        print(
            f"full decoder {stage}, max {delta.max():.8g}, mean {delta.mean():.8g}",
            flush=True,
        )


if __name__ == "__main__":
    if sys.argv[1:2] == ["--reference"]:
        capture_reference(*map(Path, sys.argv[2:]))
    else:
        unittest.main()
