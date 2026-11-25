"""Compare the first v1 Gemma layer with the pinned original PEFT inference."""

import os
import unittest
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
import tvm
from causal_layer_ir import (
    LAYER_INDEX,
    CausalLayerSource,
    compile_causal_layer,
    import_causal_layer,
    verify_files,
)
from safetensors import safe_open
from strands_decider.prompting import render_question, render_state
from strands_decider.schema import ChoiceQuestion
from transformers import AutoTokenizer
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4RMSNorm,
    Gemma4TextRotaryEmbedding,
)
from tvm import relax

BASE = os.environ.get("WEB_DECISION_GEMMA_V1_BASE")
CHECKPOINT = os.environ.get("WEB_DECISION_GEMMA_V1_CHECKPOINT")
TOKENIZER_FILES = {
    "tokenizer.json": "30383f3ce90a169cbabe9eb3da7bb77988bf906aad2e59184b378f5470ec4dbd",
    "tokenizer_config.json": "3ab5c7b94dc97d65ca7064496fa69b88ff875378e1cb7ee3e43070c3a8170999",
}


def source_inputs(config, length, index=LAYER_INDEX):
    random = torch.Generator(device="cpu").manual_seed(429 + length)
    hidden = (
        torch.randn((1, length, config.hidden_size), generator=random) * 0.333
    ).bfloat16()
    per_layer = (
        torch.randn((1, length, config.hidden_size_per_layer_input), generator=random)
        * 0.333
    ).bfloat16()
    return prepare_layer_inputs(config, hidden, per_layer, index)


def prepare_layer_inputs(config, hidden, per_layer, index=LAYER_INDEX):
    positions = torch.arange(hidden.shape[1]).unsqueeze(0)
    rotary = Gemma4TextRotaryEmbedding(config)
    layer_type = config.layer_types[index]
    cosine, sine = rotary(hidden, positions, layer_type=layer_type)
    make_mask = (
        create_causal_mask
        if layer_type == "full_attention"
        else create_sliding_window_causal_mask
    )
    mask = make_mask(
        config=config,
        inputs_embeds=hidden,
        attention_mask=None,
        past_key_values=None,
        position_ids=positions,
    )
    return hidden, per_layer, cosine, sine, mask


def original_prompt_parts(checkpoint):
    verify_files(checkpoint, TOKENIZER_FILES)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True
    ).backend_tokenizer
    state = render_state("The customer was billed twice for the same order.")
    question = render_question(
        ChoiceQuestion(
            instructions="Which team should review this?",
            criteria={"billing": "Payment issues", "support": "Other issues"},
        )
    )
    return (
        tokenizer.encode(state, add_special_tokens=True).ids,
        tokenizer.encode(question.text, add_special_tokens=False).ids,
    )


def read_rows(rows, tokens, width):
    return torch.cat([rows[token : token + 1, :width] for token in tokens]).unsqueeze(0)


def prompt_embeddings(base, config, tokens):
    width = config.hidden_size_per_layer_input
    prefix = "model.language_model."
    with safe_open(base / "model.safetensors", framework="pt", device="cpu") as source:
        hidden = read_rows(
            source.get_slice(prefix + "embed_tokens.weight"), tokens, config.hidden_size
        )
        hidden *= torch.tensor(config.hidden_size**0.5, dtype=torch.bfloat16)
        per_token = read_rows(
            source.get_slice(prefix + "embed_tokens_per_layer.weight"), tokens, width
        )
        per_token *= torch.tensor(width**0.5, dtype=torch.bfloat16)
        projection = source.get_tensor(prefix + "per_layer_model_projection.weight")
        norm = Gemma4RMSNorm(width, eps=config.rms_norm_eps).bfloat16().eval()
        norm.load_state_dict(
            {"weight": source.get_tensor(prefix + "per_layer_projection_norm.weight")}
        )
    projected = (
        torch.nn.functional.linear(hidden, projection) * config.hidden_size**-0.5
    )
    projected = projected.reshape(1, len(tokens), config.num_hidden_layers, width)
    return hidden, (norm(projected)[:, :, LAYER_INDEX, :] + per_token) * 2.0**-0.5


def digest(array):
    return sha256(array.tobytes()).digest()


def assert_webgpu_codegen(test, module):
    target = tvm.target.Target(
        "webgpu", host={"kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"}
    )
    executable = compile_causal_layer(module, target)
    test.assertTrue(
        any(
            part.kind == "webgpu"
            for host in executable.mod.imports
            for part in host.imports
        )
    )


def compile_cpu(module, device):
    return relax.VirtualMachine(compile_causal_layer(module, "llvm"), device)


def call_layer(vm, params, device, inputs):
    tensors = tuple(
        tvm.runtime.tensor(value.float().numpy(), device=device) for value in inputs
    )
    return vm["main"](*tensors, *params)


@unittest.skipUnless(
    BASE and CHECKPOINT, "Set the two pinned local Gemma v1 snapshot paths"
)
class CausalLayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)
        cls.source = CausalLayerSource(Path(BASE), Path(CHECKPOINT))
        cls.config, cls.layer = cls.source.config, cls.source.load(LAYER_INDEX)
        cls.module, cls.parameters = import_causal_layer(
            cls.layer, source_inputs(cls.config, 8)
        )
        cls.cpu = tvm.cpu()
        cls.vm = compile_cpu(cls.module, cls.cpu)

    def run_tvm(self, inputs):
        return call_layer(self.vm, self.parameters, self.cpu, inputs).numpy()

    def test_external_parameters_preserve_source_bits(self):
        expected_base = Counter()
        expected_lora = Counter()
        for weight in self.layer.state_dict().values():
            if weight.dtype == torch.bfloat16:
                halves = weight.view(torch.uint16).numpy().reshape(-1)
                expected_base[digest(np.pad(halves, (0, len(halves) % 2)))] += 1
            else:
                self.assertEqual(weight.dtype, torch.float32)
                expected_lora[digest(weight.numpy())] += 1
        actual_base = Counter(
            digest(weight.numpy())
            for weight in self.parameters
            if str(weight.dtype) == "uint32"
        )
        actual_lora = Counter(
            digest(weight.numpy())
            for weight in self.parameters
            if str(weight.dtype) == "float32"
        )
        self.assertEqual(actual_base, expected_base)
        self.assertEqual(actual_lora, expected_lora)
        self.assertEqual(actual_base.total(), 17)
        self.assertEqual(actual_lora.total(), 14)

    def test_cpu_layer_differential(self):
        for length in (2, 8, 37, 513):
            with self.subTest(length=length):
                inputs = source_inputs(self.config, length)
                actual = self.run_tvm(inputs)
                with torch.no_grad():
                    expected = self.layer(*inputs).float().numpy()
                    with self.layer.layer.disable_adapter():
                        base_only = self.layer(*inputs).float().numpy()
                self.assertEqual(actual.shape, expected.shape)
                self.assertTrue(np.isfinite(actual).all())
                self.assertTrue((actual.view(np.uint32) & np.uint32(0xFFFF) == 0).all())
                difference = np.abs(actual - expected)
                adapter_effect = np.abs(base_only - expected)
                print(
                    f"{length} tokens, max error {difference.max():.8g}, "
                    f"mean error {difference.mean():.8g}, "
                    f"mean error without LoRA {adapter_effect.mean():.8g}",
                    flush=True,
                )
                self.assertLess(difference.mean(), adapter_effect.mean())

    def test_pinned_prompt_layer_input(self):
        state, question = original_prompt_parts(Path(CHECKPOINT))
        with torch.no_grad():
            hidden, per_layer = prompt_embeddings(
                Path(BASE), self.config, state + question
            )
            inputs = prepare_layer_inputs(self.config, hidden, per_layer)
            expected = self.layer(*inputs).float().numpy()
            with self.layer.layer.disable_adapter():
                base_only = self.layer(*inputs).float().numpy()
        actual = self.run_tvm(inputs)
        difference = np.abs(actual - expected)
        print(
            f"{len(state) + len(question)} real prompt tokens, max error {difference.max():.8g}, "
            f"mean error {difference.mean():.8g}",
            flush=True,
        )
        self.assertLess(difference.mean(), np.abs(base_only - expected).mean())

    def test_causal_and_sliding_mask(self):
        inputs = source_inputs(self.config, 513)
        hidden, per_layer, *rest = inputs
        # Position 512 sees position 1 and later.
        mask = inputs[-1]
        self.assertEqual(mask[0, 0, 512, 0], torch.finfo(torch.bfloat16).min)
        self.assertEqual(mask[0, 0, 512, 1], 0)
        actual = self.run_tvm(inputs)
        without_first = hidden.clone()
        without_first[:, 0] *= -4
        sliding = self.run_tvm((without_first, per_layer, *rest))
        np.testing.assert_array_equal(sliding[:, -1], actual[:, -1])

        short = source_inputs(self.config, 8)
        hidden, per_layer, *rest = short
        actual = self.run_tvm(short)
        without_last = hidden.clone()
        without_last[:, -1] *= -4
        causal = self.run_tvm((without_last, per_layer, *rest))
        np.testing.assert_array_equal(causal[:, :-1], actual[:, :-1])

    def test_webgpu_codegen(self):
        assert_webgpu_codegen(self, self.module)

    def check_layer_family(self, index, reference_kv=(), actual_kv=()):
        layer = self.source.load(index)
        inputs = source_inputs(self.config, 8, index)
        module, params = import_causal_layer(layer, inputs + reference_kv)
        self.assertEqual(
            sum(str(weight.dtype) == "float32" for weight in params),
            10 if reference_kv else 14,
        )
        vm = compile_cpu(module, self.cpu)
        with torch.no_grad():
            expected = layer(*(inputs + reference_kv))
        result = call_layer(vm, params, self.cpu, inputs + actual_kv)
        reference_outputs = expected if layer.writes_shared_kv else (expected,)
        actual_outputs = tuple(result) if layer.writes_shared_kv else (result,)
        actual_tensors = []
        for reference, actual in zip(reference_outputs, actual_outputs, strict=True):
            candidate = actual.numpy()
            self.assertEqual(candidate.shape, tuple(reference.shape))
            self.assertTrue(np.isfinite(candidate).all())
            self.assertTrue((candidate.view(np.uint32) & np.uint32(0xFFFF) == 0).all())
            delta = np.abs(candidate - reference.float().numpy())
            print(
                f"layer {index} {tuple(candidate.shape)}, max {delta.max():.8g}, mean {delta.mean():.8g}",
                flush=True,
            )
            actual_tensors.append(torch.from_numpy(candidate).bfloat16())
        assert_webgpu_codegen(self, module)
        return reference_outputs[1:], tuple(actual_tensors[1:])

    def test_nonshared_full_layer_and_shared_kv_families(self):
        self.check_layer_family(4)
        for source, consumer in ((13, 15), (14, 19)):
            with self.subTest(source=source, consumer=consumer):
                reference_kv, actual_kv = self.check_layer_family(source)
                self.assertEqual(len(reference_kv), 2)
                self.check_layer_family(consumer, reference_kv, actual_kv)


if __name__ == "__main__":
    unittest.main()
