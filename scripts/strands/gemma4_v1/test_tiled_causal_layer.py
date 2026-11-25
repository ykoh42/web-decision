"""Compare a tiled v1 layer and a reusable prefix with the original CPU layer."""

import unittest
from copy import deepcopy
from pathlib import Path

import torch
import tvm
from causal_layer_ir import CausalLayerSource, compile_causal_layer
from test_causal_layer_ir import (
    BASE,
    CHECKPOINT,
    original_prompt_parts,
    prompt_embeddings,
    source_inputs,
)
from tiled_causal_layer import (
    LayerKv,
    TiledCausalLayer,
    append_kv,
    import_tiled_layer,
    tile_mask,
)
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding


class PrefixStorageTests(unittest.TestCase):
    def test_window_keeps_compact_bf16_prefix_without_mutating_the_shared_view(self):
        key = torch.arange(513, dtype=torch.float32).bfloat16().view(1, 1, 513, 1)
        active = LayerKv(key, -key, 513)
        prefix = active.compact(512)
        next_key = torch.ones((1, 1, 1, 1), dtype=torch.bfloat16)
        combined = append_kv(prefix, next_key, next_key)
        self.assertEqual((prefix.start, prefix.end), (2, 513))
        self.assertEqual(prefix.key.untyped_storage().nbytes(), 511 * 2)
        self.assertEqual((combined.start, combined.end), (2, 514))
        torch.testing.assert_close(active.key, key, rtol=0, atol=0)
        self.assertEqual(prefix.end, 513)


@unittest.skipUnless(
    BASE and CHECKPOINT, "Set the two pinned local Gemma v1 snapshot paths"
)
class TiledCausalLayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)
        cls.source = CausalLayerSource(Path(BASE), Path(CHECKPOINT))
        cls.layer = cls.source.load(0)
        cls.rotary = Gemma4TextRotaryEmbedding(cls.source.config)
        state, question = original_prompt_parts(Path(CHECKPOINT))
        cls.prefix_length = len(state)
        with torch.no_grad():
            cls.hidden, cls.per_layer = prompt_embeddings(
                Path(BASE), cls.source.config, state + question
            )
        cls.programs = cls.import_programs(
            cls.layer, cls.hidden[:, :8], cls.per_layer[:, :8]
        )
        cls.tiled = TiledCausalLayer(cls.layer, cls.programs)

    @classmethod
    def import_programs(cls, layer, hidden, per_layer):
        positions = torch.arange(hidden.shape[1]).unsqueeze(0)
        cosine, sine = cls.rotary(hidden, positions, layer_type=layer.layer_type)
        return import_tiled_layer(layer, (hidden, per_layer, cosine, sine))

    def upstream(self, hidden, per_layer, cache, layer=None, shared=None):
        layer = self.layer if layer is None else layer
        index = layer.layer.get_base_model().layer_idx
        start = cache.get_seq_length(index)
        positions = torch.arange(start, start + hidden.shape[1]).unsqueeze(0)
        mask_func = (
            create_causal_mask
            if layer.layer_type == "full_attention"
            else create_sliding_window_causal_mask
        )
        mask = mask_func(
            config=self.source.config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=cache,
            position_ids=positions,
            layer_idx=index,
        )
        output = self.forward_upstream(
            layer, hidden, per_layer, positions, mask, cache, shared
        )
        return output, mask

    def forward_upstream(
        self, layer, hidden, per_layer, positions, mask, cache=None, shared=None
    ):
        cosine, sine = self.rotary(hidden, positions, layer_type=layer.layer_type)
        with torch.no_grad():
            return layer.layer(
                hidden,
                per_layer,
                position_embeddings=(cosine, sine),
                attention_mask=mask,
                shared_kv_states={} if shared is None else shared,
                position_ids=positions,
                past_key_values=cache,
            )

    def record_delta(self, stage, expected, actual):
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(actual.isfinite().all())
        delta = (expected.float() - actual.float()).abs()
        print(f"{stage}, max {delta.max():.8g}, mean {delta.mean():.8g}", flush=True)

    def test_pinned_prompt_prefill_and_continuation(self):
        reference_cache = DynamicCache(config=self.source.config)
        expected, mask = self.upstream(self.hidden, self.per_layer, reference_cache)
        actual, _, _ = self.tiled(self.hidden, self.per_layer)
        self.record_delta("whole prompt", expected, actual)
        torch.testing.assert_close(
            tile_mask(0, 64, 0, self.hidden.shape[1], 512),
            mask[:, :, :64],
            rtol=0,
            atol=0,
        )

        cut = self.prefix_length
        cached = DynamicCache(config=self.source.config)
        expected_prefix, _ = self.upstream(
            self.hidden[:, :cut], self.per_layer[:, :cut], cached
        )
        actual_prefix, prefix, _ = self.tiled(
            self.hidden[:, :cut], self.per_layer[:, :cut]
        )
        self.record_delta("state prefix", expected_prefix, actual_prefix)
        self.assertEqual(prefix.end, cut)
        key_before = prefix.key.clone()
        value_before = prefix.value.clone()

        suffix = (self.hidden[:, cut:], self.per_layer[:, cut:])
        expected_suffix, mask = self.upstream(*suffix, deepcopy(cached))
        actual_suffix, after, _ = self.tiled(*suffix, prefix=prefix)
        self.record_delta("question continuation", expected_suffix, actual_suffix)
        torch.testing.assert_close(
            tile_mask(cut, suffix[0].shape[1], 0, after.end, 512), mask, rtol=0, atol=0
        )
        repeated, _, _ = self.tiled(*suffix, prefix=prefix)
        torch.testing.assert_close(repeated, actual_suffix, rtol=0, atol=0)
        torch.testing.assert_close(prefix.key, key_before, rtol=0, atol=0)
        torch.testing.assert_close(prefix.value, value_before, rtol=0, atol=0)

    def test_single_final_token_uses_the_same_prefix_layout(self):
        before = (self.hidden[:, :-1], self.per_layer[:, :-1])
        last = (self.hidden[:, -1:], self.per_layer[:, -1:])
        cache = DynamicCache(config=self.source.config)
        self.upstream(*before, cache)
        expected, mask = self.upstream(*last, cache)
        _, prefix, _ = self.tiled(*before)
        actual, after, _ = self.tiled(*last, prefix=prefix)
        self.record_delta("single-token continuation", expected, actual)
        torch.testing.assert_close(
            tile_mask(prefix.end, 1, 0, after.end, 512), mask, rtol=0, atol=0
        )

    def test_sliding_mask_after_the_prefix_is_compacted(self):
        cache = DynamicCache(config=self.source.config)
        key = torch.zeros((1, 1, 513, 256), dtype=torch.bfloat16)
        cache.update(key, key, 0)
        hidden, _, *_ = source_inputs(self.source.config, 2)
        original = create_sliding_window_causal_mask(
            config=self.source.config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=cache,
            position_ids=torch.arange(513, 515).unsqueeze(0),
            layer_idx=0,
        )
        prefix = LayerKv(key, key, 513).compact(512)
        torch.testing.assert_close(
            tile_mask(513, 2, prefix.start, 513, 512), original, rtol=0, atol=0
        )

    def check_shared_continuation(self, producer_index, consumer_index):
        producer, consumer = (
            self.source.load(producer_index),
            self.source.load(consumer_index),
        )
        hidden, per_layer, *_ = source_inputs(self.source.config, 8, producer_index)
        prefix_args = (hidden[:, :4], per_layer[:, :4])
        suffix_args = (hidden[:, 4:], per_layer[:, 4:])
        reference_cache = DynamicCache(config=self.source.config)
        reference_shared = {}
        self.upstream(*prefix_args, reference_cache, producer, reference_shared)
        expected, mask = self.upstream(
            *suffix_args, reference_cache, producer, reference_shared
        )

        source = TiledCausalLayer(
            producer, self.import_programs(producer, hidden, per_layer)
        )
        _, prefix, _ = source(*prefix_args)
        actual, _, active = source(*suffix_args, prefix=prefix)
        self.record_delta(f"layer {producer_index} continuation", expected, actual)
        self.assertEqual((active.start, active.end), (0, 8))
        key, value = reference_shared[producer.layer_type]
        self.record_delta(f"layer {producer_index} shared key", key, active.key)
        self.record_delta(f"layer {producer_index} shared value", value, active.value)

        next_layer = TiledCausalLayer(
            consumer, self.import_programs(consumer, hidden, per_layer)
        )
        result, consumer_prefix, consumer_output = next_layer(
            *suffix_args, shared=active
        )
        self.assertIsNone(consumer_prefix)
        self.assertIsNone(consumer_output)
        expected = self.forward_upstream(
            consumer,
            *suffix_args,
            torch.arange(4, 8).unsqueeze(0),
            mask,
            shared=reference_shared,
        )
        self.record_delta(
            f"layer {consumer_index} shared continuation", expected, result
        )

    def test_shared_sliding_and_full_continuations(self):
        for source, consumer in ((13, 15), (14, 19)):
            with self.subTest(source=source, consumer=consumer):
                self.check_shared_continuation(source, consumer)

    def test_webgpu_codegen_for_all_three_stages(self):
        target = tvm.target.Target(
            "webgpu", host={"kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"}
        )
        for module, _ in self.programs:
            result = compile_causal_layer(module, target)
            self.assertTrue(
                any(
                    part.kind == "webgpu"
                    for host in result.mod.imports
                    for part in host.imports
                )
            )


if __name__ == "__main__":
    unittest.main()
