"""Exercise bounded eager attention with a one-row final tile."""

import unittest

import numpy as np
import torch
import tvm
from causal_attention_ir import (
    QUERY_HEADS,
    GemmaCausalAttentionTile,
    import_attention_tile,
    query_tiles,
)
from causal_layer_ir import compile_causal_layer
from transformers.models.gemma4.modeling_gemma4 import eager_attention_forward
from tvm import relax


class EagerReference(torch.nn.Module):
    num_key_value_groups = QUERY_HEADS


def make_inputs(dimension, queries=65, keys=513):
    generator = torch.Generator(device="cpu").manual_seed(dimension)
    query = torch.randn(
        (1, QUERY_HEADS, queries, dimension), generator=generator
    ).bfloat16()
    key = torch.randn((1, 1, keys, dimension), generator=generator).bfloat16()
    value = torch.randn((1, 1, keys, dimension), generator=generator).bfloat16()
    query_positions = torch.arange(0, queries).view(-1, 1) + keys - queries
    key_positions = torch.arange(keys).view(1, -1)
    blocked = key_positions > query_positions
    if dimension == 256:
        blocked = blocked | (key_positions <= query_positions - 512)
    mask = torch.zeros((1, 1, queries, keys), dtype=torch.bfloat16)
    mask = mask.masked_fill(
        blocked.view(1, 1, queries, keys), torch.finfo(torch.bfloat16).min
    )
    return query, key, value, mask


def tile_inputs(inputs, interval):
    query, key, value, mask = inputs
    return query[:, :, interval, :], key, value, mask[:, :, interval, :]


def run_tile(vm, inputs):
    values = [tvm.runtime.tensor(value.float().numpy()) for value in inputs]
    return vm["main"](*values).numpy()


class AttentionTileTests(unittest.TestCase):
    def check_tile(self, dimension):
        inputs = make_inputs(dimension)
        examples = tile_inputs(inputs, slice(0, 8))
        module = import_attention_tile(examples)
        cpu = tvm.cpu()
        vm = relax.VirtualMachine(compile_causal_layer(module, "llvm"), cpu)
        intervals = list(query_tiles(inputs[0].shape[2]))
        self.assertEqual(
            [interval.stop - interval.start for interval in intervals], [64, 1]
        )
        actual = np.concatenate(
            [run_tile(vm, tile_inputs(inputs, interval)) for interval in intervals],
            axis=1,
        )
        expected = (
            eager_attention_forward(EagerReference().eval(), *inputs, scaling=1.0)[0]
            .float()
            .numpy()
        )
        torch.testing.assert_close(
            GemmaCausalAttentionTile().eval()(*inputs).float(),
            torch.from_numpy(expected),
            rtol=0,
            atol=0,
        )
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue((actual.view(np.uint32) & np.uint32(0xFFFF) == 0).all())
        changed = inputs[2].clone()
        changed[:, :, -1, :] *= -40
        first_tile = tile_inputs(
            (inputs[0], inputs[1], changed, inputs[3]), intervals[0]
        )
        np.testing.assert_array_equal(run_tile(vm, first_tile), actual[:, :64])
        longest = run_tile(vm, make_inputs(dimension, queries=1, keys=4096))
        self.assertEqual(longest.shape, (1, 1, QUERY_HEADS, dimension))
        self.assertTrue(np.isfinite(longest).all())
        difference = np.abs(actual - expected)
        print(
            f"attention {dimension}, 65 query rows × 513 keys, "
            f"max {difference.max():.8g}, mean {difference.mean():.8g}",
            flush=True,
        )
        target = tvm.target.Target(
            "webgpu", host={"kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"}
        )
        generated = compile_causal_layer(module, target)
        self.assertTrue(
            any(
                part.kind == "webgpu"
                for host in generated.mod.imports
                for part in host.imports
            )
        )

    def test_full_position_capacity_has_a_single_unrepeated_kv_head(self):
        keys = 131072
        module = import_attention_tile(make_inputs(512, 3, 17), max_keys=keys)
        target = tvm.target.Target("webgpu", host="llvm")
        with target, tvm.transform.PassContext(disabled_pass=["FuseOps"]):
            lowered = relax.get_default_pipeline(target)(module)
        self.assertNotIn("broadcast_to", lowered.script())
        vm = relax.VirtualMachine(compile_causal_layer(module, "llvm"), tvm.cpu())
        columns = np.zeros((1, 1, keys, 512), dtype=np.float32)
        key = tvm.runtime.tensor(columns)
        columns[0, 0, -1, :] = 1024
        value = tvm.runtime.tensor(columns)
        del columns
        query = tvm.runtime.tensor(np.zeros((1, 8, 1, 512), dtype=np.float32))
        mask = tvm.runtime.tensor(np.zeros((1, 1, 1, keys), dtype=np.float32))
        output = vm["main"](query, key, value, mask).numpy()
        np.testing.assert_array_equal(output, np.full((1, 1, 8, 512), 0.0078125))

    def test_sliding_and_full_head_shapes(self):
        for dimension in (256, 512):
            with self.subTest(dimension=dimension):
                self.check_tile(dimension)


if __name__ == "__main__":
    unittest.main()
