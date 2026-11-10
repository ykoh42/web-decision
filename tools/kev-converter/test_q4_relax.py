"""CPU checks for integrating direct Q4 kernels into Relax graphs."""

import unittest

import numpy as np
import tvm
from q4 import dequantize_matmul_rows, dequantize_rows
from q4_relax import (
    Q4_GPU_EXPERIMENTAL_ROW4_SCHEDULE,
    quantize_weight,
    rewrite_main_graph,
)
from tvm import relax, tirx
from tvm.script import relax as R


@tvm.script.ir_module
class MatmulGraph:
    @R.function
    def main(
        x: R.Tensor((2, 64), dtype="float32"),
        attention_mask: R.Tensor((1,), dtype="int64"),
        p_backbone_layers_0_self_attn_q_proj_weight: R.Tensor((64, 5), dtype="float16"),
    ) -> R.Tensor((2, 5), dtype="float32"):
        weight = R.astype(p_backbone_layers_0_self_attn_q_proj_weight, dtype="float32")
        return R.matmul(x, weight, out_dtype="float32")


@tvm.script.ir_module
class EmbeddingGraph:
    @R.function
    def main(
        input_ids: R.Tensor((3,), dtype="int32"),
        attention_mask: R.Tensor((1,), dtype="int64"),
        p_backbone_embed_tokens_weight: R.Tensor((5, 64), dtype="float16"),
    ) -> R.Tensor((3, 64), dtype="float16"):
        return R.take(p_backbone_embed_tokens_weight, input_ids, axis=0, mode="fast")


seq_len = tirx.Var("seq_len", "int64")


@tvm.script.ir_module
class DynamicMatmulGraph:
    @R.function
    def main(
        x: R.Tensor((1, seq_len, 64), dtype="float16"),
        attention_mask: R.Tensor((1,), dtype="int64"),
        p_backbone_layers_0_self_attn_q_proj_weight: R.Tensor((64, 5), dtype="float16"),
    ) -> R.Tensor((seq_len, 5), dtype="float16"):
        x2 = R.reshape(x, (seq_len, 64))
        return R.matmul(
            x2,
            p_backbone_layers_0_self_attn_q_proj_weight,
            out_dtype="float16",
        )


@tvm.script.ir_module
class DynamicWideMatmulGraph:
    @R.function
    def main(
        x: R.Tensor((1, seq_len, 1024), dtype="float16"),
        attention_mask: R.Tensor((1,), dtype="int64"),
        p_backbone_layers_0_self_attn_q_proj_weight: R.Tensor(
            (1024, 19), dtype="float16"
        ),
    ) -> R.Tensor((seq_len, 19), dtype="float16"):
        x2 = R.reshape(x, (seq_len, 1024))
        return R.matmul(
            x2,
            p_backbone_layers_0_self_attn_q_proj_weight,
            out_dtype="float16",
        )


@tvm.script.ir_module
class DynamicQProjectionGraph:
    @R.function
    def main(
        x: R.Tensor((1, seq_len, 1024), dtype="float16"),
        attention_mask: R.Tensor((1,), dtype="int64"),
        p_backbone_layers_0_self_attn_q_proj_weight: R.Tensor(
            (1024, 3072), dtype="float16"
        ),
    ) -> R.Tensor((seq_len, 3072), dtype="float16"):
        x2 = R.reshape(x, (seq_len, 1024))
        return R.matmul(
            x2,
            p_backbone_layers_0_self_attn_q_proj_weight,
            out_dtype="float16",
        )


@tvm.script.ir_module
class DynamicDownProjectionGraph:
    @R.function
    def main(
        x: R.Tensor((1, seq_len, 3072), dtype="float16"),
        attention_mask: R.Tensor((1,), dtype="int64"),
        p_backbone_layers_0_mlp_down_proj_weight: R.Tensor(
            (3072, 1024), dtype="float16"
        ),
    ) -> R.Tensor((seq_len, 1024), dtype="float16"):
        x2 = R.reshape(x, (seq_len, 3072))
        return R.matmul(
            x2,
            p_backbone_layers_0_mlp_down_proj_weight,
            out_dtype="float16",
        )


@tvm.script.ir_module
class DynamicEmbeddingGraph:
    @R.function
    def main(
        input_ids: R.Tensor((seq_len,), dtype="int32"),
        attention_mask: R.Tensor((1,), dtype="int64"),
        p_backbone_embed_tokens_weight: R.Tensor((5, 64), dtype="float16"),
    ) -> R.Tensor((seq_len, 64), dtype="float16"):
        return R.take(p_backbone_embed_tokens_weight, input_ids, axis=0, mode="fast")


def _dynamic_fixture(graph, name, weight, target="llvm", experimental_schedule=None):
    packed, scales = quantize_weight(name, weight)
    main = graph["main"]
    packed_var = relax.Var("q4_packed", relax.TensorType(packed.shape, "uint32"))
    scales_var = relax.Var("q4_scales", relax.TensorType(scales.shape, "float16"))
    params = [*main.params[:2], packed_var, scales_var]
    rewritten, _, _, _ = rewrite_main_graph(
        graph,
        params,
        {main.params[2].name: (packed_var, scales_var)},
        target=target,
        experimental_schedule=experimental_schedule,
    )
    executable = relax.build(rewritten, target="llvm")
    return relax.VirtualMachine(executable, tvm.cpu()), packed, scales


def _run_dynamic(vm, value, packed, scales):
    return vm["main"](
        tvm.runtime.tensor(value, device=tvm.cpu()),
        tvm.runtime.tensor(np.ones((1,), dtype=np.int64), device=tvm.cpu()),
        tvm.runtime.tensor(packed, device=tvm.cpu()),
        tvm.runtime.tensor(scales, device=tvm.cpu()),
    ).numpy()


class Q4RelaxTests(unittest.TestCase):
    def test_linear_quantization_uses_output_channels_as_rows(self):
        stored = np.zeros((64, 3), dtype=np.float16)
        stored[:, 0] = np.linspace(-0.1, 0.1, 64, dtype=np.float16)
        stored[:, 1] = np.linspace(-2.0, 2.0, 64, dtype=np.float16)
        stored[:, 2] = np.linspace(-0.5, 0.5, 64, dtype=np.float16)

        packed, scales = quantize_weight(
            "backbone.layers.0.self_attn.q_proj.weight", stored
        )

        self.assertEqual(packed.shape, (3, 8))
        self.assertEqual(scales.shape, (3, 2))
        self.assertLess(scales[0, 0], scales[1, 0])
        self.assertLess(scales[2, 0], scales[1, 0])

    def test_real_relax_matmul_is_replaced_with_q4_call_tir(self):
        rng = np.random.default_rng(20261112)
        x_value = rng.normal(0, 0.5, (2, 64)).astype(np.float32)
        source_weight = rng.normal(0, 0.25, (64, 5)).astype(np.float16)
        packed, scales = quantize_weight(
            "backbone.layers.0.self_attn.q_proj.weight", source_weight
        )
        main = MatmulGraph["main"]
        packed_var = relax.Var("q4_packed", relax.TensorType(packed.shape, "uint32"))
        scales_var = relax.Var("q4_scales", relax.TensorType(scales.shape, "float16"))
        params = [*main.params[:2], packed_var, scales_var]
        rewritten, matmul_count, gather_count, replacement_counts = rewrite_main_graph(
            MatmulGraph,
            params,
            {main.params[2].name: (packed_var, scales_var)},
            target="llvm",
        )
        executable = relax.build(rewritten, target="llvm")
        vm = relax.VirtualMachine(executable, tvm.cpu())
        result = vm["main"](
            tvm.runtime.tensor(x_value, device=tvm.cpu()),
            tvm.runtime.tensor(np.ones((1,), dtype=np.int64), device=tvm.cpu()),
            tvm.runtime.tensor(packed, device=tvm.cpu()),
            tvm.runtime.tensor(scales, device=tvm.cpu()),
        ).numpy()
        expected = x_value @ dequantize_matmul_rows(packed, scales).T

        self.assertEqual((matmul_count, gather_count), (1, 0))
        self.assertEqual(replacement_counts, {str(main.params[2].name): 1})
        np.testing.assert_allclose(result, expected, atol=1e-4, rtol=1e-4)

    def test_explicit_webgpu_target_generates_cooperative_q4_reduction(self):
        rng = np.random.default_rng(20261116)
        weight = rng.normal(0, 0.25, (1024, 19)).astype(np.float16)
        packed, scales = quantize_weight(
            "backbone.layers.0.self_attn.q_proj.weight", weight
        )
        main = DynamicWideMatmulGraph["main"]
        packed_var = relax.Var("q4_packed", relax.TensorType(packed.shape, "uint32"))
        scales_var = relax.Var("q4_scales", relax.TensorType(scales.shape, "float16"))
        params = [*main.params[:2], packed_var, scales_var]
        rewritten, matmul_count, gather_count, _ = rewrite_main_graph(
            DynamicWideMatmulGraph,
            params,
            {main.params[2].name: (packed_var, scales_var)},
            target="webgpu",
        )
        self.assertEqual((matmul_count, gather_count), (1, 0))

        prim_func = next(
            function
            for _, function in rewritten.functions_items()
            if isinstance(function, tirx.PrimFunc)
        )
        scheduled_script = prim_func.script()
        self.assertIn('thread="blockIdx.y"', scheduled_script)
        self.assertIn('thread="threadIdx.y"', scheduled_script)
        self.assertIn('thread="threadIdx.x"', scheduled_script)
        self.assertIn("T.axis.reduce(1024", scheduled_script)

        executable = tirx.build(prim_func, target="webgpu")
        webgpu_module = next(
            module for module in executable.imports if module.kind == "webgpu"
        )
        wgsl = webgpu_module.inspect_source("wgsl")
        self.assertIn("@workgroup_size(32, 8, 1)", wgsl)
        self.assertIn("var<workgroup> red_buf", wgsl)
        self.assertIn("workgroupBarrier();", wgsl)
        self.assertIn("q4_k_0 < 32i", wgsl)
        self.assertIn("threadIdx.x) == 0i", wgsl)

    def test_webgpu_q4_schedule_uses_the_target_workgroup_budget(self):
        name = "backbone.layers.0.self_attn.q_proj.weight"
        weight = np.zeros((1024, 3072), dtype=np.float16)
        packed, scales = quantize_weight(name, weight)
        main = DynamicQProjectionGraph["main"]
        packed_var = relax.Var("q4_packed", relax.TensorType(packed.shape, "uint32"))
        scales_var = relax.Var("q4_scales", relax.TensorType(scales.shape, "float16"))
        params = [*main.params[:2], packed_var, scales_var]

        for workgroup_budget, expected_reduction_lanes in (
            (256, 32),
            (512, 64),
            (1024, 128),
        ):
            with self.subTest(workgroup_budget=workgroup_budget):
                target = tvm.target.Target(
                    {"kind": "webgpu", "max_num_threads": workgroup_budget}
                )
                rewritten, matmul_count, gather_count, _ = rewrite_main_graph(
                    DynamicQProjectionGraph,
                    params,
                    {main.params[2].name: (packed_var, scales_var)},
                    target=target,
                )
                self.assertEqual((matmul_count, gather_count), (1, 0))
                prim_func = next(
                    function
                    for _, function in rewritten.functions_items()
                    if isinstance(function, tirx.PrimFunc)
                )
                executable = tirx.build(prim_func, target=target)
                webgpu_module = next(
                    module for module in executable.imports if module.kind == "webgpu"
                )
                wgsl = webgpu_module.inspect_source("wgsl")
                self.assertIn(
                    f"@workgroup_size({expected_reduction_lanes}, 8, 1)", wgsl
                )
                self.assertEqual(expected_reduction_lanes * 8, workgroup_budget)

    def test_experimental_row4_schedule_matches_current_q4_on_cpu(self):
        rng = np.random.default_rng(20261118)
        name = "backbone.layers.0.self_attn.q_proj.weight"
        weight = rng.normal(0, 0.25, (1024, 19)).astype(np.float16)
        current_vm, packed, scales = _dynamic_fixture(
            DynamicWideMatmulGraph, name, weight
        )
        experimental_vm, experimental_packed, experimental_scales = _dynamic_fixture(
            DynamicWideMatmulGraph,
            name,
            weight,
            experimental_schedule=Q4_GPU_EXPERIMENTAL_ROW4_SCHEDULE,
        )
        np.testing.assert_array_equal(experimental_packed, packed)
        np.testing.assert_array_equal(experimental_scales, scales)

        for length in (1, 7, 22):
            values = rng.normal(0, 0.5, (1, length, 1024)).astype(np.float16)
            current = _run_dynamic(current_vm, values, packed, scales)
            experimental = _run_dynamic(
                experimental_vm, values, experimental_packed, experimental_scales
            )
            self.assertEqual(experimental.shape, (length, 19))
            np.testing.assert_allclose(experimental, current, atol=1e-3, rtol=1e-3)

    def test_experimental_row4_schedule_generates_workgroup_and_dispatch_grid(self):
        target = tvm.target.Target({"kind": "webgpu", "max_num_threads": 256})
        graphs = (
            (DynamicQProjectionGraph, 1024, 3072, 384),
            (DynamicDownProjectionGraph, 3072, 1024, 128),
        )
        for graph, width, output_columns, expected_grid_x in graphs:
            packed = np.zeros((output_columns, width // 8), dtype=np.uint32)
            scales = np.zeros((output_columns, width // 32), dtype=np.float16)
            main = graph["main"]
            packed_var = relax.Var(
                "q4_packed", relax.TensorType(packed.shape, "uint32")
            )
            scales_var = relax.Var(
                "q4_scales", relax.TensorType(scales.shape, "float16")
            )
            rewritten, matmul_count, gather_count, _ = rewrite_main_graph(
                graph,
                [*main.params[:2], packed_var, scales_var],
                {main.params[2].name: (packed_var, scales_var)},
                target=target,
                experimental_schedule=Q4_GPU_EXPERIMENTAL_ROW4_SCHEDULE,
            )
            self.assertEqual((matmul_count, gather_count), (1, 0))

            prim_func = next(
                function
                for _, function in rewritten.functions_items()
                if isinstance(function, tirx.PrimFunc)
            )
            scheduled_script = prim_func.script()
            self.assertIn(
                "T.thread_binding((seq_len + T.int64(3)) // T.int64(4), "
                'thread="blockIdx.y")',
                scheduled_script,
            )
            self.assertIn(
                f'T.thread_binding({expected_grid_x}, thread="blockIdx.x")',
                scheduled_script,
            )
            self.assertIn(
                'T.thread_binding(T.int64(4), thread="threadIdx.z")',
                scheduled_script,
            )
            self.assertIn('T.thread_binding(8, thread="threadIdx.y")', scheduled_script)
            self.assertIn('T.thread_binding(8, thread="threadIdx.x")', scheduled_script)

            executable = tirx.build(prim_func, target=target)
            webgpu_module = next(
                module for module in executable.imports if module.kind == "webgpu"
            )
            wgsl = webgpu_module.inspect_source("wgsl")
            self.assertIn("@workgroup_size(8, 8, 4)", wgsl)
            self.assertIn("var<workgroup> red_buf0 : array<f32, 256>;", wgsl)
            self.assertEqual(8 * 8 * 4, 256)
            for length, expected_grid_y in ((22, 6), (8192, 2048)):
                self.assertEqual((length + 3) // 4, expected_grid_y)

    def test_embedding_take_is_replaced_with_q4_call_tir(self):
        rng = np.random.default_rng(20261113)
        ids = np.array([1, 4, 2], dtype=np.int32)
        source_embedding = rng.normal(0, 0.25, (5, 64)).astype(np.float16)
        packed, scales = quantize_weight(
            "backbone.embed_tokens.weight", source_embedding
        )
        main = EmbeddingGraph["main"]
        packed_var = relax.Var(
            "embedding_q4_packed", relax.TensorType(packed.shape, "uint32")
        )
        scales_var = relax.Var(
            "embedding_q4_scales", relax.TensorType(scales.shape, "float16")
        )
        params = [*main.params[:2], packed_var, scales_var]
        rewritten, matmul_count, gather_count, replacement_counts = rewrite_main_graph(
            EmbeddingGraph,
            params,
            {main.params[2].name: (packed_var, scales_var)},
        )
        executable = relax.build(rewritten, target="llvm")
        vm = relax.VirtualMachine(executable, tvm.cpu())
        result = vm["main"](
            tvm.runtime.tensor(ids, device=tvm.cpu()),
            tvm.runtime.tensor(np.ones((1,), dtype=np.int64), device=tvm.cpu()),
            tvm.runtime.tensor(packed, device=tvm.cpu()),
            tvm.runtime.tensor(scales, device=tvm.cpu()),
        ).numpy()
        expected = dequantize_rows(packed, scales)[ids].astype(np.float16)

        self.assertEqual((matmul_count, gather_count), (0, 1))
        self.assertEqual(replacement_counts, {str(main.params[2].name): 1})
        np.testing.assert_array_equal(result, expected)

    def test_dynamic_matmul_uses_runtime_sequence_length(self):
        rng = np.random.default_rng(20261114)
        weight = rng.normal(0, 0.25, (64, 5)).astype(np.float16)
        vm, packed, scales = _dynamic_fixture(
            DynamicMatmulGraph,
            "backbone.layers.0.self_attn.q_proj.weight",
            weight,
        )
        decoded = dequantize_matmul_rows(packed, scales)
        for length in (1, 7, 32768):
            values = rng.normal(0, 0.5, (1, length, 64)).astype(np.float16)
            result = _run_dynamic(vm, values, packed, scales)
            expected = (
                values.reshape(length, 64).astype(np.float32) @ decoded.T
            ).astype(np.float16)
            self.assertEqual(result.shape, (length, 5))
            np.testing.assert_allclose(result, expected, atol=1e-3, rtol=1e-3)

    def test_dynamic_embedding_gather_uses_runtime_sequence_length(self):
        rng = np.random.default_rng(20261115)
        embedding = rng.normal(0, 0.25, (5, 64)).astype(np.float16)
        vm, packed, scales = _dynamic_fixture(
            DynamicEmbeddingGraph, "backbone.embed_tokens.weight", embedding
        )
        decoded = dequantize_rows(packed, scales)
        for length in (1, 7, 32768):
            ids = np.arange(length, dtype=np.int32) % 5
            result = _run_dynamic(vm, ids, packed, scales)
            np.testing.assert_array_equal(result, decoded[ids].astype(np.float16))


if __name__ == "__main__":
    unittest.main()
