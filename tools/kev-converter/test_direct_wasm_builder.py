"""Unit tests for the direct Kev WebGPU build-to-package wiring."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from build_tvm import build_and_package_direct_model, build_webgpu_package
from direct_wasm_packager import LICENSE_PROVENANCE


class DirectWasmBuilderTests(unittest.TestCase):
    def test_compiled_executable_and_official_configs_reach_packager(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = object()
            module = object()
            executable = object()
            parameters = [object(), object()]
            output_dir = root / "run"
            base_dir = root / "official-base"
            kev_source_dir = root / "official-kev"
            tvm = SimpleNamespace(
                relax=SimpleNamespace(build=Mock(return_value=executable))
            )
            expected = object()

            with patch(
                "build_tvm.package_tvm_webgpu_artifacts", return_value=expected
            ) as package:
                result = build_and_package_direct_model(
                    tvm,
                    target,
                    module,
                    parameters,
                    output_dir,
                    base_dir,
                    kev_source_dir,
                )

            self.assertIs(result, expected)
            tvm.relax.build.assert_called_once_with(module, target=target)
            package.assert_called_once_with(
                executable,
                parameters,
                output_dir / "browser-package",
                base_dir / "tokenizer.json",
                base_dir / "tokenizer_config.json",
                base_dir / "config.json",
                kev_source_dir / "LICENSE",
                base_dir / "LICENSE",
            )

    def test_webgpu_build_uses_scheduled_q4_and_packages_its_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "run"
            output_dir.mkdir()
            base_dir = root / "official-base"
            kev_source_dir = root / "official-kev"
            target = object()
            cpu_device = object()
            fp16_module = object()
            webgpu_module = object()
            executable = object()
            parameters = [object() for _ in range(511)]
            names = [f"source_param_{index}" for index in range(314)]
            source_parameters = [object() for _ in range(314)]
            quantization = {"format": "q4"}
            package_dir = output_dir / "browser-package"
            package_dir.mkdir()
            (package_dir / "model.wasm").write_bytes(b"wasm")
            package_result = SimpleNamespace(
                output_dir=package_dir,
                parameter_count=511,
                shard_count=1,
                manifest_sha256="a" * 64,
            )
            tvm = SimpleNamespace(
                target=SimpleNamespace(Target=Mock(return_value=target)),
                relax=SimpleNamespace(build=Mock(return_value=executable)),
                cpu=Mock(return_value=cpu_device),
            )
            integrate_q4 = Mock(return_value=(webgpu_module, parameters, quantization))
            rewritten_module = object()
            attention_report = SimpleNamespace(
                attention_blocks=28,
                metadata_bindings=("lv23", "lv3273", "lv5", "lv59"),
                preprocess_calls=1,
                fused_function="kev_attention_grid",
            )
            rewrite = Mock(return_value=(rewritten_module, attention_report))
            timeline = Mock()
            timeline.attach_mock(rewrite, "rewrite")
            timeline.attach_mock(tvm.relax.build, "build")

            with (
                patch("build_tvm.rewrite_kev_attention", new=rewrite),
                patch(
                    "build_tvm.package_tvm_webgpu_artifacts",
                    return_value=package_result,
                ) as package,
            ):
                result = build_webgpu_package(
                    tvm,
                    integrate_q4,
                    fp16_module,
                    names,
                    source_parameters,
                    quantization,
                    511,
                    output_dir,
                    base_dir,
                    kev_source_dir,
                )

            tvm.target.Target.assert_called_once_with(
                "webgpu",
                host={"kind": "llvm", "mtriple": "wasm32-unknown-unknown"},
            )
            integrate_q4.assert_called_once_with(
                fp16_module,
                names,
                source_parameters,
                cpu_device,
                target=target,
            )
            rewrite.assert_called_once_with(
                webgpu_module,
                question_id=151660,
                decide_id=151661,
                expected_attention_blocks=28,
            )
            tvm.relax.build.assert_called_once_with(rewritten_module, target=target)
            self.assertEqual(timeline.mock_calls[0][0], "rewrite")
            self.assertEqual(timeline.mock_calls[1][0], "build")
            package.assert_called_once_with(
                executable,
                parameters,
                package_dir,
                base_dir / "tokenizer.json",
                base_dir / "tokenizer_config.json",
                base_dir / "config.json",
                kev_source_dir / "LICENSE",
                base_dir / "LICENSE",
            )
            self.assertEqual(
                result,
                {
                    "target": "webgpu",
                    "directory": "browser-package",
                    "parameterCount": 511,
                    "shardCount": 1,
                    "wasmBytes": 4,
                    "manifestSha256": "a" * 64,
                    "license": "Apache-2.0",
                    "licenses": [dict(record) for record in LICENSE_PROVENANCE],
                    "attentionRewrite": {
                        "attentionBlocks": 28,
                        "metadataBindings": ["lv23", "lv3273", "lv5", "lv59"],
                        "preprocessCalls": 1,
                        "fusedFunction": "kev_attention_grid",
                    },
                },
            )

    def test_webgpu_build_rejects_non_pinned_parameter_count(self):
        tvm = SimpleNamespace(
            target=SimpleNamespace(Target=Mock()),
            cpu=Mock(),
        )
        integrate_q4 = Mock()
        with self.assertRaisesRegex(ValueError, "exactly 511"):
            build_webgpu_package(
                tvm,
                integrate_q4,
                object(),
                ["param_0"],
                [object()],
                {},
                1,
                Path("unused-output"),
                Path("unused-base"),
                Path("unused-kev"),
            )
        integrate_q4.assert_not_called()


if __name__ == "__main__":
    unittest.main()
