"""Unit tests for the isolated direct TVM WebGPU packager."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import direct_wasm_packager as packager


class FakeExecutable:
    def __init__(self):
        self.exports = []

    def export_library(self, path, *, fcompile):
        self.exports.append((path, fcompile))
        Path(path).write_bytes(b"\0asm")


class FakeTvmjs:
    create_tvmjs_wasm = object()

    def __init__(
        self,
        *,
        omit_shard=False,
        drop_last_param=False,
        metadata_count=None,
        shard_size=None,
        record_updates=None,
    ):
        self.omit_shard = omit_shard
        self.drop_last_param = drop_last_param
        self.metadata_count = metadata_count
        self.shard_size = shard_size
        self.record_updates = record_updates or {}
        self.dump_call = None

    def dump_tensor_cache(self, named_params, path, **kwargs):
        params = list(named_params)
        self.dump_call = (params, path, kwargs)
        cache = Path(path)
        cache.mkdir()
        records = [
            {
                "name": name,
                "shape": [1],
                "dtype": "uint32",
                "format": "raw",
                "nbytes": 4,
                "byteOffset": index * 4,
            }
            for index, (name, _value) in enumerate(params)
        ]
        if self.drop_last_param:
            records = records[:-1]
        if records and self.record_updates:
            records[0].update(self.record_updates)
        shard_bytes = b"x" * (len(params) * 4)
        if not self.omit_shard:
            if params:
                (cache / "params_shard_0.bin").write_bytes(shard_bytes)
        manifest = {
            "metadata": {
                "numParams": (
                    kwargs["meta_data"]["numParams"]
                    if self.metadata_count is None
                    else self.metadata_count
                )
            },
            "records": (
                [
                    {
                        "dataPath": "params_shard_0.bin",
                        "format": "raw-shard",
                        "nbytes": (
                            len(shard_bytes)
                            if self.shard_size is None
                            else self.shard_size
                        ),
                        "records": records,
                    }
                ]
                if params
                else []
            ),
        }
        (cache / "tensor-cache.json").write_text(json.dumps(manifest), encoding="utf-8")


class DirectWasmPackagerTests(unittest.TestCase):
    def _inputs(self, root):
        sources = []
        for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
            path = root / name
            value = (
                {"model_type": "qwen3", "max_position_embeddings": 32768}
                if name == "config.json"
                else {"file": name}
            )
            path.write_text(json.dumps(value), encoding="utf-8")
            sources.append(path)
        license_text = (
            "                                 Apache License\n"
            "                           Version 2.0, January 2004\n"
        )
        for name in ("kev-LICENSE", "qwen-LICENSE"):
            (root / name).write_text(license_text, encoding="utf-8")
        return sources

    def _package(
        self,
        executable,
        runtime_params,
        output,
        tokenizer_json,
        tokenizer_config,
        model_config,
    ):
        input_dir = Path(tokenizer_json).parent
        return packager.package_tvm_webgpu_artifacts(
            executable,
            runtime_params,
            output,
            tokenizer_json,
            tokenizer_config,
            model_config,
            input_dir / "kev-LICENSE",
            input_dir / "qwen-LICENSE",
        )

    def test_writes_wasm_configs_and_verified_raw_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer, tokenizer_config, model_config = self._inputs(root)
            output = root / "model"
            executable = FakeExecutable()
            tvmjs = FakeTvmjs()
            params = [object(), object()]

            with patch.object(
                packager, "_load_tvmjs", return_value=(tvmjs, "0.27.0.post1")
            ):
                result = self._package(
                    executable,
                    params,
                    output,
                    tokenizer,
                    tokenizer_config,
                    model_config,
                )

            self.assertEqual(result.output_dir, output)
            self.assertEqual(result.parameter_count, 2)
            self.assertEqual(result.shard_count, 1)
            self.assertEqual(len(result.manifest_sha256), 64)
            self.assertEqual(
                executable.exports,
                [(str(output / "model.wasm"), tvmjs.create_tvmjs_wasm)],
            )
            names, cache_path, options = tvmjs.dump_call
            self.assertEqual(names, [("param_0", params[0]), ("param_1", params[1])])
            self.assertEqual(cache_path, str(output / "weights"))
            self.assertEqual(options["encode_format"], "raw")
            self.assertEqual(options["meta_data"], {"numParams": 2})
            self.assertEqual(options["shard_cap_mb"], 128)
            self.assertEqual((output / "model.wasm").read_bytes(), b"\0asm")
            self.assertEqual(
                (output / "tokenizer.json").read_bytes(), tokenizer.read_bytes()
            )
            self.assertEqual(
                (output / "tokenizer_config.json").read_bytes(),
                tokenizer_config.read_bytes(),
            )
            license_sources = {
                "jaredpalmer/kev": root / "kev-LICENSE",
                "Qwen/Qwen3-0.6B-Base": root / "qwen-LICENSE",
            }
            for record in packager.LICENSE_PROVENANCE:
                packaged_license = output / record["file"]
                source_license = license_sources[record["repository"]]
                self.assertEqual(
                    packaged_license.read_bytes(), source_license.read_bytes()
                )
            self.assertEqual(
                json.loads((output / "model_config.json").read_text()),
                {
                    "model_type": "qwen3",
                    "max_position_embeddings": 32768,
                    "kev": {
                        "delimiter_ids": packager.KEV_DELIMITER_IDS,
                        "max_state_tokens": 8192,
                        "max_branch_tokens": 8192,
                        "max_options": 255,
                    },
                },
            )
            manifest_bytes = (output / "manifest.json").read_bytes()
            manifest = json.loads(manifest_bytes)
            canonical = json.dumps(
                manifest,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            self.assertEqual(
                result.manifest_sha256,
                hashlib.sha256(f"{canonical}\n".encode("utf-8")).hexdigest(),
            )
            self.assertEqual(manifest["model"], "kev-0.6b-q4f16")
            self.assertEqual(manifest["backend"], "tvm-webgpu")
            self.assertEqual(manifest["parameterCount"], 2)
            self.assertEqual(manifest["maxStateTokens"], 8192)
            self.assertEqual(manifest["maxBranchTokens"], 8192)
            self.assertEqual(manifest["license"], "Apache-2.0")
            self.assertEqual(
                manifest["licenses"],
                [dict(record) for record in packager.LICENSE_PROVENANCE],
            )
            self.assertEqual(
                manifest["checkpoint"]["revision"], packager.CHECKPOINT["revision"]
            )
            self.assertEqual(
                manifest["base"]["revision"], packager.BASE_MODEL["revision"]
            )
            self.assertNotIn("maxInputTokens", manifest)
            self.assertNotIn("manifest.json", manifest["files"])
            for name, expected in manifest["files"].items():
                asset = output / name
                self.assertEqual(asset.stat().st_size, expected["bytes"])
                self.assertEqual(
                    hashlib.sha256(asset.read_bytes()).hexdigest(), expected["sha256"]
                )

    def test_non_apache_license_source_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer, tokenizer_config, model_config = self._inputs(root)
            (root / "kev-LICENSE").write_text("unverified license", encoding="utf-8")
            output = root / "model"
            tvmjs = FakeTvmjs()

            with patch.object(
                packager, "_load_tvmjs", return_value=(tvmjs, "0.27.0.post1")
            ) as load_tvmjs:
                with self.assertRaisesRegex(ValueError, "Apache-2.0"):
                    self._package(
                        FakeExecutable(),
                        [],
                        output,
                        tokenizer,
                        tokenizer_config,
                        model_config,
                    )

            load_tvmjs.assert_not_called()
            self.assertFalse(output.exists())

    def test_refuses_existing_output_without_touching_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer, tokenizer_config, model_config = self._inputs(root)
            output = root / "model"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("preserve")

            with patch.object(packager, "_load_tvmjs") as load_tvmjs:
                with self.assertRaises(FileExistsError):
                    self._package(
                        FakeExecutable(),
                        [],
                        output,
                        tokenizer,
                        tokenizer_config,
                        model_config,
                    )

            load_tvmjs.assert_not_called()
            self.assertEqual(sentinel.read_text(), "preserve")

    def test_refuses_symlinked_output_without_touching_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer, tokenizer_config, model_config = self._inputs(root)
            target = root / "target"
            target.mkdir()
            sentinel = target / "keep.txt"
            sentinel.write_text("preserve")
            output = root / "model-link"
            output.symlink_to(target, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "symlink"):
                self._package(
                    FakeExecutable(),
                    [],
                    output,
                    tokenizer,
                    tokenizer_config,
                    model_config,
                )

            self.assertEqual(sentinel.read_text(), "preserve")

    def test_refuses_symlinked_parent_component(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer, tokenizer_config, model_config = self._inputs(root)
            target = root / "real-parent"
            target.mkdir()
            (root / "linked-parent").symlink_to(target, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "symlink"):
                self._package(
                    FakeExecutable(),
                    [],
                    root / "linked-parent" / "model",
                    tokenizer,
                    tokenizer_config,
                    model_config,
                )

    def test_parameter_count_mismatch_removes_partial_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer, tokenizer_config, model_config = self._inputs(root)
            output = root / "model"
            tvmjs = FakeTvmjs(drop_last_param=True)

            with patch.object(
                packager, "_load_tvmjs", return_value=(tvmjs, "0.27.0.post1")
            ):
                with self.assertRaisesRegex(ValueError, "names or order"):
                    self._package(
                        FakeExecutable(),
                        [object(), object()],
                        output,
                        tokenizer,
                        tokenizer_config,
                        model_config,
                    )

            self.assertFalse(output.exists())

    def test_missing_manifest_shard_is_rejected_and_cleaned_up(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer, tokenizer_config, model_config = self._inputs(root)
            output = root / "model"
            tvmjs = FakeTvmjs(omit_shard=True)

            with patch.object(
                packager, "_load_tvmjs", return_value=(tvmjs, "0.27.0.post1")
            ):
                with self.assertRaisesRegex(ValueError, "missing"):
                    self._package(
                        FakeExecutable(),
                        [object()],
                        output,
                        tokenizer,
                        tokenizer_config,
                        model_config,
                    )

            self.assertFalse(output.exists())

    def test_manifest_parameter_count_mismatch_removes_partial_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer, tokenizer_config, model_config = self._inputs(root)
            output = root / "model"
            tvmjs = FakeTvmjs(metadata_count=0)

            with patch.object(
                packager, "_load_tvmjs", return_value=(tvmjs, "0.27.0.post1")
            ):
                with self.assertRaisesRegex(ValueError, "parameter count"):
                    self._package(
                        FakeExecutable(),
                        [object()],
                        output,
                        tokenizer,
                        tokenizer_config,
                        model_config,
                    )

            self.assertFalse(output.exists())

    def test_shard_size_mismatch_is_rejected_and_cleaned_up(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer, tokenizer_config, model_config = self._inputs(root)
            output = root / "model"
            tvmjs = FakeTvmjs(shard_size=100)

            with patch.object(
                packager, "_load_tvmjs", return_value=(tvmjs, "0.27.0.post1")
            ):
                with self.assertRaisesRegex(ValueError, "size does not match"):
                    self._package(
                        FakeExecutable(),
                        [object()],
                        output,
                        tokenizer,
                        tokenizer_config,
                        model_config,
                    )

            self.assertFalse(output.exists())

    def test_shard_larger_than_cap_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer, tokenizer_config, model_config = self._inputs(root)
            output = root / "model"
            tvmjs = FakeTvmjs()

            with (
                patch.object(
                    packager, "_load_tvmjs", return_value=(tvmjs, "0.27.0.post1")
                ),
                patch.object(packager, "MAX_SHARD_BYTES", 1),
            ):
                with self.assertRaisesRegex(ValueError, "exceeds 128 MiB"):
                    self._package(
                        FakeExecutable(),
                        [object()],
                        output,
                        tokenizer,
                        tokenizer_config,
                        model_config,
                    )

            self.assertFalse(output.exists())

    def test_manifest_records_actual_kev_parameter_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer, tokenizer_config, model_config = self._inputs(root)
            output = root / "model"
            tvmjs = FakeTvmjs()
            params = [object() for _ in range(511)]

            with patch.object(
                packager, "_load_tvmjs", return_value=(tvmjs, "0.27.0.post1")
            ):
                result = self._package(
                    FakeExecutable(),
                    params,
                    output,
                    tokenizer,
                    tokenizer_config,
                    model_config,
                )

            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(result.parameter_count, 511)
            self.assertEqual(manifest["parameterCount"], 511)
            self.assertEqual(
                manifest["files"]["weights/params_shard_0.bin"]["bytes"], 511 * 4
            )

    def test_unaligned_or_out_of_range_tensor_records_are_rejected(self):
        cases = (
            ({"byteOffset": 2}, "four-byte aligned"),
            ({"nbytes": 2, "dtype": "float16"}, "four-byte aligned"),
            ({"byteOffset": 4}, "byte range"),
            ({"dtype": "int32"}, "unsupported parameter record"),
            ({"shape": [0]}, "invalid shape"),
            ({"nbytes": 8}, "byte count differs"),
        )
        for record_updates, message in cases:
            with self.subTest(record_updates=record_updates):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    tokenizer, tokenizer_config, model_config = self._inputs(root)
                    output = root / "model"
                    tvmjs = FakeTvmjs(record_updates=record_updates)

                    with patch.object(
                        packager, "_load_tvmjs", return_value=(tvmjs, "0.27.0.post1")
                    ):
                        with self.assertRaisesRegex(ValueError, message):
                            self._package(
                                FakeExecutable(),
                                [object()],
                                output,
                                tokenizer,
                                tokenizer_config,
                                model_config,
                            )

                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
