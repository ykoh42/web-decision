"""Focused checks for pinned Kanana custom-source verification."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import marshal
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import pinned_model_files
except ModuleNotFoundError:
    from scripts.kanana import pinned_model_files

try:
    import probe_torch_export_vision_tvm as vision_probe
except ModuleNotFoundError:
    from scripts.kanana import probe_torch_export_vision_tvm as vision_probe


def _function_calls(function):
    calls = (node for node in ast.walk(function) if isinstance(node, ast.Call))
    return sorted(calls, key=lambda node: (node.lineno, node.col_offset))


def _named_call_lines(function, name):
    return [
        call.lineno
        for call in _function_calls(function)
        if getattr(call.func, "id", getattr(call.func, "attr", None)) == name
    ]


def _trusted_load_lines(function):
    return [
        call.lineno
        for call in _function_calls(function)
        if any(
            keyword.arg == "trust_remote_code"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )
    ]


def _script_functions(name):
    path = Path(__file__).resolve().parent / name
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


class PinnedModelFilesTests(unittest.TestCase):
    def _model_directory(self, root: Path, content: bytes) -> dict[str, str]:
        expected = {}
        for name in pinned_model_files.PINNED_KANANA_MODEL_FILES:
            (root / name).write_bytes(content)
            expected[name] = hashlib.sha256(content).hexdigest()
        return expected

    def test_accepts_all_files_that_match_the_pinned_hashes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory)
            expected = self._model_directory(model_dir, b"verified source")
            with patch.dict(
                pinned_model_files.PINNED_KANANA_MODEL_FILES, expected, clear=True
            ):
                self.assertEqual(
                    pinned_model_files.verify_pinned_model_files(model_dir), expected
                )

    def test_rejects_a_tampered_custom_model_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory)
            expected = self._model_directory(model_dir, b"verified source")
            tampered_name = "modeling_kanana_v_decide.py"
            (model_dir / tampered_name).write_bytes(b"tampered source")
            with patch.dict(
                pinned_model_files.PINNED_KANANA_MODEL_FILES, expected, clear=True
            ):
                with self.assertRaisesRegex(SystemExit, tampered_name):
                    pinned_model_files.verify_pinned_model_files(model_dir)

    def test_rejects_a_missing_custom_model_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory)
            expected = self._model_directory(model_dir, b"verified source")
            missing_name = "configuration_kanana_v_decide.py"
            (model_dir / missing_name).unlink()
            with patch.dict(
                pinned_model_files.PINNED_KANANA_MODEL_FILES, expected, clear=True
            ):
                with self.assertRaisesRegex(SystemExit, missing_name):
                    pinned_model_files.verify_pinned_model_files(model_dir)

    def test_vision_importer_rejects_tampered_files_before_importlib(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory)
            expected = self._model_directory(model_dir, b"verified source")
            (model_dir / "modeling_kanana_v_decide.py").write_bytes(b"tampered source")
            with patch.dict(
                pinned_model_files.PINNED_KANANA_MODEL_FILES, expected, clear=True
            ):
                with patch.object(
                    vision_probe.importlib, "import_module"
                ) as import_module:
                    with self.assertRaises(SystemExit):
                        vision_probe.import_pinned_vision_classes(model_dir)
                    import_module.assert_not_called()

    def test_vision_importer_ignores_tampered_timestamp_pyc(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory)
            expected = self._model_directory(model_dir, b"verified source")
            config_source = b"class KananaVDecideVisionConfig: pass\n"
            model_source = (
                b"from .configuration_kanana_v_decide import KananaVDecideVisionConfig\n"
                b"class KananaVDecideVisionModel: pass\n"
            )
            config_path = model_dir / "configuration_kanana_v_decide.py"
            model_path = model_dir / "modeling_kanana_v_decide.py"
            config_path.write_bytes(config_source)
            model_path.write_bytes(model_source)
            expected[config_path.name] = hashlib.sha256(config_source).hexdigest()
            expected[model_path.name] = hashlib.sha256(model_source).hexdigest()

            pyc_path = Path(importlib.util.cache_from_source(str(config_path)))
            pyc_path.parent.mkdir(parents=True, exist_ok=True)
            stat = config_path.stat()
            malicious_code = compile(
                "raise RuntimeError('malicious timestamp pyc executed')",
                str(config_path),
                "exec",
                dont_inherit=True,
            )
            header = importlib.util.MAGIC_NUMBER + struct.pack(
                "<III", 0, int(stat.st_mtime), stat.st_size
            )
            pyc_path.write_bytes(header + marshal.dumps(malicious_code))

            with patch.dict(
                pinned_model_files.PINNED_KANANA_MODEL_FILES, expected, clear=True
            ):
                config_class, model_class = vision_probe.import_pinned_vision_classes(
                    model_dir
                )

            self.assertEqual(config_class.__name__, "KananaVDecideVisionConfig")
            self.assertEqual(model_class.__name__, "KananaVDecideVisionModel")

    def test_vision_importer_keeps_valid_pinned_files_usable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory)
            expected = self._model_directory(model_dir, b"verified source")
            config_module = SimpleNamespace(KananaVDecideVisionConfig=object)
            model_module = SimpleNamespace(KananaVDecideVisionModel=object)
            with patch.dict(
                pinned_model_files.PINNED_KANANA_MODEL_FILES, expected, clear=True
            ):
                with patch.object(
                    vision_probe.importlib,
                    "import_module",
                    side_effect=[config_module, model_module],
                ):
                    self.assertEqual(
                        vision_probe.import_pinned_vision_classes(model_dir),
                        (object, object),
                    )

    def test_decision_probe_verifies_custom_code_before_trusted_load(self):
        loader = _script_functions("generate_decision_probe.py")["load_pinned_model"]
        verifier_lines = _named_call_lines(loader, "verify_pinned_model_files")
        trusted_load_lines = _trusted_load_lines(loader)
        self.assertLess(min(verifier_lines), min(trusted_load_lines))

    def test_text_export_verifies_custom_code_before_trusted_load(self):
        functions = _script_functions("prototype_text_tvm_relax.py")
        prepare = functions["_prepare_run"]
        verifier_lines = _named_call_lines(prepare, "verify_pinned_model_files")
        config_load_lines = _named_call_lines(prepare, "_load_config")
        main_prepare_lines = _named_call_lines(functions["main"], "_prepare_run")
        main_pipeline_lines = _named_call_lines(functions["main"], "_run_pipeline")
        self.assertLess(min(verifier_lines), min(config_load_lines))
        self.assertTrue(_trusted_load_lines(functions["_load_config"]))
        self.assertTrue(_trusted_load_lines(functions["_load_model_stage"]))
        self.assertLess(min(main_prepare_lines), min(main_pipeline_lines))

    def test_conversion_recipe_pins_all_local_build_scripts(self):
        repository_root = Path(__file__).resolve().parents[2]
        recipe_path = (
            repository_root
            / "packages/web-decision/src/catalog/provenance/kanana-conversion-recipe.json"
        )
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        sources = {
            recipe[section]["script"]: recipe[section]["scriptSha256"]
            for section in ("text", "vision", "package")
        }
        sources.update(recipe["supportingScripts"])
        sources[recipe["buildEnvironment"]["evidenceScript"]] = recipe[
            "buildEnvironment"
        ]["evidenceScriptSha256"]
        for name, expected in sources.items():
            with self.subTest(source=name):
                actual = hashlib.sha256(
                    (repository_root / name).read_bytes()
                ).hexdigest()
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
