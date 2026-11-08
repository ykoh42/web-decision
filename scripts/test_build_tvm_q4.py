"""Exercise the TVM build's staging and orchestration boundaries."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import build_tvm_q4 as builder
import pytest


def test_custom_operator_modules_are_loaded_before_building():
    expected = {"tvm_selective_params", "tvm_narrow_int64", "tvm_q4_operators"}
    assert set(builder.CUSTOM_OPERATOR_MODULES) == expected
    assert expected <= sys.modules.keys()


def make_root(root: Path) -> Path:
    for name in ("models", "public", "reports"):
        (root / name).mkdir()
    return root


def stage_names(root: Path) -> list[str]:
    return [
        path.name for path in root.iterdir() if path.name.startswith(".tvm-q4-stage")
    ]


@pytest.mark.parametrize(
    ("directory", "output", "existing"),
    [
        ("public", "tvm-kev-q4-seq69.wasm", b"keep wasm"),
        ("models", "tvm-kev-q4-seq69", None),
    ],
)
def test_open_workspace_refuses_existing_outputs_without_changes(
    tmp_path, directory, output, existing
):
    root = make_root(tmp_path)
    destination = root / directory / output
    if existing is None:
        destination.mkdir()
    else:
        destination.write_bytes(existing)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        builder.open_workspace(root, "seq69")

    assert stage_names(root) == []
    assert (
        destination.is_dir()
        if existing is None
        else destination.read_bytes() == existing
    )


def test_open_workspace_closes_open_directories_when_stage_creation_fails(
    tmp_path, monkeypatch
):
    root = make_root(tmp_path)
    opened_descriptors = []
    open_directory = builder.open_directory

    def record_directory(*args):
        descriptor = open_directory(*args)
        opened_descriptors.append(descriptor)
        return descriptor

    def fail_stage(*_args):
        raise OSError("stage unavailable")

    monkeypatch.setattr(builder, "open_directory", record_directory)
    monkeypatch.setattr(builder, "create_temporary_directory", fail_stage)

    with pytest.raises(OSError, match="stage unavailable"):
        builder.open_workspace(root, "seq69")

    for descriptor in opened_descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert stage_names(root) == []


def test_close_workspace_removes_staging_tree(tmp_path):
    root = make_root(tmp_path)
    workspace = builder.open_workspace(root, "seq69")
    (workspace.staged_cache / "partial").mkdir(parents=True)
    (workspace.staged_cache / "partial" / "weights.bin").write_bytes(b"partial")

    builder.close_workspace(workspace)

    assert stage_names(root) == []


def test_wasm_promotion_failure_rolls_back_cache_and_preserves_racing_output(
    tmp_path, monkeypatch
):
    root = make_root(tmp_path)
    workspace = builder.open_workspace(root, "seq69")
    workspace.staged_wasm.write_bytes(b"built wasm")
    workspace.staged_cache.mkdir()
    (workspace.staged_cache / "tensor-cache.json").write_bytes(b"manifest")
    promote_file = builder.promote_file

    def collide_with_wasm(source_fd, source_name, target_fd, target_name):
        descriptor = os.open(
            target_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=target_fd,
        )
        try:
            os.write(descriptor, b"racing output")
        finally:
            os.close(descriptor)
        promote_file(source_fd, source_name, target_fd, target_name)

    monkeypatch.setattr(builder, "promote_file", collide_with_wasm)
    try:
        with pytest.raises(FileExistsError):
            builder.promote_outputs(workspace, {})
        assert not workspace.cache_output.exists()
        assert workspace.wasm_output.read_bytes() == b"racing output"
    finally:
        builder.close_workspace(workspace)

    assert stage_names(root) == []


def test_run_build_preserves_compile_stage_order(monkeypatch):
    events = []

    def record(name, result=None):
        def run(*_args):
            events.append(name)
            return result

        return run

    monkeypatch.setattr(builder, "load_and_import_model", record("load", ("mod", [])))
    monkeypatch.setattr(builder, "create_target", record("target", "target"))
    monkeypatch.setattr(builder, "lower_model", record("lower", "lowered"))
    monkeypatch.setattr(builder, "validate_q4_kernels", record("validate"))
    monkeypatch.setattr(builder, "compile_model", record("compile", "executable"))
    monkeypatch.setattr(builder, "export_wasm", record("export"))
    monkeypatch.setattr(builder, "write_tensor_cache", record("cache"))
    monkeypatch.setattr(builder, "promote_outputs", record("promote"))

    builder.run_build(
        SimpleNamespace(dynamic=False, seq_len=69), Path("model.onnx"), None, {}, 0
    )

    assert events == [
        "load",
        "target",
        "lower",
        "validate",
        "compile",
        "export",
        "cache",
        "promote",
    ]


def test_main_writes_failure_report_and_cleans_staging(tmp_path, monkeypatch):
    root = make_root(tmp_path)

    def fail_build(*_args):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(builder, "ROOT", root)
    monkeypatch.setattr(builder, "run_build", fail_build)

    with pytest.raises(SystemExit) as error:
        builder.main([])

    report = json.loads((root / "reports/tvm-q4-build.json").read_text())
    assert error.value.code == 1
    assert report["status"] == "failed"
    assert report["error"] == "compile failed"
    assert stage_names(root) == []
