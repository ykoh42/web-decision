"""Package an already-built Relax executable for direct TVM WebGPU loading."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any

SHARD_CAP_MB = 128
MAX_SHARD_BYTES = SHARD_CAP_MB * 1024 * 1024
MAX_SAFE_INTEGER = 2**53 - 1
MAX_POSITION_EMBEDDINGS = 32_768
MODEL_ID = "kev-0.6b-q4f16"
MODEL_BACKEND = "tvm-webgpu"
MAX_STATE_TOKENS = 8192
MAX_BRANCH_TOKENS = 8192
MAX_OPTIONS = 255
KEV_DELIMITER_IDS = {
    "state": 151659,
    "question": 151660,
    "option_start": 151648,
    "option_end": 151649,
    "decide": 151661,
}
CHECKPOINT = {
    "repository": "jaredpalmer/kev-0.6b",
    "revision": "abe7110dedfd68a185839550079bcf7c26f561ac",
}
BASE_MODEL = {
    "repository": "Qwen/Qwen3-0.6B-Base",
    "revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
}
KEV_CODE = {
    "repository": "jaredpalmer/kev",
    "revision": "37cb2f6155afec1054de2cc78dfd7c7dd24a956d",
}
LICENSE_PROVENANCE = (
    {
        "file": "notices/kev-LICENSE",
        "spdxId": "Apache-2.0",
        "repository": KEV_CODE["repository"],
        "revision": KEV_CODE["revision"],
        "sourcePath": "LICENSE",
    },
    {
        "file": "notices/qwen3-0.6b-base-LICENSE",
        "spdxId": "Apache-2.0",
        "repository": BASE_MODEL["repository"],
        "revision": BASE_MODEL["revision"],
        "sourcePath": "LICENSE",
    },
)
_LICENSE_MARKERS = (b"Apache License", b"Version 2.0, January 2004")
_CONFIG_NAMES = ("tokenizer.json", "tokenizer_config.json", "model_config.json")
_DTYPE_BYTES = {"float16": 2, "float32": 4, "uint32": 4}


@dataclass(frozen=True)
class PackageResult:
    """Summary of a successfully written direct TVM WebGPU package."""

    output_dir: Path
    parameter_count: int
    shard_count: int
    manifest_sha256: str


def package_tvm_webgpu_artifacts(
    executable: Any,
    runtime_params: Iterable[Any],
    output_dir: str | os.PathLike[str],
    tokenizer_json: str | os.PathLike[str],
    tokenizer_config: str | os.PathLike[str],
    model_config: str | os.PathLike[str],
    kev_license: str | os.PathLike[str],
    base_license: str | os.PathLike[str],
) -> PackageResult:
    """Write a new model directory using only the public TVM JS export APIs.

    The destination must be new, and no component of its path may be a symlink.
    Its parent must already exist. Any failure after creating the destination
    removes that directory, leaving no partial package behind.
    """
    params = _ordered_params(runtime_params)
    config_sources = _config_sources(tokenizer_json, tokenizer_config, model_config)
    license_sources = _license_sources(kev_license, base_license)
    merged_config = _merged_model_config(config_sources[2][1])
    destination = _new_destination(output_dir)
    tvmjs, tvm_version = _load_tvmjs()

    destination.mkdir(mode=0o700)
    try:
        wasm_path = destination / "model.wasm"
        executable.export_library(str(wasm_path), fcompile=tvmjs.create_tvmjs_wasm)
        for name, source in config_sources[:2]:
            shutil.copyfile(source, destination / name)
        _write_json_file(destination / "model_config.json", merged_config)
        notices_dir = destination / "notices"
        notices_dir.mkdir(mode=0o700)
        for name, source in license_sources:
            shutil.copyfile(source, notices_dir / name)

        cache_dir = destination / "weights"
        tvmjs.dump_tensor_cache(
            ((f"param_{index}", value) for index, value in enumerate(params)),
            str(cache_dir),
            encode_format="raw",
            meta_data={"numParams": len(params)},
            shard_cap_mb=SHARD_CAP_MB,
        )
        shard_names = _verify_package(destination, len(params))
        manifest = _build_manifest(destination, shard_names, len(params), tvm_version)
        manifest_bytes = _serialize_json(manifest)
        _write_new_bytes(destination / "manifest.json", manifest_bytes)
        return PackageResult(
            destination,
            len(params),
            len(shard_names),
            hashlib.sha256(_canonical_manifest_bytes(manifest)).hexdigest(),
        )
    except BaseException:
        _remove_output(destination)
        raise


def _ordered_params(runtime_params: Iterable[Any]) -> tuple[Any, ...]:
    if isinstance(runtime_params, (str, bytes, Mapping, AbstractSet)):
        raise TypeError("runtime_params must be an ordered iterable of runtime values")
    try:
        return tuple(runtime_params)
    except TypeError as error:
        raise TypeError("runtime_params must be an ordered iterable") from error


def _config_sources(*paths: str | os.PathLike[str]) -> tuple[tuple[str, Path], ...]:
    sources = []
    for destination_name, path in zip(_CONFIG_NAMES, paths, strict=True):
        source = Path(path).expanduser()
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"configuration source is not a regular file: {source}")
        sources.append((destination_name, source))
    return tuple(sources)


def _license_sources(*paths: str | os.PathLike[str]) -> tuple[tuple[str, Path], ...]:
    names = tuple(Path(record["file"]).name for record in LICENSE_PROVENANCE)
    sources = []
    for name, path in zip(names, paths, strict=True):
        source = Path(path).expanduser()
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"license source is not a regular file: {source}")
        try:
            license_bytes = source.read_bytes()
        except OSError as error:
            raise ValueError(f"could not read license source: {source}") from error
        if any(marker not in license_bytes for marker in _LICENSE_MARKERS):
            raise ValueError(
                f"license source does not contain Apache-2.0 text: {source}"
            )
        sources.append((name, source))
    return tuple(sources)


def _new_destination(output_dir: str | os.PathLike[str]) -> Path:
    destination = Path(os.path.abspath(Path(output_dir).expanduser()))
    for component in (destination, *destination.parents):
        if component.is_symlink():
            raise ValueError(
                f"Refusing output path with symlink component: {component}"
            )
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"Output parent must already exist: {destination.parent}"
        )
    return destination


def _load_tvmjs() -> tuple[Any, str]:
    import tvm
    from tvm.contrib import tvmjs

    return tvmjs, tvm.__version__


def _merged_model_config(source: Path) -> dict[str, Any]:
    try:
        config = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Official base model config is not valid JSON") from error
    if not isinstance(config, dict):
        raise ValueError("Official base model config must be a JSON object")
    if config.get("max_position_embeddings") != MAX_POSITION_EMBEDDINGS:
        raise ValueError("Official base model config has an unexpected position limit")

    kev_config = config.get("kev", {})
    if not isinstance(kev_config, dict):
        raise ValueError("Official base model config has an invalid Kev section")
    values = {
        "delimiter_ids": KEV_DELIMITER_IDS,
        "max_state_tokens": MAX_STATE_TOKENS,
        "max_branch_tokens": MAX_BRANCH_TOKENS,
        "max_options": MAX_OPTIONS,
    }
    for key, expected in values.items():
        if key in kev_config and kev_config[key] != expected:
            raise ValueError(f"Official base config has unexpected Kev {key}")
    config["kev"] = {**kev_config, **values, "delimiter_ids": dict(KEV_DELIMITER_IDS)}
    return config


def _verify_package(output_dir: Path, expected_params: int) -> tuple[str, ...]:
    _require_regular_file(output_dir / "model.wasm", nonempty=True)
    for name in _CONFIG_NAMES:
        _require_regular_file(output_dir / name, nonempty=True)
    expected_root = {
        "model.wasm",
        "tokenizer.json",
        "tokenizer_config.json",
        "model_config.json",
        "weights",
        "notices",
    }
    if {child.name for child in output_dir.iterdir()} != expected_root:
        raise ValueError("TVM wrote unexpected files into the model package")
    cache_dir = output_dir / "weights"
    if cache_dir.is_symlink() or not cache_dir.is_dir():
        raise ValueError("TVM tensor cache is not a regular directory")
    notices_dir = output_dir / "notices"
    if notices_dir.is_symlink() or not notices_dir.is_dir():
        raise ValueError("Package notices are not a regular directory")
    expected_notices = {Path(record["file"]).name for record in LICENSE_PROVENANCE}
    for name in expected_notices:
        _require_regular_file(notices_dir / name, nonempty=True)
    if {child.name for child in notices_dir.iterdir()} != expected_notices:
        raise ValueError("Package notices do not match the pinned license inputs")

    manifest_path = cache_dir / "tensor-cache.json"
    _require_regular_file(manifest_path, nonempty=True)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("TVM tensor-cache manifest is not valid JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("TVM tensor-cache manifest must be an object")

    _verify_parameter_records(manifest, expected_params)
    return _verify_shards(cache_dir, manifest)


def _verify_parameter_records(manifest: dict[str, Any], expected_params: int) -> None:
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict) or type(metadata.get("numParams")) is not int:
        raise ValueError("TVM tensor-cache manifest is missing metadata.numParams")
    if metadata["numParams"] != expected_params:
        raise ValueError("TVM tensor-cache parameter count does not match input")

    names = _parameter_record_names(manifest.get("records"), expected_params)
    expected_names = [f"param_{index}" for index in range(expected_params)]
    if names != expected_names:
        raise ValueError("TVM tensor-cache parameter names or order do not match input")


def _parameter_record_names(shards: Any, expected_params: int) -> list[str]:
    if not isinstance(shards, list):
        raise ValueError("TVM tensor-cache manifest is missing shard records")
    names = []
    for shard in shards:
        names.extend(_shard_record_names(shard, expected_params))
    return names


def _shard_record_names(shard: Any, expected_params: int) -> list[str]:
    if not isinstance(shard, dict) or shard.get("format") != "raw-shard":
        raise ValueError("TVM tensor-cache contains an invalid raw shard")
    records = shard.get("records") if isinstance(shard, dict) else None
    shard_bytes = shard.get("nbytes")
    if not isinstance(records, list):
        raise ValueError("TVM tensor-cache shard is missing tensor records")
    if not _safe_integer(shard_bytes) or shard_bytes < 0:
        raise ValueError("TVM tensor-cache shard has an invalid byte count")
    names = []
    for record in records:
        names.append(_verify_tensor_record(record, shard_bytes, expected_params))
    return names


def _verify_tensor_record(record: Any, shard_bytes: int, expected_params: int) -> str:
    if not isinstance(record, dict) or record.get("format") != "raw":
        raise ValueError("TVM tensor-cache contains an invalid raw tensor record")
    name = record.get("name")
    match = re.fullmatch(r"param_(\d+)", name) if isinstance(name, str) else None
    dtype = record.get("dtype")
    if match is None or not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
        raise ValueError("TVM tensor-cache contains an unsupported parameter record")
    index = int(match.group(1))
    if index >= expected_params:
        raise ValueError("TVM tensor-cache parameter index exceeds input count")
    shape = record.get("shape")
    if not isinstance(shape, list) or any(
        not _safe_integer(dimension) or dimension < 1 for dimension in shape
    ):
        raise ValueError(f"Tensor-cache record has an invalid shape: {name}")
    expected_bytes = prod(shape) * _DTYPE_BYTES[dtype]
    nbytes = record.get("nbytes")
    if not _safe_integer(nbytes) or nbytes != expected_bytes:
        raise ValueError(f"Tensor-cache record byte count differs from shape: {name}")
    offset = record.get("byteOffset")
    if not _safe_integer(offset) or offset < 0 or offset % 4 or nbytes % 4:
        raise ValueError(f"Tensor-cache record is not four-byte aligned: {name}")
    if offset + nbytes > shard_bytes:
        raise ValueError(f"Tensor-cache record exceeds its shard byte range: {name}")
    return name


def _safe_integer(value: Any) -> bool:
    return type(value) is int and 0 <= value <= MAX_SAFE_INTEGER


def _verify_shards(cache_dir: Path, manifest: dict[str, Any]) -> tuple[str, ...]:
    shards = manifest["records"]
    expected_names = set()
    for shard in shards:
        name = _verify_shard(cache_dir, shard)
        if name in expected_names:
            raise ValueError(f"Duplicate tensor-cache shard: {name}")
        expected_names.add(name)
    _verify_cache_entries(cache_dir, expected_names)
    return tuple(sorted(expected_names))


def _verify_shard(cache_dir: Path, shard: Any) -> str:
    if not isinstance(shard, dict) or shard.get("format") != "raw-shard":
        raise ValueError("TVM tensor-cache contains an invalid raw shard")
    name = _shard_name(shard.get("dataPath"))
    shard_path = cache_dir / name
    _require_regular_file(shard_path)
    size = shard_path.stat().st_size
    if not _safe_integer(shard.get("nbytes")) or shard["nbytes"] != size:
        raise ValueError(f"Tensor-cache shard size does not match manifest: {name}")
    if size > MAX_SHARD_BYTES:
        raise ValueError(f"Tensor-cache shard exceeds {SHARD_CAP_MB} MiB: {name}")
    return name


def _verify_cache_entries(cache_dir: Path, expected_names: set[str]) -> None:
    actual_names = {
        child.name for child in cache_dir.iterdir() if child.name.endswith(".bin")
    }
    if actual_names != expected_names:
        raise ValueError("Tensor-cache shard count does not match manifest")
    expected_entries = expected_names | {"tensor-cache.json"}
    if {child.name for child in cache_dir.iterdir()} != expected_entries:
        raise ValueError("TVM wrote unexpected files into the tensor cache")


def _build_manifest(
    output_dir: Path,
    shard_names: tuple[str, ...],
    parameter_count: int,
    tvm_version: str,
) -> dict[str, Any]:
    names = (
        "model.wasm",
        "model_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        *(str(record["file"]) for record in LICENSE_PROVENANCE),
        "weights/tensor-cache.json",
        *(f"weights/{name}" for name in shard_names),
    )
    files = {name: _file_metadata(output_dir / name) for name in sorted(names)}
    return {
        "formatVersion": 1,
        "model": MODEL_ID,
        "backend": MODEL_BACKEND,
        "experimental": True,
        "parameterCount": parameter_count,
        "maxStateTokens": MAX_STATE_TOKENS,
        "maxBranchTokens": MAX_BRANCH_TOKENS,
        "maxOptions": MAX_OPTIONS,
        "license": "Apache-2.0",
        "licenses": [dict(record) for record in LICENSE_PROVENANCE],
        "tvmVersion": tvm_version,
        "checkpoint": dict(CHECKPOINT),
        "base": dict(BASE_MODEL),
        "sourceCode": dict(KEV_CODE),
        "files": files,
    }


def _file_metadata(path: Path) -> dict[str, Any]:
    _require_regular_file(path, nonempty=True)
    return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_file(path: Path, value: dict[str, Any]) -> None:
    _write_new_bytes(path, _serialize_json(value))


def _serialize_json(value: dict[str, Any]) -> bytes:
    text = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    )
    return f"{text}\n".encode("utf-8")


def _canonical_manifest_bytes(value: dict[str, Any]) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return f"{text}\n".encode("utf-8")


def _write_new_bytes(path: Path, content: bytes) -> None:
    with path.open("xb") as output:
        output.write(content)


def _shard_name(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value.endswith(".bin")
        or Path(value).name != value
        or value in (".", "..")
        or "\\" in value
    ):
        raise ValueError(f"Unsafe tensor-cache shard path: {value!r}")
    return value


def _require_regular_file(path: Path, *, nonempty: bool = False) -> None:
    try:
        file_stat = path.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"Expected output file is missing: {path.name}") from error
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"Expected a regular output file: {path.name}")
    if nonempty and file_stat.st_size == 0:
        raise ValueError(f"Expected a nonempty output file: {path.name}")


def _remove_output(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
