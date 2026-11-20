"""Read exact Qwen TVMScript inputs and export from the pinned native TVM toolchain."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COMPILER_INPUTS = HERE / "compiler-inputs"
PINNED_TVM_COMMIT = "7b5acb32bd271342a24130f4ef439ff6c2e52c6c"
PINNED_EMSDK_COMMIT = "d7b4130688bfa348392b921ab04015aa15c4f154"
PINNED_EMSCRIPTEN_VERSION = "4.0.23"
METADATA_SHA256 = "0542af92a73d0c5944b0c8a33e41e7c80e251d433ebfc48c5c8e06b66bd3495b"
SOURCE_GRAPH_DIGESTS = {
    "text": "624cc890ac1ffb92caac65fd27b9c924f2d0b71bb009325bf16aa77fe8f6830a",
    "vision": "41b4495d1494328ed6092119daac990c15d3fc1e7b8aa993b1b6d5255b8aa4d4",
}


def pinned(path: Path, digest: str) -> bytes:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Qwen compiler input hash mismatch for {path.name}")
    return data


def load_compiler_input(kind: str, tvm, relax):
    import numpy as np

    record = json.loads(pinned(COMPILER_INPUTS / "metadata.json", METADATA_SHA256))[
        kind
    ]
    if record["parentGraphSha256"] != SOURCE_GRAPH_DIGESTS[kind]:
        raise ValueError("Qwen TVMScript parent hash differs from its pinned source")
    source = pinned(
        COMPILER_INPUTS / f"{kind}.tvmscript", record["scriptSha256"]
    ).decode()
    frequencies = np.asarray(record["inverseFrequency"], dtype=np.float32)
    if (
        hashlib.sha256(frequencies.tobytes()).hexdigest()
        != record["inverseFrequencySha256"]
    ):
        raise ValueError("Qwen RoPE frequencies differ from their pinned bytes")
    constants = [relax.const(frequencies, "float32")]
    for _ in range(record.get("zeroBiasCount", 0)):
        constants.append(relax.const(np.zeros((6144,), dtype=np.float32), "float32"))
    metadata = {"ir.GenericConst": constants}
    if "parameterCount" in record:
        metadata["ffi.Tensor"] = [
            tvm.runtime.empty((1,), dtype="float32")
            for _ in range(record["parameterCount"])
        ]
    return source, metadata


def inspect_webgpu_source(module: Any) -> str:
    pending = [module]
    seen: set[int] = set()
    shaders = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if current.kind == "webgpu":
            shaders.append(current.inspect_source())
        pending.extend(current.imports)
    if not shaders:
        raise RuntimeError("Qwen executable has no WebGPU shaders")
    return "\n\n".join(shaders)


def verify_tvm_checkout(tvm_home: Path) -> str:
    revision = subprocess.check_output(
        ["git", "-C", str(tvm_home), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(tvm_home), "status", "--porcelain"], text=True
    )
    if revision != PINNED_TVM_COMMIT or dirty:
        raise RuntimeError(
            "TVM source checkout differs from the pinned, unchanged upstream revision"
        )
    return revision


def verify_emsdk_checkout(emsdk: Path) -> str:
    revision = subprocess.check_output(
        ["git", "-C", str(emsdk), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != PINNED_EMSDK_COMMIT:
        raise RuntimeError("Emscripten SDK checkout differs from the pinned revision")
    return revision


def export_tvmjs(executable: Any, wasm_path: Path, emsdk: Path, tvm_home: Path) -> str:
    from tvm.contrib import tvmjs

    node = next((emsdk / "node").glob("*/bin/node"))
    python = next((emsdk / "python").glob("*/bin/python3"))
    cert_file = next(
        (emsdk / "python").glob("*/lib/python*/site-packages/certifi/cacert.pem")
    )
    keys = (
        "TVM_HOME",
        "EMSDK",
        "EMSDK_NODE",
        "EMSDK_PYTHON",
        "EM_CONFIG",
        "SSL_CERT_FILE",
        "PATH",
    )
    previous = {key: os.environ.get(key) for key in keys}
    os.environ.update(
        {
            "TVM_HOME": str(tvm_home),
            "EMSDK": str(emsdk),
            "EMSDK_NODE": str(node),
            "EMSDK_PYTHON": str(python),
            "EM_CONFIG": str(emsdk / ".emscripten"),
            "SSL_CERT_FILE": str(cert_file),
            "PATH": os.pathsep.join(
                (str(emsdk), str(emsdk / "upstream/emscripten"), previous["PATH"] or "")
            ),
        }
    )
    try:
        version = subprocess.check_output(
            ["emcc", "--version"], text=True
        ).splitlines()[0]
        if PINNED_EMSCRIPTEN_VERSION not in version:
            raise RuntimeError(
                f"Emscripten version differs from the pinned SDK: {version}"
            )
        executable.export_library(
            str(wasm_path),
            fcompile=tvmjs.create_tvmjs_wasm,
            options=[f"-ffile-prefix-map={ROOT}=/src"],
        )
        return version
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
