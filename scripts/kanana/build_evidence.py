"""Write a verifiable record beside a completed native TVM build."""

from __future__ import annotations

import hashlib
from pathlib import Path


def add_tvm_build_evidence(
    report: dict,
    *,
    model_id: str,
    repository: str,
    revision: str,
    checkpoint_sha256: str,
    graph: str,
    wasm_path: Path,
    build_root: Path,
    tvm_version: str,
) -> dict:
    """Add standard release fields after the compiler has produced the WASM."""
    root = build_root.resolve(strict=True)
    wasm = wasm_path.resolve(strict=True)
    relative_wasm = wasm.relative_to(root)
    wasm_bytes = wasm.read_bytes()
    if str(root).encode() in wasm_bytes:
        raise RuntimeError(f"The TVM WASM embeds its build root: {root}")

    evidence = dict(report)
    evidence.update(
        {
            "evidenceSchema": "web-decision-tvm-build-v1",
            "modelId": model_id,
            "artifact": wasm.name,
            "wasm": relative_wasm.as_posix(),
            "status": "success",
            "target": "webgpu / wasm32",
            "tvmVersion": tvm_version,
            "embeddedBuildRoot": False,
            "wasmBytes": len(wasm_bytes),
            "wasmSha256": hashlib.sha256(wasm_bytes).hexdigest(),
            "repository": repository,
            "modelRevision": revision,
            "checkpointSha256": checkpoint_sha256,
            "graph": graph,
            "buildBackend": "native-tvm",
            "weightSource": "pinned-safetensors",
            "onnxImportOrExportUsed": False,
        }
    )
    return evidence
