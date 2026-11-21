"""Independently rebuild Mapika's TVM modules and write release evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from build_text_tvm import MODEL_REVISION
from build_text_tvm import build as build_text

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "mapika-decider-2b-v12"
ASSETS = ROOT / "packages/web-decision/assets/mapika2b"


def record_build(output: Path, artifact: str, environment: dict) -> dict:
    build_root = output / ("vision" if artifact.startswith("vision/") else "text")
    wasm = build_root / Path(artifact).name
    wasm_bytes = wasm.read_bytes()
    if str(build_root.resolve()).encode() in wasm_bytes:
        raise ValueError("The TVM WASM embeds its build root")
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    pin = manifest["files"][artifact]
    digest = hashlib.sha256(wasm_bytes).hexdigest()
    if pin != {"bytes": len(wasm_bytes), "sha256": digest}:
        raise ValueError(f"Rebuilt TVM WASM differs from the manifest: {artifact}")
    report = {
        "evidenceSchema": "web-decision-tvm-build-v1",
        "modelId": MODEL_ID,
        "artifact": artifact,
        "status": "success",
        "target": "webgpu / wasm32",
        "wasm": wasm.name,
        "wasmBytes": len(wasm_bytes),
        "wasmSha256": digest,
        "tvmVersion": environment["pythonPackage"]["version"],
        "embeddedBuildRoot": False,
        "repository": "Mapika/decider-2b",
        "modelRevision": MODEL_REVISION,
        "buildBackend": "native-tvm",
        "onnxImportOrExportUsed": False,
        "buildEnvironment": {"nativeTvm": environment},
    }
    (build_root / "release-build.json").write_text(json.dumps(report, indent=2) + "\n")
    return {
        "modelId": MODEL_ID,
        "artifact": artifact,
        "buildRoot": build_root.relative_to(output).as_posix(),
        "wasm": wasm.name,
        "report": "release-build.json",
    }


def build(output: Path) -> None:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("The build evidence output directory must be empty")
    sys.path.insert(0, str(ROOT / "scripts/kanana"))
    sys.path.insert(0, str(ROOT / "scripts/strands/qwen35"))
    from build_evidence import collect_native_tvm
    from build_fp32_vision_webgpu import build as build_vision

    environment = collect_native_tvm(ROOT)
    build_text(output / "text")
    build_vision(output / "vision")
    artifacts = ("text.wasm", "vision/qwen35-dynamic-vision.webgpu.wasm")
    records = [record_build(output, artifact, environment) for artifact in artifacts]
    (output / "build-evidence.json").write_text(
        json.dumps({"schemaVersion": 1, "builds": records}, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    build(parser.parse_args().output_dir)
