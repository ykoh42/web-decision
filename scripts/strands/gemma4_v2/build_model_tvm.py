"""Build the pinned Strands Gemma 4 E2B v2 WebGPU module with native TVM."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
QWEN_BUILD = ROOT / "scripts/strands/qwen35"
MODEL_ID = "strands-decider-e2b-gemma4-v2-2611"
MODEL_REPOSITORY = "StrandsAgents/strands-decider-E2B-gemma4-v2-2611"
MODEL_REVISION = "a5f32ea1bffe2e91c25c0c641d444e8115e28212"
WASM_NAME = "layer.wasm"
REPORT_NAME = "release-build.json"
EVIDENCE_NAME = "build-evidence.json"
WASM_RUNTIME_ASSETS = ("wasm_runtime.bc", "tvmjs_support.bc", "webgpu_runtime.bc")
MODEL_FILES = {
    "config.json": (
        4954,
        "1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330",
    ),
    "model.safetensors": (
        10246621918,
        "593a28a67754653c7ffffff80a5186f18966cc6262c2f4f097414c187e5fd48c",
    ),
    "readout.safetensors": (
        783472,
        "37b3f1c4670f1afb78007624e3ec2a2f55503d9b78602216e2f29a4cfb5111f6",
    ),
}
BUILD_PACKAGES = {
    "apache-tvm": "0.27.0.post1",
    "apache-tvm-ffi": "0.1.15.post1",
    "numpy": "2.4.6",
    "safetensors": "0.8.0",
    "torch": "2.8.0",
    "transformers": "5.22.2",
}


def verify_file(path: Path, expected_size: int, expected_sha256: str) -> None:
    if not path.is_file() or path.stat().st_size != expected_size:
        raise ValueError(
            f"Checkpoint file is missing or has an unexpected size: {path.name}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"Checkpoint file hash differs: {path.name}")


def verify_checkpoint(checkpoint: Path) -> None:
    for name, (size, digest) in MODEL_FILES.items():
        verify_file(checkpoint / name, size, digest)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Use the locked environment from scripts/strands/qwen35/build-env/uv.lock. "
            "For example: uv sync --project scripts/strands/qwen35/build-env --frozen --no-install-project."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Pinned Hugging Face checkpoint directory",
    )
    parser.add_argument(
        "--tvm-source",
        required=True,
        type=Path,
        help="Pinned, unchanged Apache TVM checkout",
    )
    parser.add_argument(
        "--emsdk", required=True, type=Path, help="Pinned Emscripten SDK checkout"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory for the compiled layer.wasm and its build evidence",
    )
    return parser.parse_args(argv)


def validate_paths(args):
    for name in ("checkpoint", "tvm_source", "emsdk"):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(
                f"{name.replace('_', '-')} must be an existing directory: {path}"
            )
        setattr(args, name, path)
    args.output = args.output.expanduser().resolve()
    if args.output.exists() and not args.output.is_dir():
        raise ValueError(f"output must be a directory: {args.output}")


def verify_build_environment():
    if sys.version_info[:3] != (3, 11, 11):
        raise ValueError("The locked build environment requires Python 3.11.11")
    for name, expected in BUILD_PACKAGES.items():
        try:
            actual = version(name)
        except PackageNotFoundError as error:
            raise ValueError(
                f"The locked build environment requires {name}=={expected}"
            ) from error
        if actual != expected:
            raise ValueError(
                f"The locked build environment requires {name}=={expected}"
            )


def source_checkout(root: Path) -> dict:
    sys.path.insert(0, str(ROOT))
    from scripts.ci.git_environment import git_environment

    environment = git_environment(root)
    git = ["git", "-C", str(root)]
    head = subprocess.check_output(
        [*git, "rev-parse", "HEAD"], text=True, env=environment
    ).strip()
    status = subprocess.check_output(
        [*git, "status", "--porcelain", "--untracked-files=normal"],
        text=True,
        env=environment,
    )
    return {"head": head, "dirty": bool(status)}


def runtime_asset(tvm_source: Path, name: str) -> dict:
    from build_evidence import sha256_file

    dist = tvm_source / "web/dist"
    path = dist / "wasm" / name
    if not path.is_file():
        path = dist / name
    return {"name": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def build_environment(args, tvm_revision, emsdk_revision, emcc_version, source_before):
    import tvm
    import tvm_ffi
    from build_evidence import (
        find_shared_library,
        installed_python_packages,
        sha256_file,
    )
    from qwen35_tvm_build import PINNED_EMSCRIPTEN_VERSION

    lib_dirs = [Path(path).resolve() for path in tvm.libinfo.package_lib_paths()]
    lockfile = QWEN_BUILD / "build-env/uv.lock"
    return {
        "repository": {"before": source_before, "after": source_checkout(ROOT)},
        "host": {"system": platform.system(), "machine": platform.machine()},
        "python": {"version": platform.python_version()},
        "pythonBuild": {
            "lockfile": lockfile.relative_to(ROOT).as_posix(),
            "lockfileSha256": sha256_file(lockfile),
            "packages": installed_python_packages(),
        },
        "nativeTvm": {
            "pythonPackage": {
                "name": "apache-tvm",
                "version": tvm.__version__,
                "moduleSha256": sha256_file(Path(tvm.__file__)),
                "compilerLibrarySha256": sha256_file(
                    find_shared_library(lib_dirs, "libtvm_compiler")
                ),
                "runtimeLibrarySha256": sha256_file(
                    Path(tvm.libinfo.find_libtvm_runtime())
                ),
                "ffiPackageVersion": version("apache-tvm-ffi"),
                "ffiLibrarySha256": sha256_file(
                    Path(tvm_ffi.libinfo.find_libtvm_ffi())
                ),
            },
            "source": {
                "revision": tvm_revision,
                "patches": [],
                "wasmRuntimeAssets": [
                    runtime_asset(args.tvm_source, name) for name in WASM_RUNTIME_ASSETS
                ],
            },
            "emscripten": {
                "revision": emsdk_revision,
                "version": PINNED_EMSCRIPTEN_VERSION,
                "emccVersion": emcc_version,
                "emccSha256": sha256_file(args.emsdk / "upstream/emscripten/emcc"),
                "clangSha256": sha256_file(args.emsdk / "upstream/bin/clang"),
                "wasmLdSha256": sha256_file(args.emsdk / "upstream/bin/wasm-ld"),
            },
        },
    }


def write_build_evidence(staging: Path, build_root: Path, environment: dict) -> None:
    wasm_bytes = (staging / WASM_NAME).read_bytes()
    if str(build_root.resolve()).encode() in wasm_bytes:
        raise ValueError("The TVM WASM embeds its build root")
    report = {
        "evidenceSchema": "web-decision-tvm-build-v1",
        "modelId": MODEL_ID,
        "artifact": "graph.wasm",
        "status": "success",
        "target": "webgpu / wasm32",
        "wasm": WASM_NAME,
        "wasmBytes": len(wasm_bytes),
        "wasmSha256": hashlib.sha256(wasm_bytes).hexdigest(),
        "tvmVersion": environment["nativeTvm"]["pythonPackage"]["version"],
        "embeddedBuildRoot": False,
        "repository": MODEL_REPOSITORY,
        "modelRevision": MODEL_REVISION,
        "checkpointSha256": MODEL_FILES["model.safetensors"][1],
        "configSha256": MODEL_FILES["config.json"][1],
        "readoutSha256": MODEL_FILES["readout.safetensors"][1],
        "buildBackend": "native-tvm",
        "weightSource": "pinned-safetensors",
        "onnxImportOrExportUsed": False,
        "buildEnvironment": environment,
    }
    evidence = {
        "schemaVersion": 1,
        "builds": [
            {
                "modelId": MODEL_ID,
                "artifact": report["artifact"],
                "buildRoot": ".",
                "wasm": WASM_NAME,
                "report": REPORT_NAME,
            }
        ],
    }
    (staging / REPORT_NAME).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (staging / EVIDENCE_NAME).write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )


def with_global_symbol(func, symbol):
    if func.attrs and "global_symbol" in func.attrs:
        return func.with_attr("global_symbol", symbol)
    return func


def link_modules(modules):
    import tvm
    from tvm import relax
    from tvm.relax.expr_functor import PyExprMutator

    @relax.expr_functor.mutator
    class RenameGlobals(PyExprMutator):
        def __init__(self, module, renamed):
            super().__init__(module)
            self.renamed = renamed

        def visit_global_var_(self, old):
            return self.renamed.get(old.name_hint, old)

    combined = {}
    for prefix, module in modules:
        builder = relax.BlockBuilder()
        rename = {}
        for original, func in module.functions.items():
            symbol = prefix + "_" + original.name_hint
            rename[original.name_hint] = builder.add_func(
                with_global_symbol(func, symbol), symbol
            )
        mutator = RenameGlobals(builder.get(), rename)
        for original, func in module.functions.items():
            if isinstance(func, relax.Function):
                func = mutator.visit_expr(func)
            combined[rename[original.name_hint]] = with_global_symbol(
                func, prefix + "_" + original.name_hint
            )
    return tvm.IRModule(combined)


def build(args):
    import image_embedding_ir
    import numpy as np
    import tvm
    import vision_graphs
    from concat_ir import build as build_concat
    from model_partition import partition_model
    from per_layer_ir import build_per_layer_ir
    from query_families import build as build_query_family
    from query_rope_slice import build as build_query_rope_slice
    from readout_ir import build as build_readout
    from rotary_ir import build as build_rope
    from slice_ir import build as build_slice
    from split_transposed_family_ir import build as build_split_family
    from tvm import relax
    from webgpu_grid import guard_model_grids
    from word_embedding_ir import build as build_word

    sys.path.insert(0, str(ROOT / "scripts/kanana"))
    sys.path.insert(0, str(QWEN_BUILD))
    from qwen35_tvm_build import (
        export_tvmjs,
        verify_emsdk_checkout,
        verify_tvm_checkout,
    )

    source_before = source_checkout(ROOT)
    tvm_revision = verify_tvm_checkout(args.tvm_source)
    emsdk_revision = verify_emsdk_checkout(args.emsdk)
    if tvm.__version__ != "0.27.0.post1":
        raise ValueError("The pinned TVM and TVM FFI versions are required")
    verify_checkpoint(args.checkpoint)
    modules = []
    modules.append(
        ("ple", partition_model(build_per_layer_ir(args.checkpoint), "float_main"))
    )
    modules.append(("word", build_word()))
    modules.append(("rope", build_rope()))
    readout, _, _ = build_readout(
        args.checkpoint, np.zeros((1, 80, 1536), dtype=np.float32), host_readout=True
    )
    modules.append(("readout", readout))
    modules.append(("image_insert", image_embedding_ir.build_image_embedding_ir()))
    for name, builder in (
        ("vision_patch", lambda: vision_graphs.build_patch_ir(args.checkpoint)),
        ("vision_encoder", lambda: vision_graphs.build_encoder_ir(args.checkpoint)),
        ("vision_pool", vision_graphs.build_pool_ir),
        (
            "vision_projection",
            lambda: vision_graphs.build_projection_ir(args.checkpoint),
        ),
    ):
        module, _ = builder()
        if name == "vision_projection":
            module = partition_model(module, "main")
        modules.append((name, module))

    for index in (0, 4, 15, 19):
        attention, unpack, tail = build_split_family(args.checkpoint, index)
        attention = partition_model(attention, "float_main")
        tail = partition_model(tail, "main")
        modules.extend(
            (
                (f"split{index}attn", attention),
                (f"split{index}unpack_t", unpack),
                (f"split{index}optimized_tail", tail),
            )
        )
    for index in (0, 4, 15, 19):
        modules.extend(build_query_family(args.checkpoint, index))
    modules.extend(
        (f"query_rope{width}", build_query_rope_slice(width)) for width in (256, 512)
    )
    modules.extend((("split_slice", build_slice()), ("split_concat", build_concat())))
    module = link_modules(modules)
    print(f"Combined {len(module.functions)} TVM functions", flush=True)
    target = tvm.target.Target(
        "webgpu", host={"kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"}
    )
    pipeline = tvm.transform.Sequential(
        [relax.get_default_pipeline(target), guard_model_grids]
    )
    with tvm.transform.PassContext(disabled_pass=["StaticPlanBlockMemory", "FuseOps"]):
        executable = relax.build(module, target=target, relax_pipeline=pipeline)
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output) as temporary:
        staging = Path(temporary)
        emcc_version = export_tvmjs(
            executable, staging / WASM_NAME, args.emsdk, args.tvm_source
        )
        verify_tvm_checkout(args.tvm_source)
        environment = build_environment(
            args, tvm_revision, emsdk_revision, emcc_version, source_before
        )
        write_build_evidence(staging, args.output, environment)
        for name in (WASM_NAME, REPORT_NAME, EVIDENCE_NAME):
            (staging / name).replace(args.output / name)


def main(argv=None):
    args = parse_args(argv)
    try:
        validate_paths(args)
        verify_build_environment()
        build(args)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Build failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
