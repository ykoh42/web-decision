"""Write a verifiable record beside a completed native TVM build."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

BUILD_ENV = Path(__file__).resolve().parent / "build-env"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(*command: str) -> str:
    return subprocess.run(
        command, check=True, capture_output=True, text=True
    ).stdout.strip()


def installed_python_packages() -> dict[str, str]:
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[re.sub(r"[-_.]+", "-", name).lower()] = distribution.version
    return dict(sorted(packages.items()))


def locked_python_packages(workspace_root: Path) -> dict[str, str]:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    result = subprocess.run(
        [
            "uv",
            "export",
            "--project",
            str(BUILD_ENV),
            "--format",
            "requirements.txt",
            "--no-header",
            "--no-emit-project",
            "--locked",
        ],
        check=True,
        capture_output=True,
        cwd=workspace_root,
        text=True,
    )
    packages = {}
    for line in result.stdout.splitlines():
        requirement_text = line.split("\\", 1)[0].strip()
        if not requirement_text or requirement_text.startswith(("#", "--")):
            continue
        requirement = Requirement(requirement_text)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        exact_versions = [
            specifier.version
            for specifier in requirement.specifier
            if specifier.operator == "=="
        ]
        if requirement.url or len(exact_versions) != 1:
            raise RuntimeError(
                f"Kanana build lock has a non-exact package pin: {requirement_text}"
            )
        packages[canonicalize_name(requirement.name)] = exact_versions[0]
    return dict(sorted(packages.items()))


def verify_locked_python_packages(
    expected: dict[str, str], installed: dict[str, str]
) -> dict[str, str]:
    if installed != expected:
        missing = {
            name: version
            for name, version in expected.items()
            if installed.get(name) != version
        }
        extra = {
            name: version for name, version in installed.items() if name not in expected
        }
        raise RuntimeError(
            "The active Python environment does not match scripts/kanana/build-env/uv.lock: "
            f"missingOrMismatched={missing}, extra={extra}. Run uv sync --locked --project scripts/kanana/build-env."
        )
    return installed


def verify_locked_python_environment(workspace_root: Path) -> dict[str, str]:
    expected_version = (
        (BUILD_ENV / ".python-version").read_text(encoding="utf-8").strip()
    )
    if platform.python_version() != expected_version:
        raise RuntimeError(
            f"Kanana build requires Python {expected_version}; found {platform.python_version()}."
        )

    expected = locked_python_packages(workspace_root)
    installed = installed_python_packages()
    return verify_locked_python_packages(expected, installed)


def find_shared_library(directories: list[Path], basename: str) -> Path:
    suffixes = (".dylib", ".so", ".dll")
    for directory in directories:
        for suffix in suffixes:
            path = directory / f"{basename}{suffix}"
            if path.is_file():
                return path.resolve(strict=True)
    raise RuntimeError(f"The active native TVM build is missing {basename}.")


def git_revision(path: Path) -> str:
    return command_output("git", "-C", str(path), "rev-parse", "HEAD")


def collect_native_tvm(workspace_root: Path) -> dict:
    import tvm
    import tvm_ffi

    toolchain_path = (
        workspace_root
        / "packages/web-decision/src/catalog/provenance/kanana-conversion-toolchain.json"
    )
    toolchain = json.loads(toolchain_path.read_text(encoding="utf-8"))
    tvm_source = workspace_root / ".toolchains/apache-tvm-src"
    emsdk = workspace_root / ".toolchains/emsdk"
    expected_tvm = toolchain["tvm"]
    expected_emscripten = toolchain["emscripten"]
    if git_revision(tvm_source) != expected_tvm["revision"]:
        raise RuntimeError(
            "The TVM source checkout differs from the Kanana toolchain pin."
        )
    if git_revision(emsdk) != expected_emscripten["revision"]:
        raise RuntimeError(
            "The Emscripten source checkout differs from the Kanana toolchain pin."
        )
    if tvm.__version__ != expected_tvm["version"]:
        raise RuntimeError(
            f"Kanana build requires TVM {expected_tvm['version']}; found {tvm.__version__}."
        )

    tvm_directories = [Path(path).resolve() for path in tvm.libinfo.package_lib_paths()]
    runtime_library = Path(tvm.libinfo.find_libtvm_runtime()).resolve(strict=True)
    compiler_library = find_shared_library(tvm_directories, "libtvm_compiler")
    ffi_library = Path(tvm_ffi.libinfo.find_libtvm_ffi()).resolve(strict=True)
    module_path = Path(tvm.__file__).resolve(strict=True)

    if expected_tvm["sourceChanges"] != []:
        raise RuntimeError("Kanana builds require unmodified pinned TVM source.")
    source_status = command_output(
        "git", "-C", str(tvm_source), "status", "--porcelain"
    )
    if source_status:
        raise RuntimeError(f"Kanana TVM source checkout is modified:\n{source_status}")

    runtime_assets = []
    for item in expected_tvm["wasmRuntimeAssets"]:
        path = tvm_source / "web/dist/wasm" / item["name"]
        actual = {
            "name": item["name"],
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if actual != item:
            raise RuntimeError(
                f"The built TVM WebAssembly runtime asset differs: {item['name']}"
            )
        runtime_assets.append(actual)

    emcc = emsdk / "upstream/emscripten/emcc"
    clang = emsdk / "upstream/bin/clang"
    wasm_ld = emsdk / "upstream/bin/wasm-ld"
    emcc_version = command_output(str(emcc), "--version").splitlines()[0]
    if expected_emscripten["version"] not in emcc_version:
        raise RuntimeError(
            f"Kanana build requires Emscripten {expected_emscripten['version']}."
        )

    return {
        "pythonPackage": {
            "name": "apache-tvm",
            "version": importlib.metadata.version("apache-tvm"),
            "moduleSha256": sha256_file(module_path),
            "compilerLibrarySha256": sha256_file(compiler_library),
            "runtimeLibrarySha256": sha256_file(runtime_library),
            "ffiPackageVersion": importlib.metadata.version("apache-tvm-ffi"),
            "ffiLibrarySha256": sha256_file(ffi_library),
        },
        "source": {
            "revision": expected_tvm["revision"],
            "patches": [],
            "wasmRuntimeAssets": runtime_assets,
        },
        "emscripten": {
            "revision": expected_emscripten["revision"],
            "version": expected_emscripten["version"],
            "emccVersion": emcc_version,
            "emccSha256": sha256_file(emcc),
            "clangSha256": sha256_file(clang),
            "wasmLdSha256": sha256_file(wasm_ld),
        },
    }


def collect_build_environment(workspace_root: Path) -> dict:
    python_path = Path(sys.executable).resolve(strict=True)
    node_path = Path(shutil.which("node") or "").resolve(strict=True)
    pnpm_path = Path(shutil.which("pnpm") or "").resolve(strict=True)
    python_lock = BUILD_ENV / "uv.lock"
    python_project = BUILD_ENV / "pyproject.toml"
    python_version_file = BUILD_ENV / ".python-version"
    pnpm_lock = workspace_root / "pnpm-lock.yaml"
    package_json = workspace_root / "package.json"
    system = platform.system()
    os_build = (
        command_output("sw_vers", "-buildVersion")
        if system == "Darwin"
        else platform.version()
    )

    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "compiler": platform.python_compiler(),
            "executableSha256": sha256_file(python_path),
            "uvVersion": command_output("uv", "--version"),
        },
        "pythonBuild": {
            "project": "scripts/kanana/build-env/pyproject.toml",
            "projectSha256": sha256_file(python_project),
            "versionFile": "scripts/kanana/build-env/.python-version",
            "versionFileSha256": sha256_file(python_version_file),
            "lockfile": "scripts/kanana/build-env/uv.lock",
            "lockfileSha256": sha256_file(python_lock),
            "packages": verify_locked_python_environment(workspace_root),
        },
        "host": {
            "system": system,
            "version": platform.mac_ver()[0]
            if system == "Darwin"
            else platform.release(),
            "kernel": platform.release(),
            "build": os_build,
            "machine": platform.machine(),
        },
        "javascript": {
            "nodeVersion": command_output("node", "--version"),
            "nodeExecutableSha256": sha256_file(node_path),
            "pnpmVersion": command_output("pnpm", "--version"),
            "pnpmExecutableSha256": sha256_file(pnpm_path),
            "lockfile": "pnpm-lock.yaml",
            "lockfileSha256": sha256_file(pnpm_lock),
            "projectSha256": sha256_file(package_json),
        },
        "nativeTvm": collect_native_tvm(workspace_root),
    }


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
            "buildEnvironment": collect_build_environment(
                Path(__file__).resolve().parents[2]
            ),
        }
    )
    return evidence
