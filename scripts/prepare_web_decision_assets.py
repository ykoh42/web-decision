#!/usr/bin/env python3
"""Package the verified, locally built Kev Q4 TVM artifacts for a web server."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

if __package__:
    from .safe_output import (
        create_new_file,
        create_temporary_directory,
        promote_directory,
        remove_tree,
        require_absent,
    )
else:
    from safe_output import (
        create_new_file,
        create_temporary_directory,
        promote_directory,
        remove_tree,
        require_absent,
    )

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA256 = "777335dbc052e2e02ec7591f6eefcfb29e2c28caf52075e492d226fc27e0824b"
EXPECTED_SOURCE_HASHES = {
    "onnx/model_q4f16.onnx": "26e059ddd954efd9897772b098db10b9eeda9b873984dc1c73b32a9ce177c7c8",
    "config.json": "40ce1c875acf016ab7742ecdc45d3bad0fe8c51d3ead0ca2263ec5eb4e96f42d",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5",
}
EXPECTED_NOTICE_HASHES = {
    "PROVENANCE.md": "b649e445aba65d3aa12c855c4d2e5ecfbef608100bfdaead3b0e46cfee4dfc9a",
    "MODEL-APACHE-2.0.txt": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    "EMSCRIPTEN-LICENSE.txt": "620a78084fc7ca97c0b5dea9abf891f3ffcadfdbf305276f099c9c4e12fc1d86",
    "MUSL-COPYRIGHT.txt": "f9bc4423732350eb0b3f7ed7e91d530298476f8fec0c6c427a1c04ade22655af",
    "LIBCXX-LICENSE.txt": "539dd7aed86e8a4f12cbdd0e6c50c189c7d74847e4fecc64ce2c6ee3a01da38b",
    "LIBCXXABI-LICENSE.txt": "e2b35be49f7284a45b7baca8fc7b3ab7440e7902392b2528a457816b5bb2a15c",
    "EMSCRIPTEN-COMPILER-RT-LICENSE.txt": "1a8f1058753f1ba890de984e48f0242a3a5c29a6a8f2ed9fd813f36985387e8d",
}


def digest_stream(stream):
    result = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(chunk)
    return result.hexdigest()


def digest(path):
    with path.open("rb") as stream:
        return digest_stream(stream)


def canonical_manifest_sha256(value):
    canonical = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_manifest_pin(root):
    try:
        metadata = json.loads(
            (root / "packages/web-decision/src/kev-preview-model.json").read_text()
        )
        expected = metadata["assets"]["manifestSha256"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise SystemExit(
            f"Could not read the current Kev manifest pin: {error}"
        ) from error
    if not is_sha256(expected):
        raise SystemExit(
            "The current Kev model metadata contains an invalid manifest SHA-256"
        )
    return expected


def is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def read_build_report(root):
    path = root / "reports/tvm-q4-build-dynamic.json"
    try:
        build = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"Could not read the dynamic TVM build report: {error}"
        ) from error
    if (
        not isinstance(build, dict)
        or build.get("status") != "success"
        or build.get("inputShape") != [1, "sequence_length"]
        or build.get("numberOfParameters") != 394
    ):
        raise SystemExit("The verified dynamic build report is missing or incompatible")
    return build


def verify_dynamic_cache(root):
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/verify_q4_cache.py"),
            "--dynamic",
            "--check-only",
        ],
        cwd=root,
        check=True,
    )


def safe_shard_name(value):
    if (
        not isinstance(value, str)
        or not value
        or value in (".", "..")
        or "/" in value
        or "\\" in value
        or not value.endswith(".bin")
    ):
        raise SystemExit(f"Unsafe shard name: {value}")
    return value


def collect_asset_sources(root):
    source = root / "models/kev-0.6b-onnx"
    cache = root / "models/tvm-kev-q4-dynamic"
    wasm = root / "public/tvm-kev-q4-dynamic.wasm"
    cache_path = cache / "tensor-cache.json"
    try:
        cache_manifest = json.loads(cache_path.read_text())
        records = cache_manifest["records"]
        num_params = cache_manifest["metadata"]["numParams"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise SystemExit(
            f"Could not read the verified tensor cache manifest: {error}"
        ) from error
    if num_params != 394 or not isinstance(records, list):
        raise SystemExit("Unexpected number of model parameters")
    sources = {
        "model.wasm": wasm,
        "tokenizer.json": source / "tokenizer.json",
        "tokenizer_config.json": source / "tokenizer_config.json",
        "model_config.json": source / "config.json",
        "weights/tensor-cache.json": cache_path,
    }
    for shard in records:
        name = safe_shard_name(
            shard.get("dataPath") if isinstance(shard, dict) else None
        )
        sources[f"weights/{name}"] = cache / name
    return source, cache, wasm, sources


def verify_pinned_file(path, expected, description):
    if digest(path) != expected:
        raise SystemExit(
            f"{description} does not match the pinned SHA-256: {path.name}"
        )


def verify_source_pins(source):
    verify_pinned_file(
        source / "onnx/model_q4f16.onnx_data",
        CHECKPOINT_SHA256,
        "Source checkpoint",
    )
    for name, expected in EXPECTED_SOURCE_HASHES.items():
        verify_pinned_file(source / name, expected, "Source model file")


def verify_build_artifacts(root, build, wasm, cache):
    if build.get("tvmVersion") != "0.27.0.post1" or build.get("wasmSha256") != digest(
        wasm
    ):
        raise SystemExit(
            "The WebAssembly executable does not match the pinned TVM build report; rebuild it."
        )
    if (
        build.get("embeddedBuildRoot") is not False
        or str(root).encode() in wasm.read_bytes()
    ):
        raise SystemExit(
            "The WebAssembly executable contains an absolute build directory; rebuild it with the preview setup script."
        )
    if build.get("tensorCacheManifestSha256") != digest(cache / "tensor-cache.json"):
        raise SystemExit(
            "The tensor cache does not match the TVM build report; rebuild it."
        )


def collect_notice_sources(root):
    tvm = root / ".toolchains/apache-tvm-src"
    emscripten = root / ".toolchains/emsdk/upstream/emscripten"
    sources = {
        "PROVENANCE.md": root / "scripts/asset-notices/PROVENANCE.md",
        "MODEL-APACHE-2.0.txt": root
        / "packages/web-decision/assets/experimental-kev-upstream/notices/MODEL-APACHE-2.0.txt",
        "EMSCRIPTEN-LICENSE.txt": emscripten / "LICENSE",
        "MUSL-COPYRIGHT.txt": emscripten / "system/lib/libc/musl/COPYRIGHT",
        "LIBCXX-LICENSE.txt": emscripten / "system/lib/libcxx/LICENSE.TXT",
        "LIBCXXABI-LICENSE.txt": emscripten / "system/lib/libcxxabi/LICENSE.TXT",
        "EMSCRIPTEN-COMPILER-RT-LICENSE.txt": emscripten
        / "system/lib/compiler-rt/LICENSE.TXT",
        "TVM-LICENSE.txt": tvm / "LICENSE",
        "TVM-NOTICE.txt": tvm / "NOTICE",
        "TVM-FFI-LICENSE.txt": tvm / "3rdparty/tvm-ffi/LICENSE",
        "TVM-FFI-NOTICE.txt": tvm / "3rdparty/tvm-ffi/NOTICE",
        "DLPACK-LICENSE.txt": tvm / "3rdparty/tvm-ffi/3rdparty/dlpack/LICENSE",
        "COMPILER-RT-LICENSE.txt": tvm / "licenses/LICENSE.builtin_fp16.txt",
    }
    hashes = {}
    for name, path in sources.items():
        if not path.is_file():
            raise SystemExit(f"Missing required notice source: {name}")
        actual = digest(path)
        expected = EXPECTED_NOTICE_HASHES.get(name)
        if expected is not None and actual != expected:
            raise SystemExit(f"Unexpected notice source: {name}")
        hashes[name] = actual
    header = (tvm / "3rdparty/compiler-rt/builtin_fp16.h").read_bytes()
    if not header.startswith(b"/*") or b"*/" not in header:
        raise SystemExit("Unexpected compiler-rt header notice")
    return sources, hashes, header[: header.index(b"*/") + 2] + b"\n"


def collect_asset_metadata(sources):
    files = {}
    for name, path in sources.items():
        if not path.is_file():
            raise SystemExit(f"Missing required asset source: {name}")
        files[name] = {"sha256": digest(path), "bytes": path.stat().st_size}
    return files


def build_manifest(build, files):
    return {
        "formatVersion": 1,
        "model": "kev-0.6b-q4f16",
        "backend": "tvm-webgpu",
        "experimental": True,
        "maxStateTokens": 8192,
        "maxBranchTokens": 8192,
        "parameterCount": 394,
        "tvmVersion": build["tvmVersion"],
        "checkpoint": {
            "repository": "onnx-community/kev-0.6b-ONNX",
            "revision": "5a827091b3d58fe4fdfc02a5a5b2ce364047d678",
            "weightSha256": CHECKPOINT_SHA256,
        },
        "files": files,
    }


def verify_manifest_pin(manifest, expected_manifest_sha256):
    if canonical_manifest_sha256(manifest) != expected_manifest_sha256:
        raise SystemExit(
            "Prepared Kev 0.6B manifest does not match the reviewed package pin."
        )


def load_verified_inputs(root):
    expected_manifest_sha256 = read_manifest_pin(root)
    build = read_build_report(root)
    verify_dynamic_cache(root)
    source, cache, wasm, sources = collect_asset_sources(root)
    verify_source_pins(source)
    verify_build_artifacts(root, build, wasm, cache)
    notices, notice_hashes, header_notice = collect_notice_sources(root)
    files = collect_asset_metadata(sources)
    manifest = build_manifest(build, files)
    verify_manifest_pin(manifest, expected_manifest_sha256)
    return sources, files, manifest, notices, notice_hashes, header_notice


def reject_symlink_components(path):
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current.parent if component == ".." else current / component
        if current.is_symlink():
            raise SystemExit(f"Refusing output path with symlink component: {current}")


def safe_output_path(output):
    requested = output.expanduser()
    if not requested.is_absolute():
        requested = Path.cwd() / requested
    reject_symlink_components(requested)
    resolved = requested.resolve()
    if resolved.exists() or resolved.is_symlink():
        raise SystemExit(f"Refusing to overwrite existing path: {resolved}")
    if not resolved.name or "\\" in resolved.name:
        raise SystemExit("Output path must end with a safe directory name")
    return resolved


def open_or_create_output_parent(output):
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent = output.parent
    descriptor = os.open(Path(output.anchor), flags)
    try:
        for component in parent.relative_to(Path(output.anchor)).parts:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o777, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def ensure_stage_directory(parent_fd, name):
    try:
        os.mkdir(name, 0o777, dir_fd=parent_fd)
    except FileExistsError:
        pass
    return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)


def digest_at(directory_fd, name):
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    with os.fdopen(descriptor, "rb") as stream:
        return digest_stream(stream)


def copy_verified_file(source, directory_fd, name, expected_sha256, message):
    def copy(output):
        with source.open("rb") as input_stream:
            shutil.copyfileobj(input_stream, output, 1024 * 1024)

    create_new_file(directory_fd, name, copy)
    if digest_at(directory_fd, name) != expected_sha256:
        raise SystemExit(message)


def write_asset_files(stage_fd, sources, files):
    weights_fd = ensure_stage_directory(stage_fd, "weights")
    try:
        for name, source in sources.items():
            if name.startswith("weights/"):
                target_fd = weights_fd
                target_name = name.removeprefix("weights/")
            else:
                target_fd = stage_fd
                target_name = name
            expected = files[name]["sha256"]
            copy_verified_file(
                source,
                target_fd,
                target_name,
                expected,
                f"Copy failed integrity check: {name}",
            )
    finally:
        os.close(weights_fd)


def write_notice_files(stage_fd, notices, hashes, header_notice):
    notices_fd = ensure_stage_directory(stage_fd, "notices")
    try:
        for name, source in notices.items():
            copy_verified_file(
                source,
                notices_fd,
                name,
                hashes[name],
                f"Notice copy failed integrity check: {name}",
            )
        create_new_file(
            notices_fd,
            "COMPILER-RT-HEADER-NOTICE.txt",
            lambda output: output.write(header_notice),
        )
    finally:
        os.close(notices_fd)


def write_stage(stage_fd, sources, files, manifest, notices, hashes, header_notice):
    write_asset_files(stage_fd, sources, files)
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    create_new_file(
        stage_fd, "manifest.json", lambda output: output.write(manifest_bytes)
    )
    # Notices stay beside the assets and outside `manifest.files`, which totals inference bytes.
    write_notice_files(stage_fd, notices, hashes, header_notice)


def default_directory_mode():
    mask = os.umask(0)
    os.umask(mask)
    return 0o777 & ~mask


def cleanup_stage(parent_fd, stage_name):
    try:
        remove_tree(parent_fd, stage_name)
    except FileNotFoundError:
        pass


def write_output_atomically(
    output, sources, files, manifest, notices, hashes, header_notice
):
    parent_fd = open_or_create_output_parent(output)
    stage_name = None
    stage_fd = None
    try:
        require_absent(parent_fd, output.name)
        stage_name, stage_fd = create_temporary_directory(
            parent_fd, f".{output.name}.staging"
        )
        write_stage(stage_fd, sources, files, manifest, notices, hashes, header_notice)
        os.fchmod(stage_fd, default_directory_mode())
        promote_directory(parent_fd, stage_name, parent_fd, output.name)
        stage_name = None
    except BaseException:
        if stage_name is not None:
            cleanup_stage(parent_fd, stage_name)
        raise
    finally:
        if stage_fd is not None:
            os.close(stage_fd)
        os.close(parent_fd)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory; must not already exist or traverse symlinks",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    output = safe_output_path(args.output)
    sources, files, manifest, notices, notice_hashes, header_notice = (
        load_verified_inputs(ROOT)
    )
    write_output_atomically(
        output,
        sources,
        files,
        manifest,
        notices,
        notice_hashes,
        header_notice,
    )
    total_bytes = sum(entry["bytes"] for entry in files.values())
    print(
        f"Prepared {len(files)} verified files at {output}; {total_bytes:,} bytes",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
