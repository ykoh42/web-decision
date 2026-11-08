"""Verify the asset packager's pins and atomic, symlink-safe output handling."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import prepare_web_decision_assets as packager


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class PrepareWebDecisionAssetsTests(unittest.TestCase):
    def test_collects_and_verifies_pinned_notice_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tvm = root / ".toolchains/apache-tvm-src"
            emscripten = root / ".toolchains/emsdk/upstream/emscripten"
            paths = {
                "PROVENANCE.md": "scripts/asset-notices/PROVENANCE.md",
                "MODEL-APACHE-2.0.txt": "packages/web-decision/assets/experimental-kev-upstream/notices/MODEL-APACHE-2.0.txt",
                "EMSCRIPTEN-LICENSE.txt": emscripten / "LICENSE",
                "MUSL-COPYRIGHT.txt": emscripten / "system/lib/libc/musl/COPYRIGHT",
                "LIBCXX-LICENSE.txt": emscripten / "system/lib/libcxx/LICENSE.TXT",
                "LIBCXXABI-LICENSE.txt": emscripten
                / "system/lib/libcxxabi/LICENSE.TXT",
                "EMSCRIPTEN-COMPILER-RT-LICENSE.txt": emscripten
                / "system/lib/compiler-rt/LICENSE.TXT",
                "TVM-LICENSE.txt": tvm / "LICENSE",
                "TVM-NOTICE.txt": tvm / "NOTICE",
                "TVM-FFI-LICENSE.txt": tvm / "3rdparty/tvm-ffi/LICENSE",
                "TVM-FFI-NOTICE.txt": tvm / "3rdparty/tvm-ffi/NOTICE",
                "DLPACK-LICENSE.txt": tvm / "3rdparty/tvm-ffi/3rdparty/dlpack/LICENSE",
                "COMPILER-RT-LICENSE.txt": tvm / "licenses/LICENSE.builtin_fp16.txt",
            }
            expected = {}
            for name, location in paths.items():
                path = root / location
                path.parent.mkdir(parents=True, exist_ok=True)
                data = (
                    (packager.ROOT / location).read_bytes()
                    if isinstance(location, str)
                    else name.encode()
                )
                path.write_bytes(data)
                if name not in {"PROVENANCE.md", "MODEL-APACHE-2.0.txt"}:
                    expected[name] = sha256(data)
            header = tvm / "3rdparty/compiler-rt/builtin_fp16.h"
            header.parent.mkdir(parents=True, exist_ok=True)
            header.write_bytes(b"/* license */\nint x;\n")

            with patch.dict(packager.EXPECTED_NOTICE_HASHES, expected):
                sources, hashes, extracted = packager.collect_notice_sources(root)
                self.assertEqual(
                    sources, {name: root / location for name, location in paths.items()}
                )
                self.assertEqual(extracted, b"/* license */\n")
                self.assertEqual(
                    hashes["PROVENANCE.md"],
                    packager.EXPECTED_NOTICE_HASHES["PROVENANCE.md"],
                )
                self.assertEqual(
                    hashes["MODEL-APACHE-2.0.txt"],
                    packager.EXPECTED_NOTICE_HASHES["MODEL-APACHE-2.0.txt"],
                )
                for name in ("PROVENANCE.md", "MODEL-APACHE-2.0.txt"):
                    path = sources[name]
                    original = path.read_bytes()
                    path.write_bytes(original + b"tampered")
                    with (
                        self.subTest(name=name),
                        self.assertRaisesRegex(SystemExit, "Unexpected notice source"),
                    ):
                        packager.collect_notice_sources(root)
                    path.write_bytes(original)

    def test_manifest_pin_requires_the_canonical_manifest_digest(self):
        manifest = {"files": {"model.wasm": {"sha256": sha256(b"model")}}}
        pin = packager.canonical_manifest_sha256(manifest)

        packager.verify_manifest_pin(manifest, pin)
        with self.assertRaisesRegex(SystemExit, "reviewed package pin"):
            packager.verify_manifest_pin(manifest, "0" * 64)

    def test_pin_reader_rejects_malformed_model_metadata_pin(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "packages/web-decision/src/kev-preview-model.json"
            metadata.parent.mkdir(parents=True)
            metadata.write_text(json.dumps({"assets": {"manifestSha256": "bad"}}))

            with self.assertRaisesRegex(SystemExit, "invalid manifest SHA-256"):
                packager.read_manifest_pin(root)

    def test_shard_names_reject_paths_and_non_binary_names(self):
        for name in (
            "../outside.bin",
            "nested/weights.bin",
            r"nested\weights.bin",
            "weights.json",
        ):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(SystemExit, "Unsafe shard"),
            ):
                packager.safe_shard_name(name)
        self.assertEqual(packager.safe_shard_name("shard-001.bin"), "shard-001.bin")

    def test_output_path_rejects_existing_targets_and_symlink_components(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside"
            outside.mkdir()
            sentinel = outside / "keep.txt"
            sentinel.write_text("preserve")
            existing = root / "existing"
            existing.mkdir()
            linked_parent = root / "linked-parent"
            linked_parent.symlink_to(outside, target_is_directory=True)
            linked_target = root / "linked-target"
            linked_target.symlink_to(sentinel)

            for output in (existing, linked_parent / "package", linked_target):
                with self.subTest(output=output), self.assertRaises(SystemExit):
                    packager.safe_output_path(output)
            self.assertEqual(sentinel.read_text(), "preserve")

    def test_packaging_publishes_verified_files_and_notices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "source"
            source_dir.mkdir()
            model = source_dir / "model.wasm"
            shard = source_dir / "shard.bin"
            notice = source_dir / "NOTICE.txt"
            model.write_bytes(b"verified model")
            shard.write_bytes(b"verified weights")
            notice.write_bytes(b"license notice")
            sources = {"model.wasm": model, "weights/shard.bin": shard}
            files = {
                name: {"sha256": packager.digest(path), "bytes": path.stat().st_size}
                for name, path in sources.items()
            }
            manifest = {"formatVersion": 1, "files": files}
            output = root / "new-parent/package"

            packager.write_output_atomically(
                output,
                sources,
                files,
                manifest,
                {"NOTICE.txt": notice},
                {"NOTICE.txt": packager.digest(notice)},
                b"header notice\n",
            )

            self.assertEqual((output / "model.wasm").read_bytes(), model.read_bytes())
            self.assertEqual(
                (output / "weights/shard.bin").read_bytes(), shard.read_bytes()
            )
            self.assertEqual(
                (output / "manifest.json").read_text(),
                json.dumps(manifest, indent=2) + "\n",
            )
            self.assertEqual(
                (output / "notices/NOTICE.txt").read_bytes(), notice.read_bytes()
            )
            self.assertEqual(
                (output / "notices/COMPILER-RT-HEADER-NOTICE.txt").read_bytes(),
                b"header notice\n",
            )
            self.assertEqual(
                sorted(
                    path.relative_to(output).as_posix() for path in output.rglob("*")
                ),
                [
                    "manifest.json",
                    "model.wasm",
                    "notices",
                    "notices/COMPILER-RT-HEADER-NOTICE.txt",
                    "notices/NOTICE.txt",
                    "weights",
                    "weights/shard.bin",
                ],
            )

    def test_failed_copy_cleans_the_private_stage_without_publishing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"verified")
            missing = root / "missing.bin"
            files = {
                "model.wasm": {"sha256": packager.digest(source), "bytes": 8},
                "weights/missing.bin": {"sha256": "0" * 64, "bytes": 1},
            }
            output_parent = root / "new-parent"
            output = output_parent / "package"

            with self.assertRaises(FileNotFoundError):
                packager.write_output_atomically(
                    output,
                    {"model.wasm": source, "weights/missing.bin": missing},
                    files,
                    {"files": files},
                    {},
                    {},
                    b"header\n",
                )

            self.assertFalse(output.exists())
            self.assertEqual(list(output_parent.iterdir()), [])

    def test_copy_hash_mismatch_cleans_the_private_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"actual bytes")
            output_parent = root / "new-parent"
            output = output_parent / "package"

            with self.assertRaisesRegex(SystemExit, "Copy failed integrity check"):
                packager.write_output_atomically(
                    output,
                    {"model.wasm": source},
                    {"model.wasm": {"sha256": "0" * 64, "bytes": 12}},
                    {"files": {}},
                    {},
                    {},
                    b"header\n",
                )

            self.assertFalse(output.exists())
            self.assertEqual(list(output_parent.iterdir()), [])

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "package"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("preserve")

            with self.assertRaises(FileExistsError):
                packager.write_output_atomically(output, {}, {}, {}, {}, {}, b"")

            self.assertEqual(sentinel.read_text(), "preserve")
            self.assertEqual(list(root.iterdir()), [output])


if __name__ == "__main__":
    unittest.main()
