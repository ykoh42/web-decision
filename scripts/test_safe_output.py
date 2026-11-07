import errno
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from safe_output import (
    atomic_write,
    create_new_file,
    create_temporary_directory,
    open_directory,
    promote_directory,
    promote_file,
    remove_tree,
    require_absent,
)


class SafeOutputTests(unittest.TestCase):
    def test_staging_directory_is_private_and_descriptor_anchored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary, "root")
            (root / "models").mkdir(parents=True)
            models_fd = open_directory(root, "models")
            try:
                name, stage_fd = create_temporary_directory(models_fd, ".stage")
                try:
                    self.assertEqual(stat.S_IMODE(os.fstat(stage_fd).st_mode) & 0o077, 0)
                    (root / "models" / name / "item").write_text("staged")
                finally:
                    os.close(stage_fd)
                remove_tree(models_fd, name)
                self.assertFalse((root / "models" / name).exists())
            finally:
                os.close(models_fd)

    def test_atomic_write_replaces_a_symlink_without_touching_its_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary, "root")
            outside = Path(temporary, "outside")
            output = root / "models" / "kev-0.6b-onnx" / "onnx"
            output.mkdir(parents=True)
            outside.mkdir()
            target = outside / "keep.txt"
            target.write_text("preserve me")
            (output / "model.onnx").symlink_to(target)

            directory_fd = open_directory(root, "models/kev-0.6b-onnx/onnx")
            try:
                atomic_write(directory_fd, "model.onnx", lambda stream: stream.write(b"new graph"))
            finally:
                os.close(directory_fd)

            self.assertEqual(target.read_text(), "preserve me")
            self.assertFalse((output / "model.onnx").is_symlink())
            self.assertEqual((output / "model.onnx").read_bytes(), b"new graph")

    def test_require_absent_rejects_a_symlinked_output_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary, "root")
            outside = Path(temporary, "outside")
            output = root / "public"
            output.mkdir(parents=True)
            outside.mkdir()
            target = outside / "keep.wasm"
            target.write_bytes(b"preserve me")
            (output / "runtime.wasm").symlink_to(target)

            directory_fd = open_directory(root, "public")
            try:
                with self.assertRaises(FileExistsError):
                    require_absent(directory_fd, "runtime.wasm")
            finally:
                os.close(directory_fd)

            self.assertEqual(target.read_bytes(), b"preserve me")

    def test_create_new_file_rejects_a_symlink_without_touching_its_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary, "root")
            outside = Path(temporary, "outside")
            output = root / "reports"
            output.mkdir(parents=True)
            outside.mkdir()
            target = outside / "report.json"
            target.write_text("preserve me")
            (output / "report.json").symlink_to(target)

            directory_fd = open_directory(root, "reports")
            try:
                with self.assertRaises(FileExistsError):
                    create_new_file(directory_fd, "report.json", lambda stream: stream.write(b"new report"))
            finally:
                os.close(directory_fd)

            self.assertEqual(target.read_text(), "preserve me")

    def test_promotion_rejects_static_symlink_destinations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary, "root")
            models = root / "models"
            public = root / "public"
            outside = Path(temporary, "outside")
            stage = models / ".stage"
            (stage / "cache").mkdir(parents=True)
            (stage / "cache" / "weights.bin").write_bytes(b"cache")
            (stage / "runtime.wasm").write_bytes(b"wasm")
            public.mkdir(parents=True)
            outside.mkdir()
            wasm_target = outside / "keep.wasm"
            cache_target = outside / "keep-cache"
            wasm_target.write_bytes(b"preserve wasm")
            cache_target.mkdir()
            (cache_target / "sentinel").write_text("preserve cache")
            (public / "runtime.wasm").symlink_to(wasm_target)
            (models / "cache").symlink_to(cache_target, target_is_directory=True)

            models_fd = open_directory(root, "models")
            public_fd = open_directory(root, "public")
            stage_fd = open_directory(root, "models/.stage")
            try:
                with self.assertRaises(FileExistsError):
                    promote_file(stage_fd, "runtime.wasm", public_fd, "runtime.wasm")
                with self.assertRaises(FileExistsError):
                    promote_directory(stage_fd, "cache", models_fd, "cache")
            finally:
                os.close(stage_fd)
                os.close(public_fd)
                os.close(models_fd)

            self.assertEqual(wasm_target.read_bytes(), b"preserve wasm")
            self.assertEqual((cache_target / "sentinel").read_text(), "preserve cache")

    def test_promotion_uses_open_descriptors_after_parent_paths_are_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary, "root")
            outside = Path(temporary, "outside")
            models = root / "models"
            public = root / "public"
            (models / ".stage" / "cache").mkdir(parents=True)
            (models / ".stage" / "cache" / "weights.bin").write_bytes(b"cache")
            (models / ".stage" / "runtime.wasm").write_bytes(b"wasm")
            public.mkdir(parents=True)
            outside.mkdir()
            sentinel = outside / "sentinel"
            sentinel.write_text("preserve me")

            models_fd = open_directory(root, "models")
            public_fd = open_directory(root, "public")
            stage_fd = open_directory(root, "models/.stage")
            models_saved = root / "models-opened"
            public_saved = root / "public-opened"
            os.rename(models, models_saved)
            os.rename(public, public_saved)
            models.symlink_to(outside, target_is_directory=True)
            public.symlink_to(outside, target_is_directory=True)
            try:
                promote_file(stage_fd, "runtime.wasm", public_fd, "runtime.wasm")
                promote_directory(stage_fd, "cache", models_fd, "cache")
            finally:
                os.close(stage_fd)
                os.close(public_fd)
                os.close(models_fd)

            self.assertEqual((public_saved / "runtime.wasm").read_bytes(), b"wasm")
            self.assertEqual((models_saved / "cache" / "weights.bin").read_bytes(), b"cache")
            self.assertEqual(sentinel.read_text(), "preserve me")
            self.assertFalse((outside / "runtime.wasm").exists())
            self.assertFalse((outside / "cache").exists())

    def test_directory_promotion_copies_when_staging_is_on_another_filesystem(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary, "root")
            models = root / "models"
            stage = root / ".stage"
            models.mkdir(parents=True)
            (stage / "cache" / "shards").mkdir(parents=True)
            (stage / "cache" / "tensor-cache.json").write_text("manifest")
            (stage / "cache" / "shards" / "part.bin").write_bytes(b"weights")

            models_fd = open_directory(root, "models")
            stage_fd = open_directory(root, ".stage")
            try:
                with patch("safe_output.os.rename", side_effect=OSError(errno.EXDEV, "cross-device")):
                    promote_directory(stage_fd, "cache", models_fd, "cache")
            finally:
                os.close(stage_fd)
                os.close(models_fd)

            self.assertEqual((models / "cache" / "tensor-cache.json").read_text(), "manifest")
            self.assertEqual((models / "cache" / "shards" / "part.bin").read_bytes(), b"weights")

    def test_failed_cross_filesystem_copy_removes_restrictive_partial_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary, "root")
            models = root / "models"
            stage = root / ".stage"
            restrictive = stage / "cache" / "a-restrictive"
            models.mkdir(parents=True)
            restrictive.mkdir(parents=True)
            (restrictive / "part.bin").write_bytes(b"partial")
            (stage / "cache" / "z-fail.bin").write_bytes(b"failure trigger")
            restrictive.chmod(0o500)

            models_fd = open_directory(root, "models")
            stage_fd = open_directory(root, ".stage")
            original_listdir = os.listdir

            def sorted_listdir(path):
                return sorted(original_listdir(path))

            original_promote_file = promote_file

            def fail_on_late_file(source_fd, source_name, target_fd, target_name):
                if source_name == "z-fail.bin":
                    raise OSError(errno.EIO, "injected copy failure")
                return original_promote_file(source_fd, source_name, target_fd, target_name)

            try:
                with patch("safe_output.os.rename", side_effect=OSError(errno.EXDEV, "cross-device")):
                    with patch("safe_output.os.listdir", side_effect=sorted_listdir):
                        with patch("safe_output.promote_file", side_effect=fail_on_late_file):
                            with self.assertRaises(OSError) as raised:
                                promote_directory(stage_fd, "cache", models_fd, "cache")
                self.assertEqual(raised.exception.errno, errno.EIO)
                self.assertFalse((models / "cache").exists())
                self.assertEqual(list(models.iterdir()), [])
            finally:
                os.close(stage_fd)
                os.close(models_fd)

    def test_remove_tree_opens_a_restrictive_top_level_directory_safely(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary, "root")
            models = root / "models"
            cache = models / "cache"
            cache.mkdir(parents=True)
            (cache / "weights.bin").write_bytes(b"weights")
            cache.chmod(0o000)

            models_fd = open_directory(root, "models")
            try:
                remove_tree(models_fd, "cache")
            finally:
                os.close(models_fd)

            self.assertFalse(cache.exists())

    def test_wasm_promotion_failure_removes_restrictive_published_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary, "root")
            models = root / "models"
            public = root / "public"
            stage = root / ".stage"
            cache = stage / "cache"
            models.mkdir(parents=True)
            public.mkdir()
            cache.mkdir(parents=True)
            (cache / "weights.bin").write_bytes(b"weights")
            (stage / "runtime.wasm").write_bytes(b"wasm")
            cache.chmod(0o500)

            models_fd = open_directory(root, "models")
            public_fd = open_directory(root, "public")
            stage_fd = open_directory(root, ".stage")
            try:
                with patch("safe_output.os.rename", side_effect=OSError(errno.EXDEV, "cross-device")):
                    promote_directory(stage_fd, "cache", models_fd, "cache")
                self.assertEqual(stat.S_IMODE((models / "cache").stat().st_mode), 0o500)

                with patch("safe_output.os.link", side_effect=FileExistsError("injected collision")):
                    with self.assertRaises(FileExistsError):
                        promote_file(stage_fd, "runtime.wasm", public_fd, "runtime.wasm")
                remove_tree(models_fd, "cache")
            finally:
                os.close(stage_fd)
                os.close(public_fd)
                os.close(models_fd)

            self.assertFalse((models / "cache").exists())

    def test_open_directory_rejects_a_symlinked_reports_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary, "root")
            outside = Path(temporary, "outside")
            root.mkdir()
            outside.mkdir()
            (root / "reports").symlink_to(outside, target_is_directory=True)

            with self.assertRaises(OSError):
                open_directory(root, "reports")

    def test_open_directory_rejects_symlinked_parent_components(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary, "root")
            outside = Path(temporary, "outside")
            root.mkdir()
            outside.mkdir()
            (root / "models").symlink_to(outside, target_is_directory=True)

            with self.assertRaises(OSError):
                open_directory(root, "models/kev-0.6b-onnx/onnx")

    def test_open_directory_rejects_symlinked_model_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary, "root")
            outside = Path(temporary, "outside")
            (root / "models").mkdir(parents=True)
            outside.mkdir()
            (root / "models" / "kev-0.6b-onnx").symlink_to(outside, target_is_directory=True)

            with self.assertRaises(OSError):
                open_directory(root, "models/kev-0.6b-onnx/onnx")

    def test_open_directory_rejects_absolute_and_parent_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary, "root")
            root.mkdir()

            for relative_path in (temporary, "../outside"):
                with self.subTest(relative_path=relative_path):
                    with self.assertRaises(ValueError):
                        open_directory(root, relative_path)


if __name__ == "__main__":
    unittest.main()
