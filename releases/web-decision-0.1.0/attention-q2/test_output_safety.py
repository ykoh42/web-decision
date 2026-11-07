from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from output_safety import create_fresh_output_directory, write_new_text_file


class OutputSafetyTest(unittest.TestCase):
    def test_creates_a_new_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory).resolve() / "generated"

            self.assertEqual(create_fresh_output_directory(output), output)
            self.assertTrue(output.is_dir())

    def test_refuses_an_existing_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory).resolve() / "generated"
            output.mkdir()

            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                create_fresh_output_directory(output)

    def test_refuses_a_dangling_output_symlink(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            output = root / "generated"
            output.symlink_to(root / "missing", target_is_directory=True)

            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                create_fresh_output_directory(output)

    def test_refuses_a_symlinked_output_parent(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            actual_parent = root / "actual"
            actual_parent.mkdir()
            output_parent = root / "alias"
            output_parent.symlink_to(actual_parent, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "Output parent must not be a symlink"):
                create_fresh_output_directory(output_parent / "generated")
            self.assertFalse((actual_parent / "generated").exists())

    def test_write_refuses_to_replace_an_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory).resolve() / "generated.txt"
            output.write_text("existing", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                write_new_text_file(output, "replacement")
            self.assertEqual(output.read_text(encoding="utf-8"), "existing")


if __name__ == "__main__":
    unittest.main()
