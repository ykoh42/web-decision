from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path

from build_inputs import verify_pinned_file


class PinnedInputTest(unittest.TestCase):
    def test_accepts_a_regular_file_matching_size_and_hash(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "input.bin"
            contents = b"pinned input"
            path.write_bytes(contents)

            self.assertEqual(
                verify_pinned_file(
                    path,
                    len(contents),
                    hashlib.sha256(contents).hexdigest(),
                    "input",
                ),
                hashlib.sha256(contents).hexdigest(),
            )

    def test_rejects_wrong_size_before_hashing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "input.bin"
            path.write_bytes(b"too large")

            with self.assertRaisesRegex(ValueError, "byte size"):
                verify_pinned_file(path, 1, "0" * 64, "input")

    def test_rejects_symlink_inputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.bin"
            link = root / "input.bin"
            source.write_bytes(b"input")
            link.symlink_to(source)

            with self.assertRaisesRegex(ValueError, "regular non-symlink file"):
                verify_pinned_file(link, 5, hashlib.sha256(b"input").hexdigest(), "input")

    @unittest.skipUnless(hasattr(os, "mkfifo"), "named pipes are unavailable")
    def test_rejects_fifo_inputs_without_reading_them(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "input.pipe"
            os.mkfifo(path)

            with self.assertRaisesRegex(ValueError, "regular non-symlink file"):
                verify_pinned_file(path, 1, "0" * 64, "input")


if __name__ == "__main__":
    unittest.main()
