"""Keep the complete source range map bound to its verified checkpoint bytes."""

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import source_inventory


class SourceInventoryTest(unittest.TestCase):
    def test_rejects_a_stale_range_digest_with_the_same_checkpoint_identity(self):
        current = hashlib.sha256(b"current weights").hexdigest()
        stale = hashlib.sha256(b"previous weights").hexdigest()
        inventory = {
            "base": {
                "revision": source_inventory.BASE_REVISION,
                "sha256": source_inventory.BASE_FILES["model.safetensors"],
                "chunks": [[256, 15, current]],
            },
            "adapter": {
                "revision": source_inventory.ADAPTER_REVISION,
                "sha256": source_inventory.ADAPTER_FILES["adapter_model.safetensors"],
            },
        }
        original = json.dumps(inventory).encode()
        pin = hashlib.sha256(original).hexdigest()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "source-inventory.json"
            path.write_bytes(original)
            with patch.object(source_inventory, "SOURCE_INVENTORY_SHA256", pin):
                self.assertEqual(
                    source_inventory.read_pinned_inventory(path), (original, inventory)
                )
                path.write_bytes(original.replace(current.encode(), stale.encode()))
                with self.assertRaisesRegex(
                    ValueError, "does not match the pinned ranges"
                ):
                    source_inventory.read_pinned_inventory(path)


if __name__ == "__main__":
    unittest.main()
