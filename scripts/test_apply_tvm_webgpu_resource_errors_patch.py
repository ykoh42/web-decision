"""CPU-only application and idempotency check for the TVM resource-error patch."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TVM_REVISION = "7b5acb32bd271342a24130f4ef439ff6c2e52c6c"


class KananaPatchSequenceTest(unittest.TestCase):
    def test_umbrella_sequence_applies_twice_and_fails_closed(self) -> None:
        source_checkout = REPO_ROOT / ".toolchains/apache-tvm-src"
        with tempfile.TemporaryDirectory(prefix="tvm-resource-errors-test-") as temporary:
            temporary_root = Path(temporary) / "workspace"
            temporary_tvm = temporary_root / ".toolchains/apache-tvm-src"
            scripts = temporary_root / "scripts"
            patches = temporary_root / "patches"
            scripts.mkdir(parents=True)
            patches.mkdir(parents=True)

            script_names = (
                "apply_tvm_webgpu_kanana_patches.sh",
                "apply_tvm_webgpu_safe_int64_patch.sh",
                "apply_tvm_webgpu_adapter_limits_patch.sh",
                "apply_tvm_webgpu_resource_errors_patch.sh",
                "apply_tvm_webgpu_bool_storage_int32_patch.sh",
                "apply_tvm_webgpu_int32_min_literal_patch.sh",
                "apply_tvm_webgpu_float_nonfinite_patch.sh",
                "apply_tvm_webgpu_grid_bounds_patch.sh",
            )
            for script in script_names:
                shutil.copy2(REPO_ROOT / "scripts" / script, scripts / script)
            patch_names = (
                "tvm-webgpu-safe-int64.patch",
                "tvm-webgpu-adapter-limits.patch",
                "tvm-webgpu-resource-errors.patch",
                "tvm-webgpu-bool-storage-int32.patch",
                "tvm-webgpu-int32-min-literal.patch",
                "tvm-webgpu-float-nonfinite.patch",
                "tvm-webgpu-grid-bounds.patch",
            )
            for patch in patch_names:
                shutil.copy2(REPO_ROOT / "patches" / patch, patches / patch)

            subprocess.run(
                [
                    "git",
                    "-C",
                    str(source_checkout),
                    "worktree",
                    "add",
                    "--detach",
                    "--quiet",
                    str(temporary_tvm),
                    TVM_REVISION,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            try:
                sequence_script = scripts / "apply_tvm_webgpu_kanana_patches.sh"
                first_result = subprocess.run(
                    ["bash", str(sequence_script)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                self.assertIn("Applied and verified the TVM WebGPU resource-errors patch", first_result.stdout)

                second_result = subprocess.run(
                    ["bash", str(sequence_script)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                self.assertIn("resource-errors patches are already applied", second_result.stdout)

                source_file = temporary_tvm / "web/src/webgpu.ts"
                source_file.write_bytes(source_file.read_bytes() + b"\n")
                modified_source = source_file.read_bytes()
                rejected_result = subprocess.run(
                    ["bash", str(sequence_script)],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(rejected_result.returncode, 0)
                self.assertIn("leaving it unchanged", rejected_result.stderr)
                self.assertEqual(source_file.read_bytes(), modified_source)
            finally:
                subprocess.run(
                    ["git", "-C", str(source_checkout), "worktree", "remove", "--force", str(temporary_tvm)],
                    check=False,
                    capture_output=True,
                    text=True,
                )


if __name__ == "__main__":
    unittest.main()
