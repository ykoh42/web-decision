#!/usr/bin/env bash
# Apply the pinned TVM source changes used by the Kanana runtime build.
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)

scripts=(
  apply_tvm_webgpu_safe_int64_patch.sh
  apply_tvm_webgpu_adapter_limits_patch.sh
  apply_tvm_webgpu_resource_errors_patch.sh
  apply_tvm_webgpu_grid_bounds_patch.sh
)

for script in "${scripts[@]}"; do
  bash "$repo_root/scripts/$script"
done
