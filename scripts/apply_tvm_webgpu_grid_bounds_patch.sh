#!/usr/bin/env bash
# Correct the upper-bound check for flattened WebGPU block indices in pinned TVM.
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
tvm_checkout="$repo_root/.toolchains/apache-tvm-src"
patch_file="$repo_root/patches/tvm-webgpu-grid-bounds.patch"
revision=7b5acb32bd271342a24130f4ef439ff6c2e52c6c
expected_patch=5bce7a212dd907504c2d673d9d414ab135007faf616a4e4d1550dbde14b88f4b
upstream_codegen=be22bc7c3c53c22ea77d2e29fe00387cdb91ca407ca326ccb8ec18e5df5dce60
bounds_codegen=c96f7cd4e7592a0ebec0f383386f58c313fb2e7f58a99d07fbe5c1b245830fa5
codegen_path="$tvm_checkout/src/backend/webgpu/codegen/codegen_webgpu.cc"

hash_file() { shasum -a 256 "$1" | cut -d ' ' -f 1; }

if [[ "$(git -C "$tvm_checkout" rev-parse HEAD)" != "$revision" || \
  "$(hash_file "$patch_file")" != "$expected_patch" ]]; then
  printf 'TVM revision or grid-bounds patch does not match the verified candidate.\n' >&2
  exit 1
fi

codegen_sha=$(hash_file "$codegen_path")
if [[ "$codegen_sha" == "$bounds_codegen" ]]; then
  printf 'Verified TVM WebGPU grid-bounds patch is already applied.\n'
  exit 0
fi
if [[ "$codegen_sha" != "$upstream_codegen" ]]; then
  printf 'Pinned TVM WebGPU codegen source does not match upstream or the grid-bounds patch.\n' >&2
  exit 1
fi

git -C "$tvm_checkout" apply --check "$patch_file"
git -C "$tvm_checkout" apply "$patch_file"
if [[ "$(hash_file "$codegen_path")" != "$bounds_codegen" ]]; then
  printf 'Patched TVM WebGPU codegen hash mismatch.\n' >&2
  exit 1
fi

printf 'Applied and verified the TVM WebGPU grid-bounds patch.\n'
