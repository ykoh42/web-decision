#!/usr/bin/env bash
# Emit the signed int32 minimum as a WGSL-valid expression in pinned TVM.
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
tvm_checkout="$repo_root/.toolchains/apache-tvm-src"
patch_file="$repo_root/patches/tvm-webgpu-int32-min-literal.patch"
revision=7b5acb32bd271342a24130f4ef439ff6c2e52c6c
expected_patch=75f9908d707a3bf604a4ed93723fdee96c889be42e7e98f36ff3bef8e3aa0384
codegen_path="$tvm_checkout/src/backend/webgpu/codegen/codegen_webgpu.cc"
base_codegen=be22bc7c3c53c22ea77d2e29fe00387cdb91ca407ca326ccb8ec18e5df5dce60
int32_codegen=2cb8ece4ffd5006d70d4bd576de76f07811921ce0b0c63a5becf436cd8373ad5
float_codegen=6cfd47cfac9d3dbaeb95a85d32c2f05af37a9fcbcf103184ccd172b2f07b9250
bounds_codegen=044422cae6b09e3e399767df66c37fc60524e52311939332150039ab4bf63180

hash_file() { shasum -a 256 "$1" | cut -d ' ' -f 1; }

if [[ "$(git -C "$tvm_checkout" rev-parse HEAD)" != "$revision" || \
  "$(shasum -a 256 "$patch_file" | cut -d ' ' -f 1)" != "$expected_patch" ]]; then
  printf 'TVM revision or int32-minimum patch does not match the verified candidate.\n' >&2
  exit 1
fi

codegen_sha=$(hash_file "$codegen_path")
if [[ "$codegen_sha" == "$int32_codegen" || "$codegen_sha" == "$float_codegen" || \
  "$codegen_sha" == "$bounds_codegen" ]]; then
  printf 'Verified TVM WGSL int32-minimum patch is already applied.\n'
  exit 0
fi
if [[ "$codegen_sha" != "$base_codegen" ]]; then
  printf 'TVM WebGPU codegen differs from the pinned original; leaving it unchanged.\n' >&2
  exit 1
fi

git -C "$tvm_checkout" apply --check "$patch_file"
git -C "$tvm_checkout" apply "$patch_file"
if [[ "$(hash_file "$codegen_path")" != "$int32_codegen" ]]; then
  printf 'Patched TVM WebGPU codegen does not contain the expected int32-minimum fix.\n' >&2
  exit 1
fi

printf 'Applied and verified the TVM WGSL int32-minimum patch.\n'
