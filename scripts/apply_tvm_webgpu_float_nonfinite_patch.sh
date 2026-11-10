#!/usr/bin/env bash
# Emit finite WGSL expressions for non-finite f32 and f16 literals in pinned TVM.
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
tvm_checkout="$repo_root/.toolchains/apache-tvm-src"
patch_file="$repo_root/patches/tvm-webgpu-float-nonfinite.patch"
revision=7b5acb32bd271342a24130f4ef439ff6c2e52c6c
expected_patch=fc81d99144d4ce471e2ce4523b23a99a03727f34933bdabaf9188d0201b3a17b
codegen_path="$tvm_checkout/src/backend/webgpu/codegen/codegen_webgpu.cc"
int32_codegen=2cb8ece4ffd5006d70d4bd576de76f07811921ce0b0c63a5becf436cd8373ad5
float_codegen=6cfd47cfac9d3dbaeb95a85d32c2f05af37a9fcbcf103184ccd172b2f07b9250
bounds_codegen=044422cae6b09e3e399767df66c37fc60524e52311939332150039ab4bf63180

hash_file() { shasum -a 256 "$1" | cut -d ' ' -f 1; }

if [[ "$(git -C "$tvm_checkout" rev-parse HEAD)" != "$revision" || \
  "$(shasum -a 256 "$patch_file" | cut -d ' ' -f 1)" != "$expected_patch" ]]; then
  printf 'TVM revision or non-finite-literal patch does not match the verified candidate.\n' >&2
  exit 1
fi
codegen_sha=$(hash_file "$codegen_path")
if [[ "$codegen_sha" == "$float_codegen" || "$codegen_sha" == "$bounds_codegen" ]]; then
  printf 'Verified TVM WebGPU non-finite-literal patch is already applied.\n'
  exit 0
fi
if [[ "$codegen_sha" != "$int32_codegen" ]]; then
  printf 'Apply the pinned TVM int32-minimum patch before the non-finite-literal patch.\n' >&2
  exit 1
fi

git -C "$tvm_checkout" apply --check "$patch_file"
git -C "$tvm_checkout" apply "$patch_file"
if [[ "$(hash_file "$codegen_path")" != "$float_codegen" ]]; then
  printf 'Patched TVM WebGPU codegen does not contain the expected non-finite-literal fix.\n' >&2
  exit 1
fi

printf 'Applied and verified the TVM WebGPU non-finite-literal patch.\n'
