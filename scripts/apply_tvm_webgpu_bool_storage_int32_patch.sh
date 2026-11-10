#!/usr/bin/env bash
# Apply the WGSL-compatible bool storage type to the pinned TVM checkout.
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
tvm_checkout="$repo_root/.toolchains/apache-tvm-src"
patch_file="$repo_root/patches/tvm-webgpu-bool-storage-int32.patch"
revision=7b5acb32bd271342a24130f4ef439ff6c2e52c6c
original_header=8a60c4687411c58d50c2732675cb4731cb4739924f38b0e643658a3f72eae97f
patched_header=6b759e840079f963c3e0ef47dc38b1674fd6ae8a9bff63c58e6c3e6d1175f440
expected_patch=5c83161021661adaaf52fb8bf1935fade7659fd7a3a64fab325a180702dd5adb

hash_file() { shasum -a 256 "$1" | cut -d ' ' -f 1; }

if [[ "$(git -C "$tvm_checkout" rev-parse HEAD)" != "$revision" || \
  "$(hash_file "$patch_file")" != "$expected_patch" ]]; then
  printf 'TVM revision or bool-storage patch does not match the verified candidate.\n' >&2
  exit 1
fi

header_path="$tvm_checkout/src/backend/webgpu/codegen/codegen_webgpu.h"
header_sha=$(hash_file "$header_path")
if [[ "$header_sha" == "$patched_header" ]]; then
  printf 'Verified WGSL bool-storage patch is already applied.\n'
  exit 0
fi
if [[ "$header_sha" != "$original_header" ]]; then
  printf 'TVM WebGPU codegen header differs from the pinned original; leaving it unchanged.\n' >&2
  exit 1
fi

git -C "$tvm_checkout" apply --check "$patch_file"
git -C "$tvm_checkout" apply "$patch_file"
if [[ "$(hash_file "$header_path")" != "$patched_header" ]]; then
  printf 'Patched TVM WebGPU codegen header hash mismatch.\n' >&2
  exit 1
fi

printf 'Applied and verified the TVM WebGPU bool-storage patch.\n'
