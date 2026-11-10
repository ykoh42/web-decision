#!/usr/bin/env bash
# Use the WebGPU adapter's reported buffer limits in the pinned TVM runtime.
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
tvm_checkout="$repo_root/.toolchains/apache-tvm-src"
patch_file="$repo_root/patches/tvm-webgpu-adapter-limits.patch"
revision=7b5acb32bd271342a24130f4ef439ff6c2e52c6c
original_source=77003ab2bc6c7ae103b38fb52c12a5a7a3d49597260b43f010deaacc5d65b7c1
patched_source=f9dcc047285dfe7d0b9d3eee0ce81df3c20cd5935cd34db581f6bce1ac887f23
expected_patch=51fa5d820fce59c2d5eddd959c51fc6e362b3731caaf65e249558c07a0e109fb

hash_file() { shasum -a 256 "$1" | cut -d ' ' -f 1; }

if [[ "$(git -C "$tvm_checkout" rev-parse HEAD)" != "$revision" || \
  "$(hash_file "$patch_file")" != "$expected_patch" ]]; then
  printf 'TVM revision or adapter-limits patch does not match the verified candidate.\n' >&2
  exit 1
fi

source_path="$tvm_checkout/web/src/webgpu.ts"
source_sha=$(hash_file "$source_path")
if [[ "$source_sha" == "$patched_source" ]]; then
  printf 'Verified TVM WebGPU adapter-limits patch is already applied.\n'
  exit 0
fi
if [[ "$source_sha" != "$original_source" ]]; then
  printf 'TVM WebGPU source differs from the pinned original; leaving it unchanged.\n' >&2
  exit 1
fi

git -C "$tvm_checkout" apply --check "$patch_file"
git -C "$tvm_checkout" apply "$patch_file"
if [[ "$(hash_file "$source_path")" != "$patched_source" ]]; then
  printf 'Patched TVM WebGPU source hash mismatch.\n' >&2
  exit 1
fi

printf 'Applied and verified the TVM WebGPU adapter-limits patch.\n'
