#!/usr/bin/env bash
# Use the WebGPU adapter's reported buffer limits in the pinned TVM runtime.
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
tvm_checkout="$repo_root/.toolchains/apache-tvm-src"
patch_file="$repo_root/patches/tvm-webgpu-adapter-limits.patch"
revision=7b5acb32bd271342a24130f4ef439ff6c2e52c6c
original_source=77003ab2bc6c7ae103b38fb52c12a5a7a3d49597260b43f010deaacc5d65b7c1
patched_source=b8f58d5abafc2d32d5462bdfbaa4b4b53abf0fd5b3c0a09dbafd9d8ff58502b7
resource_patched_source=01e67c6e6f0dd887bd12dac03619866456e2a9caf72e8c8a9dc249ed3bcf83e1
resource_patched_index=956c39aec01924e0c34261d903657d1c4b17044f572efcc11ed64f6d8ee9a630
resource_patched_test=37782408161988e557da278b9c62ab40ee0d900cd051b12174251a0ace29bc8e
expected_patch=fa79e8e1924f952ae67c2ce7155ff7e3b338753eb10baffcbf18d075c59d1276

hash_file() { shasum -a 256 "$1" | cut -d ' ' -f 1; }

if [[ "$(git -C "$tvm_checkout" rev-parse HEAD)" != "$revision" || \
  "$(hash_file "$patch_file")" != "$expected_patch" ]]; then
  printf 'TVM revision or adapter-limits patch does not match the verified candidate.\n' >&2
  exit 1
fi

source_path="$tvm_checkout/web/src/webgpu.ts"
source_sha=$(hash_file "$source_path")
index_path="$tvm_checkout/web/src/index.ts"
index_sha=$(hash_file "$index_path")
resource_test_path="$tvm_checkout/web/tests/node/test_webgpu_oom.js"
resource_test_sha=
if [[ -f "$resource_test_path" ]]; then
  resource_test_sha=$(hash_file "$resource_test_path")
fi
if [[ "$source_sha" == "$resource_patched_source" && \
  "$index_sha" == "$resource_patched_index" && \
  "$resource_test_sha" == "$resource_patched_test" ]]; then
  printf 'Verified TVM WebGPU adapter-limits and resource-errors patches are already applied.\n'
  exit 0
fi
if [[ "$source_sha" == "$patched_source" ]]; then
  printf 'Verified TVM WebGPU adapter-limits patch is already applied.\n'
  exit 0
fi
if [[ "$source_sha" != "$original_source" ]]; then
  printf 'TVM WebGPU source differs from the pinned original; leaving it unchanged.\n' >&2
  exit 1
fi

git -C "$tvm_checkout" apply --unidiff-zero --check "$patch_file"
git -C "$tvm_checkout" apply --unidiff-zero "$patch_file"
if [[ "$(hash_file "$source_path")" != "$patched_source" ]]; then
  printf 'Patched TVM WebGPU source hash mismatch.\n' >&2
  exit 1
fi

printf 'Applied and verified the TVM WebGPU adapter-limits patch.\n'
