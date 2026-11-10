#!/usr/bin/env bash
# Propagate captured WebGPU buffer allocation errors from the pinned TVM runtime.
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
tvm_checkout="$repo_root/.toolchains/apache-tvm-src"
patch_file="$repo_root/patches/tvm-webgpu-resource-errors.patch"
revision=7b5acb32bd271342a24130f4ef439ff6c2e52c6c
expected_patch=58ebe9d953ae04678746ee4c6ca7770745561f8865281b4423946fa8b463fc55
original_source=b8f58d5abafc2d32d5462bdfbaa4b4b53abf0fd5b3c0a09dbafd9d8ff58502b7
patched_source=01e67c6e6f0dd887bd12dac03619866456e2a9caf72e8c8a9dc249ed3bcf83e1
original_index=9867645a732ed912c3f69a551ced6d8ed978f8a751686712cf4ae400ae5abcf2
patched_index=956c39aec01924e0c34261d903657d1c4b17044f572efcc11ed64f6d8ee9a630
expected_test=37782408161988e557da278b9c62ab40ee0d900cd051b12174251a0ace29bc8e

hash_file() { shasum -a 256 "$1" | cut -d ' ' -f 1; }

if [[ "$(git -C "$tvm_checkout" rev-parse HEAD)" != "$revision" || \
  "$(hash_file "$patch_file")" != "$expected_patch" ]]; then
  printf 'TVM revision or resource-errors patch does not match the verified candidate.\n' >&2
  exit 1
fi

source_path="$tvm_checkout/web/src/webgpu.ts"
index_path="$tvm_checkout/web/src/index.ts"
test_path="$tvm_checkout/web/tests/node/test_webgpu_oom.js"
source_sha=$(hash_file "$source_path")
index_sha=$(hash_file "$index_path")
test_sha=
if [[ -f "$test_path" ]]; then
  test_sha=$(hash_file "$test_path")
fi

if [[ "$source_sha" == "$patched_source" && "$index_sha" == "$patched_index" && \
  "$test_sha" == "$expected_test" ]]; then
  printf 'Verified TVM WebGPU resource-errors patch is already applied.\n'
  exit 0
fi
if [[ "$source_sha" != "$original_source" || "$index_sha" != "$original_index" || \
  -e "$test_path" ]]; then
  printf 'TVM WebGPU source differs from the pinned adapter-limits state; leaving it unchanged.\n' >&2
  exit 1
fi

git -C "$tvm_checkout" apply --check "$patch_file"
git -C "$tvm_checkout" apply "$patch_file"
if [[ "$(hash_file "$source_path")" != "$patched_source" || \
  "$(hash_file "$index_path")" != "$patched_index" || \
  "$(hash_file "$test_path")" != "$expected_test" ]]; then
  printf 'Patched TVM WebGPU resource-errors source hash mismatch.\n' >&2
  exit 1
fi

printf 'Applied and verified the TVM WebGPU resource-errors patch.\n'
