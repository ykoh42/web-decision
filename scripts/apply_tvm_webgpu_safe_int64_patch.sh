#!/usr/bin/env bash
# Apply the signed-int64 WebGPU size decoding fix to the pinned TVM checkout.
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
tvm_checkout="$repo_root/.toolchains/apache-tvm-src"
patch_file="$repo_root/patches/tvm-webgpu-safe-int64.patch"
revision=7b5acb32bd271342a24130f4ef439ff6c2e52c6c
original_memory=69f93e17e7325eff47bc3da3bf19a77669c15b518ad9e39d5e882adb47629072
patched_memory=795610eda7fa25ff1b0f897702a7e0786f34572d5c1e3053baee6c9d154de0b2
original_runtime=47ef0c4058b57465eb9fbe5185e763d296e253dc2df2160dd7052012c8a04b36
patched_runtime=34494da78d697fb2270c7ab59e7b55e73adea0bcea4098612530df603c32f0e8
original_memory_test=c27f85f40f316f26a387c5e17ac1dfd1727c89411982f83d050396980e0c8bd6
patched_memory_test=553f89635533fa79781f2dae9b1760755a4ddb635d8d20e16041422bd5d269ef
patched_runtime_test=ad12a3af33be181bf6106a6ebf64a835547f0594713b537a599ff127aafe327e
expected_patch=d121f842a35b8bbf717583fad7b9098ba824a8aee44262d49a964f6e32eb4434

hash_file() { shasum -a 256 "$1" | cut -d ' ' -f 1; }

if [[ "$(git -C "$tvm_checkout" rev-parse HEAD)" != "$revision" || "$(hash_file "$patch_file")" != "$expected_patch" ]]; then
  printf 'TVM revision or signed-int64 patch does not match the verified candidate.\n' >&2
  exit 1
fi

memory_path="$tvm_checkout/web/src/memory.ts"
runtime_path="$tvm_checkout/web/src/runtime.ts"
memory_test_path="$tvm_checkout/web/tests/node/test_memory.js"
runtime_test_path="$tvm_checkout/web/tests/node/test_runtime_int64.js"
memory_sha=$(hash_file "$memory_path")
runtime_sha=$(hash_file "$runtime_path")
memory_test_sha=$(hash_file "$memory_test_path")
runtime_test_sha=''
if [[ -f "$runtime_test_path" ]]; then runtime_test_sha=$(hash_file "$runtime_test_path"); fi
if [[ "$memory_sha" == "$patched_memory" && "$runtime_sha" == "$patched_runtime" && \
  "$memory_test_sha" == "$patched_memory_test" && "$runtime_test_sha" == "$patched_runtime_test" ]]; then
  printf 'Verified signed-int64 patch is already applied.\n'
  exit 0
fi

if [[ "$memory_sha" != "$original_memory" || "$runtime_sha" != "$original_runtime" || \
  "$memory_test_sha" != "$original_memory_test" || -e "$runtime_test_path" ]] || \
  ! git -C "$tvm_checkout" diff --quiet HEAD -- \
    web/src/memory.ts web/src/runtime.ts web/tests/node/test_memory.js web/tests/node/test_runtime_int64.js; then
  printf 'TVM source or tests differ from the pinned original; leaving them unchanged.\n' >&2
  exit 1
fi

git -C "$tvm_checkout" apply --check "$patch_file"
git -C "$tvm_checkout" apply "$patch_file"
if [[ "$(hash_file "$memory_path")" != "$patched_memory" || \
  "$(hash_file "$runtime_path")" != "$patched_runtime" || \
  "$(hash_file "$memory_test_path")" != "$patched_memory_test" || \
  "$(hash_file "$runtime_test_path")" != "$patched_runtime_test" ]]; then
  printf 'Patched TVM source or test hash mismatch.\n' >&2
  exit 1
fi

printf 'Applied and verified the TVM signed-int64 patch.\n'
