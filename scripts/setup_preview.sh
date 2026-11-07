#!/usr/bin/env bash
# Reproducibly build the experimental Kev browser preview in a fresh checkout.
# Existing model/build outputs are never overwritten. No package is published.
set -euo pipefail
cd "$(dirname "$0")/.."
root=$PWD
output=${1:-models/web-decision-assets/kev-0.6b-q4f16}
case "$root" in
  *=*) printf 'The build directory must not contain an equals sign: %s\n' "$root" >&2; exit 1 ;;
esac

for path in models models/kev-0.6b-onnx models/kev-0.6b-onnx/onnx reports public .toolchains \
  packages packages/web-decision; do
  if [[ -L "$path" ]]; then
    printf 'Refusing to write through symlinked output directory %s.\n' "$path" >&2
    exit 1
  fi
done
for path in "$output" .preview-venv .toolchains/apache-tvm-src .toolchains/emsdk node_modules \
  packages/web-decision/node_modules packages/web-decision/dist \
  models/kev-0.6b-onnx/onnx/model_q4f16_direct_dynamic.onnx \
  models/kev-0.6b-onnx/onnx/model_q4f16_direct_dynamic_u32.onnx \
  models/tvm-kev-q4-dynamic \
  public/tvm-kev-q4-dynamic.wasm reports/q4-direct-dynamic.json \
  reports/tvm-q4-build-dynamic.json reports/q4-cache-verification-dynamic.json \
  reports/preview-build-environment.json; do
  if [[ -e "$path" || -L "$path" ]]; then
    printf 'Refusing to overwrite %s. Run in a fresh checkout; see packages/web-decision/README.md.\n' "$path" >&2
    exit 1
  fi
done
for command in python3 python3.12 node npm pnpm git curl make shasum; do
  command -v "$command" >/dev/null || { printf 'Missing command: %s\n' "$command" >&2; exit 1; }
done
python3.12 scripts/verify_preview_build_requirements.py
# Fetch first so a failed download can be retried without leaving toolchain or
# build outputs. The fetcher verifies cached inputs and preserves replacements.
bash scripts/fetch-model.sh
mkdir -p .toolchains reports public
python3.12 -m venv .preview-venv
# Install only hashed wheels so isolated source builds cannot add unpinned dependencies.
.preview-venv/bin/python -m pip install --require-hashes --only-binary=:all: -r requirements-preview-build.lock
.preview-venv/bin/python scripts/test_tvm_kev_shape_order.py

revision=7b5acb32bd271342a24130f4ef439ff6c2e52c6c
tvm=.toolchains/apache-tvm-src
git clone https://github.com/apache/tvm.git "$tvm"
git -C "$tvm" checkout --detach "$revision"
if [[ "$(git -C "$tvm" rev-parse HEAD)" != "$revision" ]] || ! git -C "$tvm" diff-index --quiet HEAD --; then
  printf 'TVM checkout must be unmodified at revision %s: %s\n' "$revision" "$tvm" >&2
  exit 1
fi
git -C "$tvm" submodule update --init --recursive 3rdparty/tvm-ffi
bash scripts/apply_tvm_webgpu_safe_int64_patch.sh
sdk=.toolchains/emsdk
git clone https://github.com/emscripten-core/emsdk.git "$sdk"
emsdk_revision=d7b4130688bfa348392b921ab04015aa15c4f154
git -C "$sdk" checkout --detach "$emsdk_revision"
if [[ "$(git -C "$sdk" rev-parse HEAD)" != "$emsdk_revision" ]] || ! git -C "$sdk" diff-index --quiet HEAD --; then
  printf 'Emscripten SDK checkout must be unmodified at revision %s.\n' "$emsdk_revision" >&2
  exit 1
fi
EMSDK_PYTHON="$(command -v python3.12)" "$sdk/emsdk" install 4.0.23
"$sdk/emsdk" activate 4.0.23 --embedded
# This file is created by the locally installed official emsdk.
source "$sdk/emsdk_env.sh" >/dev/null
if ! emcc --version | head -n 1 | grep -Fq '4.0.23'; then
  printf 'Emscripten 4.0.23 is required in %s\n' "$sdk" >&2
  exit 1
fi
export PATH="$root/.preview-venv/bin:$PATH"
export TVM_HOME="$root/$tvm"
# Keep Python's hash iteration and toolchain timestamps stable across builds.
export PYTHONHASHSEED=0
export SOURCE_DATE_EPOCH=0
# TVM's runtime embeds __FILE__ in diagnostics. Map the checkout prefix so a
# distributed model does not expose the developer's absolute directory.
tvm_absolute="$root/$tvm"
runtime_flags=''
for flag in "-I$tvm_absolute" "-I$tvm_absolute/include" "-I$tvm_absolute/3rdparty/tvm-ffi/include" \
  "-I$tvm_absolute/3rdparty/tvm-ffi/3rdparty/dlpack/include" "-I$tvm_absolute/3rdparty/compiler-rt" \
  -O3 -std=c++17 -Wno-ignored-attributes "-ffile-prefix-map=$root=/src"; do
  printf -v escaped '%q' "$flag"
  # Make expands dollar signs before passing the flags to its shell.
  escaped=${escaped//\$/\$\$}
  runtime_flags+=" $escaped"
done
make -C "$tvm/web" -j4 "EMCC_CFLAGS=$runtime_flags"
.preview-venv/bin/python scripts/specialize_q4_direct.py --dynamic
.preview-venv/bin/python scripts/build_tvm_q4.py --dynamic
.preview-venv/bin/python scripts/verify_q4_cache.py --dynamic
.preview-venv/bin/python scripts/prepare_web_decision_assets.py --output "$output"
pnpm install --frozen-lockfile --ignore-scripts
pnpm --filter web-decision test
pnpm --filter web-decision build
.preview-venv/bin/python scripts/record_preview_environment.py
printf '\nPreview assets: %s\nPackage source: %s/packages/web-decision\n' "$output" "$root"
printf 'The package is private and was not published.\n'
