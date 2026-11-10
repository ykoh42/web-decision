#!/usr/bin/env bash
# Reproducibly build a local direct-TVM Kev candidate in a fresh checkout.
# The output is not installed into the curated catalog or published.
set -euo pipefail
cd "$(dirname "$0")/.."
root=$PWD
output=${1:-models/web-decision-assets/direct-kev-0.6b-q4f16}
converter="$root/tools/kev-converter"
venv="$converter/.venv"
libatomic_path=${WEB_DECISION_LIBATOMIC_PATH:-/usr/lib/x86_64-linux-gnu/libatomic.so.1}
tvm_revision=7b5acb32bd271342a24130f4ef439ff6c2e52c6c
emsdk_revision=d7b4130688bfa348392b921ab04015aa15c4f154
kev_code_revision=37cb2f6155afec1054de2cc78dfd7c7dd24a956d
kev_checkpoint_revision=abe7110dedfd68a185839550079bcf7c26f561ac
base_revision=da87bfb608c14b7cf20ba1ce41287e8de496c0cd
tvm="$root/.toolchains/apache-tvm-src"
sdk="$root/.toolchains/emsdk"
kev_source="$root/.toolchains/kev-source"

case "$root" in
  *=*) printf 'The build directory must not contain an equals sign: %s\n' "$root" >&2; exit 1 ;;
esac
case "$output" in
  /*|..|../*|*/../*|*/..) printf 'The output path must stay inside the checkout: %s\n' "$output" >&2; exit 1 ;;
esac

for path in tools tools/kev-converter .toolchains models models/web-decision-assets; do
  if [[ -L "$path" ]]; then
    printf 'Refusing to write through symlinked directory %s.\n' "$path" >&2
    exit 1
  fi
done
for path in "$output" .toolchains/apache-tvm-src .toolchains/emsdk .toolchains/kev-source; do
  if [[ -e "$path" || -L "$path" ]]; then
    printf 'Refusing to overwrite %s. Run in a fresh checkout.\n' "$path" >&2
    exit 1
  fi
done
for command in python3.12 uv git curl make; do
  command -v "$command" >/dev/null || { printf 'Missing command: %s\n' "$command" >&2; exit 1; }
done
python="$(command -v python3.12)"
if [[ ! -f "$converter/pyproject.toml" || ! -f "$converter/uv.lock" || ! -f scripts/tvm-browser-runtime.mk ]]; then
  printf 'The pinned direct Kev converter and TVM runtime build inputs are incomplete.\n' >&2
  exit 1
fi
if [[ ! -f "$libatomic_path" ]]; then
  printf 'Missing pinned libatomic1 runtime library: %s\nSet WEB_DECISION_LIBATOMIC_PATH to its path.\n' "$libatomic_path" >&2
  exit 1
fi

# Use the converter's hash-locked Python environment and its pinned HF client.
export UV_PROJECT_ENVIRONMENT="$venv"
uv sync --locked --project "$converter" --python "$python" --no-dev
hf="$venv/bin/hf"
if [[ ! -x "$hf" ]]; then
  printf 'The pinned converter environment did not install the Hugging Face CLI.\n' >&2
  exit 1
fi

# Download only the official files consumed by the converter. The converter's
# INPUT_HASHES and source-revision checks reject any mismatched snapshot.
kev_dir=$("$hf" download --quiet --revision "$kev_checkpoint_revision" jaredpalmer/kev-0.6b \
  adapter_config.json adapter_model.safetensors head.pt provenance.json training_config.json)
base_dir=$("$hf" download --quiet --revision "$base_revision" Qwen/Qwen3-0.6B-Base \
  LICENSE config.json model.safetensors tokenizer.json tokenizer_config.json)

# Pin the Kev implementation used to assemble the adapter, base model, and head.
mkdir -p .toolchains
git init "$kev_source" >/dev/null
git -C "$kev_source" remote add origin https://github.com/jaredpalmer/kev.git
git -C "$kev_source" fetch --depth=1 origin "$kev_code_revision"
GIT_LFS_SKIP_SMUDGE=1 git -C "$kev_source" checkout --detach FETCH_HEAD

# Run the converter's own source hash and revision verifier before building TVM.
"$venv/bin/python" - "$converter" "$kev_dir" "$base_dir" "$kev_source" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from build_tvm import verify_inputs

verified = verify_inputs(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
print(f"Verified {len(verified)} pinned direct-conversion inputs.")
PY

git clone https://github.com/apache/tvm.git "$tvm"
git -C "$tvm" checkout --detach "$tvm_revision"
if [[ "$(git -C "$tvm" rev-parse HEAD)" != "$tvm_revision" ]] || ! git -C "$tvm" diff-index --quiet HEAD --; then
  printf 'TVM checkout must be unmodified at revision %s: %s\n' "$tvm_revision" "$tvm" >&2
  exit 1
fi
git -C "$tvm" submodule update --init --recursive 3rdparty/tvm-ffi

git clone https://github.com/emscripten-core/emsdk.git "$sdk"
git -C "$sdk" checkout --detach "$emsdk_revision"
if [[ "$(git -C "$sdk" rev-parse HEAD)" != "$emsdk_revision" ]] || ! git -C "$sdk" diff-index --quiet HEAD --; then
  printf 'Emscripten SDK checkout must be unmodified at revision %s.\n' "$emsdk_revision" >&2
  exit 1
fi
EMSDK_PYTHON="$python" "$sdk/emsdk" install 4.0.23
"$sdk/emsdk" activate 4.0.23 --embedded
# This file is created by the locally installed official emsdk.
source "$sdk/emsdk_env.sh" >/dev/null
if ! emcc --version | head -n 1 | grep -Fq '4.0.23'; then
  printf 'Emscripten 4.0.23 is required in %s\n' "$sdk" >&2
  exit 1
fi

export TVM_HOME="$tvm"
export PYTHONHASHSEED=0
export SOURCE_DATE_EPOCH=0
# TVM's runtime embeds __FILE__ in diagnostics. Map the checkout prefix so a
# distributed model does not expose the developer's absolute directory.
tvm_absolute="$tvm"
runtime_flags=''
for flag in "-I$tvm_absolute" "-I$tvm_absolute/include" "-I$tvm_absolute/3rdparty/tvm-ffi/include" \
  "-I$tvm_absolute/3rdparty/tvm-ffi/3rdparty/dlpack/include" "-I$tvm_absolute/3rdparty/compiler-rt" \
  -O3 -std=c++17 -Wno-ignored-attributes "-ffile-prefix-map=$root=/src"; do
  printf -v escaped '%q' "$flag"
  # Make expands dollar signs before passing the flags to its shell.
  escaped=${escaped//\$/\$\$}
  runtime_flags+=" $escaped"
done
make -C "$tvm/web" -f Makefile -f "$root/scripts/tvm-browser-runtime.mk" \
  -j4 "EMCC_CFLAGS=$runtime_flags"

"$venv/bin/python" "$converter/build_tvm.py" \
  --kev-dir "$kev_dir" \
  --base-dir "$base_dir" \
  --source-dir "$kev_source" \
  --tvm-home "$tvm" \
  --libatomic-path "$libatomic_path" \
  --output-dir "$output"

printf '\nDirect TVM candidate: %s/browser-package\nBuild report: %s/direct-tvm-report.json\n' "$output" "$output"
printf 'This local candidate is not integrated into the curated catalog or published.\n'
