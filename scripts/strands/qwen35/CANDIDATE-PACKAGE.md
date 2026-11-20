# Qwen3.5 TVM candidate

The pending `strands-decider-2b-qwen3.5-v1-2610` model has a private npm archive for exact-package browser tests. Its assets are pinned to the Strands checkpoint, the Qwen base, and the reproduced text and vision WASM files.

## Rebuild the graphs

Use the pinned Apache TVM checkout and Emscripten SDK under `.toolchains`. The compiler checks their Git revisions and builds from the hash-pinned TVMScript inputs in `compiler-inputs`.

```sh
uv sync --locked --project scripts/strands/qwen35/build-env
output=$(mktemp -d)
python=scripts/strands/qwen35/build-env/.venv/bin/python
PYTHONHASHSEED=0 TVM_NUM_THREADS=1 "$python" scripts/strands/qwen35/build_gdn_prefix_candidate.py \
  --precision fp32 --output-dir "$output/text"
PYTHONHASHSEED=0 TVM_NUM_THREADS=1 "$python" scripts/strands/qwen35/build_fp32_vision_webgpu.py \
  --output-dir "$output/vision"
cmp "$output/text/qwen35-gdn-prefix-fp32-candidate.wasm" \
  packages/web-decision/assets/qwen35/r19/qwen35-text-pointer.webgpu.wasm
cmp "$output/vision/qwen35-dynamic-vision.webgpu.wasm" \
  packages/web-decision/assets/qwen35/vision/qwen35-dynamic-vision.webgpu.wasm
```

## Test the exact install

The builder packs and installs a private npm archive in an empty directory outside the checkout. The M4 Pro test covers source parity, text and image limits, and offline cache reuse.

```sh
candidate=$(mktemp -d)
harness_root=$(mktemp -d)
node --import tsx scripts/strands/qwen35/build_candidate_package.mts --output-dir "$candidate"
node --import tsx scripts/strands/qwen35/stage_release_harness.mts --output-dir "$harness_root/assets"
archive="$candidate/pack/web-decision-0.0.0-qwen35-candidate.tgz"
installed="$candidate/consumer/node_modules/web-decision"
WEB_DECISION_QWEN35_RELEASE_GPU=1 \
WEB_DECISION_REAL_GPU=1 \
WEB_DECISION_BROWSER_NAME=chromium \
WEB_DECISION_BROWSER_CHANNEL=chromium \
WEB_DECISION_QWEN35_BUILD_ARTIFACT_ROOT="$harness_root/assets" \
WEB_DECISION_PUBLIC_ASSET_ROOT="$harness_root/assets" \
WEB_DECISION_RELEASE_ARCHIVE="$archive" \
WEB_DECISION_INSTALLED_PACKAGE="$installed" \
WEB_DECISION_INSTALLED_BUNDLE="$installed/dist/index.js" \
WEB_DECISION_QWEN35_REPORT="$candidate/browser-evidence.json" \
pnpm test:browser:strands-qwen2
```
