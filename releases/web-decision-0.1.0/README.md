# Rebuild `web-decision@0.1.0`

This directory contains the reviewed source inputs for the 13-file, code-only npm package. It does not contain model weights or the WASM asset.

## Prepare model assets from a fresh checkout

Use a clean checkout for asset preparation. `scripts/setup_preview.sh` refuses to run if paths such as `node_modules`, `.toolchains`, `.preview-venv`, or its model outputs already exist. Run it before installing workspace dependencies or building packages; it performs its own `pnpm install` after preparing the checkpoint and toolchain.

```sh
bash scripts/setup_preview.sh
```

The setup downloads the pinned checkpoint, prepares the compiler toolchain, and builds baseline files at `models/web-decision-assets/kev-0.6b-q4f16/`. It may require substantial disk space and memory. Then build q2-k128 WASM and prepare the final asset pack:

```sh
source .toolchains/emsdk/emsdk_env.sh
export TVM_HOME="$PWD/.toolchains/apache-tvm-src"
export PYTHONHASHSEED=0 SOURCE_DATE_EPOCH=0
KEY_TILE=128 .preview-venv/bin/python releases/web-decision-0.1.0/attention-q2/build_rewritten_model.py
node releases/web-decision-0.1.0/prepare-asset-pack.mjs \
  --baseline-dir models/web-decision-assets/kev-0.6b-q4f16 \
  --wasm models/kev-attention-q2-k128-build/model.wasm \
  --output models/web-decision-assets/kev-0.6b-q4f16-q2-k128
```

The graph and external tensor data must match the hashes in `pins.json`. The builder rejects any key tile other than 128 and exits nonzero if WASM export fails or its result differs from the pinned hash and size. It writes to a new `models/kev-attention-q2-k128-build/` directory.

The helper combines the 13 non-WASM baseline files with the pinned q2-k128 WASM. It verifies all 14 asset hashes and preserves all 14 pinned files in `notices/` outside `manifest.files`; the notices are served beside the assets but are not fetched by the inference loader. The output path must not already exist.

After setup, rebuild and pack the code-only npm package:

```sh
node releases/web-decision-0.1.0/build.mjs
node releases/web-decision-0.1.0/pack.mjs
```

The build verifies the source lock, asset-manifest pin, TVM WebGPU source, and expected bundle hash. The pack step checks the 13-file allowlist, runs `publint` and `@arethetypeswrong/cli`, and creates the npm archive. Generated `build/` outputs are ignored by Git.

Install the resulting local archive in an application with:

```sh
npm install ./releases/web-decision-0.1.0/build/release/web-decision-0.1.0.tgz
```

The package README contains the browser API example. Host the separately prepared asset pack at an origin that allows requests from the application.

## Package-only build

If you only need the code-only npm package, use an ordinary checkout and run:

```sh
pnpm install --frozen-lockfile
node releases/web-decision-0.1.0/build.mjs
node releases/web-decision-0.1.0/pack.mjs
```

Do not run `scripts/setup_preview.sh` after these commands in the same checkout. If you later need asset preparation, use a separate fresh checkout and run setup there first.

The expected q2-k128 WASM hash and size are recorded pins from the candidate. This source scaffold does not claim that a fresh compiler rebuild produced those bytes. Compare the generated `build-report.json` with `pins.json` before making that claim. Browser GPU evidence applies to the exact recorded `0.1.0` archive and device; see [RELEASE_SCOPE.md](../../RELEASE_SCOPE.md).

