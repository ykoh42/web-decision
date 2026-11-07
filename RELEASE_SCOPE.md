# Release scope and verification

Each stable release carries forward the supported model routes and adds one newly completed route. A model is added only after its exact package archive passes package, type, and real-model browser checks. Model weights are hosted separately from npm.

## `web-decision@0.1.0`

This release supports `kev-0.6b-q4f16` through TVM WebGPU with `shader-f16`. The npm archive contains 13 code and notice files. It excludes model weights, tokenizer files, and runtime assets. The separately prepared Kev asset pack contains 349,103,828 inference bytes.

The exact npm archive SHA-256 is `b98ee9006192257313ad736f8aec6a3e6e632d2a844ee5ca63bc9060a4cf6484`. The archive, bundle, and browser evidence are recorded in [`releases/web-decision-0.1.0/verification`](releases/web-decision-0.1.0/verification). Per-browser stage events are in [`gpu-runs`](releases/web-decision-0.1.0/verification/gpu-runs).

Real-model GPU tests ran on one Apple M4 Pro with a non-fallback adapter and `shader-f16`:

| Browser | Version | Passing GPU cases |
| --- | --- | ---: |
| Chromium | 156.0.8078.4 | 11/11 |
| Firefox | 157.0 | 11/11 |
| WebKit | 27.2 | 11/11 |

The suite covered generic and typed 8,192-token inputs, token boundaries from 4,096 through 8,192, a 2,217-token aggregate request, Choice/Noul/Score semantics, option order, probability validity, and validation at and above the accepted token limit. Chromium's combined long-input diagnostic exceeded its 360-second Playwright timeout; the seven stages passed when run independently. These results do not establish compatibility with other devices, operating systems, or browser versions, or decision accuracy.

## `web-decision@0.2.0`

This release adds `experimental-tamev-nano-tinybert-int8` and retains Kev. Kev uses TVM WebGPU with `shader-f16`; TAMEV Nano uses ONNX Runtime WebAssembly on CPU. The npm archive contains 15 code and notice files and excludes model weights and runtime assets. The separately prepared asset packs contain 349,103,828 Kev inference bytes and 15,585,045 TAMEV Nano bytes.

The exact npm archive is 102,265 bytes with SHA-256 `bd29861052f7de6d8a39f60f78253cc36ca6980e7ae0a104f09009354ee798f4`; its bundle is 342,689 bytes with SHA-256 `edf6ab23d6f47c922934814511ff06e4a6b734337b44613f4fc911a0cec148e0`. The package is published to npm as `latest`. [`npm-publication.json`](releases/web-decision-0.2.0/verification/npm-publication.json) records the registry tarball hash comparison.

Exact-archive real-model checks ran on one Apple M4 Pro in Chromium, Firefox, and WebKit. Kev passed real WebGPU inference in all three browsers with `shader-f16` and a non-fallback adapter. TAMEV Nano passed real CPU/WASM inference in all three browsers. The logs and package verification are in [`releases/web-decision-0.2.0/verification`](releases/web-decision-0.2.0/verification). The results establish runtime/API behavior for those tests, not model quality or compatibility across devices.

Package, strict consumer type, publish dry-run, dependency audit, and model-contract checks passed. The security review and its recorded coverage limit are in [`verification/security-review.md`](releases/web-decision-0.2.0/verification/security-review.md).

## `web-decision@0.3.0` candidate

This candidate adds exactly one model route, `experimental-tamev-micro-minilm-int8`, and retains Kev and TAMEV Nano. The package archive contains 17 code and notice files; it excludes model weights and runtime assets. The three separately prepared asset packs contain 349,103,828 Kev inference bytes, 15,585,045 Nano bytes, and 24,161,003 Micro bytes.

The exact archive prepared for publication is 103,515 bytes with SHA-256 `ccfa582e2070939b3f03e8c4759eaababc6b1f0ed1d546596d079123c4b54a50`; the bundle is 347,305 bytes with SHA-256 `b4aee808c55dc592abe920ae4c45aae8e1bc2abc0ca3fca161a0a25a92d5c9a0`. The exact archive passed nine real-model cases on one Apple M4 Pro: Kev TVM WebGPU, Nano CPU/WASM, and Micro CPU/WASM, each in Chromium, Firefox, and WebKit. Results and per-route logs are in [`releases/web-decision-0.3.0/verification`](releases/web-decision-0.3.0/verification).

The clean release worktree passed frozen-lockfile installation, 13 Node tests, 10 Python attention tests, 14 safe-output tests, `publint`, and package/type checks. A regenerated archive contained the same 17 package files byte-for-byte; the exact browser-tested archive is retained for publication. The source security review found no vulnerability candidates. Its 12 TVM overlay files were checked separately and match the 0.1.0 overlays byte-for-byte. See [`verification/security-review.md`](releases/web-decision-0.3.0/verification/security-review.md).

The candidate is still pending GitHub Release and npm publication. The npm `latest` version remains 0.2.0 until those steps complete.

## Limits and security notes

- The Kev adapter preserves accepted input and does not silently truncate it. Applications can configure optional `maxInputTokens` and `maxInputCharacters` limits and should set request-size and concurrency controls at their own boundary when handling untrusted input.
- TAMEV follows each pinned model's tokenizer and truncation behavior and reports truncation. Applications should set their own aggregate request-size and concurrency controls for untrusted input.
- An abort signal can cancel queued work but does not preempt inference already dispatched to a runtime.
- `assetsUrl` belongs in trusted application configuration. The loader requests model assets before it can verify their hashes, so do not derive the URL from user input. Decision input is not sent to the asset host.
- Model files are checked against pinned hashes before inference. The package excludes model weights; this does not determine training-data provenance or broader redistribution rights.
- Real-model results apply only to the tested archive and the listed M4 Pro browser runs. They do not establish general decision quality or broad device support.

## Previous preview

The `web-decision@0.1.0-preview.1` archive and evidence remain under [`releases/web-decision-0.1.0-preview.1`](releases/web-decision-0.1.0-preview.1). Its archive hash is separate from the stable `0.1.0` archive.

## Build and release checks

Each version's source lock, build, package archive, and browser evidence are stored under `releases/web-decision-<version>/`. The [release procedure](.github/RELEASING.md) requires a version-specific local build, exact-archive real-model browser tests for every included route, rights review, and attaching the same tested archive to GitHub Release and npm. Standard GitHub-hosted runners do not provide the local GPU evidence required for WebGPU routes.
