# Release scope and verification

## `web-decision@0.1.0`

This stable release supports exactly one browser-local model route: `kev-0.6b-q4f16`, using TVM WebGPU with the `shader-f16` feature. The npm archive contains 13 code and notice files; it excludes model weights, tokenizer files, and runtime assets. Applications must prepare and host the pinned 14-file asset pack separately; its inference files total 349,103,828 bytes.

The exact npm archive SHA-256 is `b98ee9006192257313ad736f8aec6a3e6e632d2a844ee5ca63bc9060a4cf6484`. The matching bundle and real-model GPU results are recorded in [`releases/web-decision-0.1.0/verification`](releases/web-decision-0.1.0/verification). Per-browser test logs and structured stage events are preserved in [`verification/gpu-runs`](releases/web-decision-0.1.0/verification/gpu-runs).

GPU runs used one Apple M4 Pro and a non-fallback adapter with `shader-f16` in all browsers:

| Browser | Version | Passing GPU cases | Lowest sampled macOS `memory_pressure` free % |
| --- | --- | ---: | ---: |
| Chromium | 156.0.8078.4 | 11/11 | Not sampled during the run |
| Firefox | 157.0 | 11/11 | 78% (85 samples) |
| WebKit | 27.2 | 11/11 | 75% (175 samples) |

The suite covered generic and typed 8,192-token inputs; token lengths 4,096, 6,144, 7,168, 8,191, and 8,192; a 2,217-token aggregate request; Choice/Noul/Score semantics; option order; probability validity; and validation at and above the accepted token limit. Chromium's first combined long-input diagnostic exceeded its 360-second Playwright test timeout. The seven long-input stages passed when the harness ran them independently. The final recorded result is 11 passing cases per browser.

The GPU results apply to the exact archive and bundle above. They do not establish compatibility with other devices, operating systems, or browser versions, nor decision accuracy or general task quality.

## Limits and security notes

- The adapter follows the model's pinned protocol and does not silently truncate accepted input or impose a default aggregate input cap. Applications accepting untrusted input can configure `maxInputTokens` and `maxInputCharacters` and control request size and concurrency at their boundary.
- `AbortSignal` can cancel queued work but cannot interrupt inference already dispatched to WebGPU.
- `assetsUrl` is application configuration. Do not derive it from user input; the loader requests the manifest before it can verify the manifest hash. Decision input is not sent to the asset host.
- The package checks the manifest and asset hashes before use. This does not settle model-weight or training-data redistribution rights.
- The stable package supports one model route. Other model routes in the development workspace are outside this release.

## Previous preview

The earlier `web-decision@0.1.0-preview.1` archive and evidence remain under [`releases/web-decision-0.1.0-preview.1`](releases/web-decision-0.1.0-preview.1). The stable archive has its own verification evidence and does not relabel the preview's archive hash.

## Build and release checks

The stable source lock, build, package archive, and browser evidence are tracked under [`releases/web-decision-0.1.0`](releases/web-decision-0.1.0). The security review is recorded separately by Codex Security. See the [release procedure](.github/RELEASING.md). Project code is Apache-2.0; model and dataset terms remain separate. No model assets are attached to the release.
