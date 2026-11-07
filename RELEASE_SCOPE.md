# Release scope and verification

## `web-decision@0.1.0`

This stable release supports exactly one browser-local model route: `kev-0.6b-q4f16`, using TVM WebGPU with the `shader-f16` feature. The npm archive contains 13 code and notice files; it excludes model weights, tokenizer files, and runtime assets. Applications must prepare and host the pinned 14-file asset pack separately; its inference files total 349,103,828 bytes.

The exact npm archive SHA-256 is `8d4eec6db8688c6b13d1386ec9222addaf82b8a8cd26de900e8096d39d9b65d0`. The matching bundle and real-model GPU results are recorded in [`releases/web-decision-0.1.0/verification`](releases/web-decision-0.1.0/verification).

GPU runs used one Apple M4 Pro and a non-fallback adapter with `shader-f16` in all browsers:

| Browser | Version | GPU cases | Minimum available host memory sampled |
| --- | --- | ---: | ---: |
| Chromium | 156.0.8078.4 | 11/11 | Not sampled during this run |
| Firefox | 157.0 | 11/11 | 18.48 GiB |
| WebKit | 27.2 | 11/11 | 17.52 GiB |

The suite covered generic and typed 8,192-token inputs, token lengths 4,096, 6,144, 7,168, 8,191, and 8,192, aggregate requests, Choice/Noul/Score semantics, and option order. All results apply to the exact archive and bundle above. They do not establish compatibility with other devices, operating systems, or browser versions, nor decision accuracy or general task quality.

## Limits and security notes

- The adapter follows the model's pinned protocol and does not silently truncate accepted input or impose a default aggregate input cap. Applications accepting untrusted input can configure `maxInputTokens` and `maxInputCharacters` and control request size and concurrency at their boundary.
- `AbortSignal` can cancel queued work but cannot interrupt inference already dispatched to WebGPU.
- `assetsUrl` is application configuration. Do not derive it from user input; the loader requests the manifest before it can verify the manifest hash. Decision input is not sent to the asset host.
- The package checks the manifest and asset hashes before use. This does not settle model-weight or training-data redistribution rights.
- The stable package supports one model route. Other model routes in the development workspace are outside this release.

## Previous preview

The earlier `web-decision@0.1.0-preview.1` archive and evidence remain under [`releases/web-decision-0.1.0-preview.1`](releases/web-decision-0.1.0-preview.1). The stable archive has its own verification evidence and does not relabel the preview's archive hash.

## Build and release checks

The stable source lock, build, package archive, browser evidence, and security review are tracked under [`releases/web-decision-0.1.0`](releases/web-decision-0.1.0). See the [release procedure](.github/RELEASING.md). Project code is Apache-2.0; model and dataset terms remain separate. No model assets are attached to the release.
