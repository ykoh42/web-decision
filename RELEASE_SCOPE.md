# Release scope and verification

## `web-decision@0.1.0-preview.1`

This preview supports exactly one browser-local model route: `kev-0.6b-q4f16`. It uses TVM WebGPU and requires the `shader-f16` feature. The npm archive contains 13 code and notice files and excludes all model weights and tokenizer/runtime assets. Applications must prepare and host the pinned 14-file asset pack separately; its inference files total 349,103,828 bytes.

The release archive SHA-256 is recorded alongside the package's build and verification evidence in [`releases/web-decision-0.1.0-preview.1/verification`](releases/web-decision-0.1.0-preview.1/verification). GPU runs apply only to that exact archive and bundle on one Apple M4 Pro:

| Browser | Version | GPU cases |
| --- | --- | ---: |
| Chromium | 156.0.8078.4 | 11/11 |
| Firefox | 157.0 | 11/11 |
| WebKit | 27.2 | 11/11 |

The test suite covered generic and typed 8,192-token inference, token boundaries, aggregate requests, Choice/Noul/Score semantics, option ordering and stability. Every run used a non-fallback adapter with `shader-f16`; the records include run IDs, artifact hashes, and sampled host memory. These results do not establish compatibility with other GPUs, operating systems, browser versions, general decision accuracy, or task quality.

## Limits and security notes

- The adapter preserves the model's pinned protocol and does not silently truncate or impose a default aggregate input cap. Applications accepting untrusted input should configure `maxInputTokens` and `maxInputCharacters` and limit request size and concurrency at their boundary.
- `AbortSignal` can cancel queued work but cannot interrupt an inference already dispatched to WebGPU.
- `assetsUrl` is application configuration. Do not derive it from user input; the loader requests the manifest before it can verify the manifest hash. Decision input is not sent to the asset host.
- The package checks the manifest and asset hashes before use. This does not settle model-weight or training-data redistribution rights.
- The public package is a preview of one model. Other model routes in the development workspace are outside this release.

## Build and release checks

The pinned source and bundle build, package archive, browser evidence, and security review are tracked under [`releases/web-decision-0.1.0-preview.1`](releases/web-decision-0.1.0-preview.1). See the [release procedure](.github/RELEASING.md). The project code is Apache-2.0; model and dataset terms remain separate. No model assets are attached to the release.
