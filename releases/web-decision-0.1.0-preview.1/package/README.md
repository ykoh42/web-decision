# web-decision preview

`web-decision@0.1.0-preview.1` runs the Kev 0.6B Q4F16 text decision model in a browser with TVM WebGPU. This preview exposes only `kev-0.6b-q4f16`.

## Install

Download the preview tarball and install it in your application:

```sh
npm install ./web-decision-0.1.0-preview.1.tgz
```

## Quick start

The model assets are separate from the package. Host the exact asset directory at a URL your browser application can fetch, then pass that directory URL to the loader.

```ts
import {loadDecisionModel} from 'web-decision';

const model = await loadDecisionModel({
  assetsUrl: 'https://your-origin.example/models/kev-0.6b-q4f16/'
});

try {
  const result = await model.decide({
    state: 'A customer requests a refund for a damaged item.',
    questions: [{
      instructions: 'Should the refund be approved?',
      options: ['Approve', 'Decline']
    }]
  });

  console.log(result.answers[0]?.selected);
} finally {
  await model.dispose();
}
```

Serve `manifest.json` and every file listed in it under the same asset directory. The loader verifies the pinned manifest and asset sizes and SHA-256 hashes before using them. Configure the host to allow browser requests from your application origin. The package does not download or redistribute weights.

`assetsUrl` must come from your application configuration. Do not pass a URL supplied by an untrusted user: loading begins with a browser GET for `manifest.json`, before its hash can be checked. Decision inputs are not sent to the asset host.

## Supported scope

- Browser-local text decisions through TVM WebGPU. The adapter requires the WebGPU `shader-f16` feature.
- Generic Choice decisions and the model's typed Choice, Noul, and Score protocol through `systemOne()`.
- 1–255 options per question. Native limits allow up to 8,192 tokens for a state including its marker, and up to 8,192 tokens per state-plus-question row. The default API does not impose a fixed aggregate token or question-count cap; device memory remains the runtime constraint.
- Inputs beyond the configured or native limits are rejected. The adapter does not silently truncate them.
- Generic Choice does not implement model-level abstention. The typed API exposes the model's native Noul result.

The loader does not impose a default aggregate token, character, or question-count cap. If your application accepts untrusted input, set `maxInputTokens` and `maxInputCharacters` and apply request-size and concurrency limits in the application. This preserves the model's native input range; device memory remains the runtime limit. An `AbortSignal` can stop a queued decision, but cannot interrupt inference after GPU work has started.

The exact preview archive was verified with real WebGPU inference in Chromium, Firefox, and WebKit on one Apple M4 Pro. The suite covers generic and typed 8,192-token inputs, token boundaries, ordered options, typed outputs, and aggregate stability. This is evidence for that artifact and device only; it does not guarantee performance or compatibility on other hardware.

## Model assets and terms

The package excludes the 14 inference assets. Their pinned manifest totals 349,103,828 bytes (about 349 MB). Obtain or prepare an asset pack matching the pinned manifest, host it yourself, and review the terms that apply to those files. The package does not host model weights or the separate model-asset notices.

The package code is licensed under Apache-2.0. The included `KEV-NOTICE`, `KEV06-NOTICE`, TVM, Emscripten, Vite, and Rolldown notices describe code and conversion attributions. They do not determine rights for training data or transformed model weights. The pinned Kev and Qwen model references are in `KEV06-NOTICE`; model assets remain subject to their own terms.

## API notes

- `loadDecisionModel({assetsUrl})` returns a session with `decide()`, typed `systemOne()`, `releaseCachedMemory()`, and `dispose()`.
- `systemOne()` preserves insertion order for native `Map` inputs and exposes `answerEntries` and `probabilityEntries` for ordered results.
- Probabilities are not guaranteed to be calibrated. This preview makes no broad task-quality claim.
- Use `describeModel('kev-0.6b-q4f16')` to inspect the pinned checkpoint, required WebGPU features, limits, and verification scope.
