# web-decision

`web-decision` runs decision models in the browser. The model computes the answer on the user's device; the library downloads model assets from an application-configured URL and verifies their pinned hashes.

This workspace package is `0.0.0-preview.34` and is not the stable npm release. The latest stable release is `web-decision@0.3.0`; its supported scope is listed in the repository's [main README](../../README.md).

## Try the preview package

```js
import {loadDecisionModel} from 'web-decision';

const model = await loadDecisionModel({
  model: 'experimental-tamev-micro-minilm-int8',
  assetsUrl: 'https://your-origin.example/models/tamev-micro-minilm/'
});

try {
  const result = await model.decide({
    state: 'A customer reports a duplicate charge.',
    questions: [{
      instructions: 'Which team should review this?',
      options: ['Billing', 'Support']
    }]
  });
  console.log(result.answers[0]?.selected);
} finally {
  await model.dispose();
}
```

Use `listDecisionModels()` to see registered routes and `describeModel(id)` to inspect each route's backend, input features, limits, and verification status. A registered `experimental-*` route is a preview adapter; it does not by itself mean that real-model inference has been verified on every browser or device.

Some models also expose typed `systemOne` inputs. Check the route descriptor before using typed questions, abstention, scores, or multimodal fields. The library preserves each model's option order and output meaning. Any truncation follows the model protocol and is reported in the result.

## Model assets

The npm package contains library code and notices, not model weights. Host the matching asset pack separately and pass its trusted base URL. Do not let untrusted input choose `assetsUrl`. The library verifies the pinned assets before use; it does not send decision inputs to an inference server.

Model files have their own terms. Review the model and asset terms for the files you choose to host or use.

## Browser requirements

A route's descriptor identifies its runtime. TVM WebGPU routes require the GPU features and limits reported by that route. ONNX Runtime Web may partition unsupported operators to CPU or WebAssembly, as documented by the route's provider configuration.

## Development

Run commands from the repository root with pnpm:

```sh
pnpm install --frozen-lockfile --ignore-scripts
pnpm test
pnpm check:quality
pnpm verify:package
```

The package build uses the pinned Apache TVM Web sources and Emscripten setup documented in the repository's package checks. Unit tests use synthetic runtime inputs unless a test explicitly requires local model assets. Local real-model browser results are reported separately from these tests.

The project code is Apache-2.0. See the package notices for bundled runtime and source attributions; model asset terms remain separate.
