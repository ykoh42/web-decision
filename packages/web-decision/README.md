# web-decision

`web-decision` runs decision models in the browser. The model computes the answer on the user's device; the library downloads model assets from pinned sources or a trusted application-configured mirror and verifies their hashes.

This workspace package is `0.0.0-preview.34` and is not the stable npm release. The latest stable release is `web-decision@0.3.0`; its supported scope is listed in the repository's [main README](../../README.md).

## Try the preview package

```js
import {loadDecisionModel} from 'web-decision';

const model = await loadDecisionModel({modelId: 'tamev-micro-minilm-int8'});

try {
  const result = await model.decide({
    state: 'A customer reports a duplicate charge.',
    questions: [
      {
        instructions: 'Which team should review this?',
        options: ['Billing', 'Support']
      }
    ]
  });
  console.log(result.answers[0]?.selected);
} finally {
  await model.dispose();
}
```

Use `listStableModelIds()` for the stable-ID catalog and `describeStableModel(id)` to inspect a model's backend, input features, limits, and verification status. The IDs map to pinned model revisions and stay separate from internal adapter names. TAMEV Micro, Nano and Nano Flash load from their pinned public revisions without an `assetsUrl`; an optional URL is only a trusted mirror. Kev's derived TVM asset pack is not hosted at a canonical URL yet, so that model ID still requires an `assetsUrl`.

Use `StableModelId` values with `loadDecisionModel({modelId})` as the stable interface.

Some models also expose typed `systemOne` inputs. Check the route descriptor before using typed questions, abstention, scores, or multimodal fields. The library preserves each model's option order and output meaning. Any truncation follows the model protocol and is reported in the result.

## Model assets

Stable TAMEV model IDs load pinned public assets without an `assetsUrl`. To use a mirror, pass its trusted base URL; the files must match the pinned hashes. Kev still requires a separately hosted asset pack. The library verifies assets before use and keeps decision inputs in the browser.

Browser caching is enabled by default for stable TAMEV routes. Once all verified files are cached, pass `offline: true` to require an offline load. Caching depends on browser storage availability and quota. The application must also make its own code and pages available offline.

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

To run the real Kev 0.6B WebGPU boundary test, provide the verified asset-pack parent directory and run `WEB_DECISION_REAL_GPU=1 WEB_DECISION_KEV06_REAL_GPU=1 WEB_DECISION_PUBLIC_ASSET_ROOT=/path/to/model-assets pnpm test:browser:kev06`.

To run the Strands Qwen 2B hardware test, point the asset root at the pinned local model directory and
run `WEB_DECISION_REAL_GPU=1 WEB_DECISION_STRANDS_REAL_GPU=1 WEB_DECISION_PUBLIC_ASSET_ROOT=/path/to/strands-assets pnpm test:browser:strands-qwen2`.
The test report records the package bundle hash and GPU device.

The project code is Apache-2.0. See the package notices for bundled runtime and source attributions; model asset terms remain separate.
