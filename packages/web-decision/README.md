# web-decision

`web-decision` runs decision models in the browser. The model computes the answer on the user's device; the library downloads model assets from pinned sources or a trusted application-configured mirror and verifies their hashes.

This workspace package is `0.0.0-preview.34` and is not the stable npm release. The latest stable release is `web-decision@0.3.0`; its supported scope is listed in the repository's [main README](../../README.md).

## Try the preview package

```js
import {loadDecisionModel} from 'web-decision';

const model = await loadDecisionModel({model: 'experimental-tamev-micro-minilm-int8'});

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

The experimental Strands Qwen 2B route uses TVM WebGPU with FP16 assets: 3.78 GB of tensor data,
up to 4,096 tokens per question, and up to 255 options. It has been verified on Apple M4 Pro with
hardware WebGPU in Chromium 158; other adapters must meet the route's reported GPU limits. The
publisher declares Apache-2.0 and notes varied training-data source terms in its
[license](https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610/blob/216d1ccb46a9e9b036eb9187b248db98480f77d4/LICENSE.md)
and [model card](https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610/blob/216d1ccb46a9e9b036eb9187b248db98480f77d4/README.md).
The local result is in the [M4 Pro verification report](https://github.com/ykoh42/web-decision/blob/main/reports/strands-qwen2-m4pro-verified-real-webgpu.json).

## Model assets

TAMEV Nano TinyBERT and Micro MiniLM load from their pinned public revisions when given only their model IDs. To use a mirror, pass its trusted base URL as `assetsUrl`; the files must match the pinned hashes. Other routes may still require a separately hosted asset pack, so check the route's documentation and options. Do not let untrusted input choose `assetsUrl`. The library verifies assets before use and keeps decision inputs in the browser.

The separate `experimental-kev-0.6b-q4f16-upstream` model ID uses TVM WebGPU and downloads its pinned upstream ONNX weights on first use. It is an experimental build with its own executable and does not replace the existing `kev-0.6b-q4f16` route. It currently does not accept an `assetsUrl` override. Its first load downloads about 335 MB and needs substantial browser memory; check GPU requirements and verification status in `describeModel`.

Browser caching is enabled by default for TAMEV. Once all verified files are cached, pass `offline: true` to require an offline load. The application must also make its own code and pages available offline.

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
