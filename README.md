# web-decision

Latest stable: `0.3.0`. Run decision models locally in the browser.

## Install

```sh
npm install web-decision
```

## Models

| Model | Runtime | Model files |
| --- | --- | ---: |
| Kev 0.6B Q4F16 | TVM WebGPU (`shader-f16`) | 349.1 MB |
| TAMEV Nano TinyBERT INT8 | ONNX Runtime WebAssembly CPU | 15.6 MB |
| TAMEV Micro MiniLM INT8 | ONNX Runtime WebAssembly CPU | 24.2 MB |

The npm package includes library code and notices. Host the model files at an application-controlled URL. See the [model setup guide](releases/web-decision-0.3.0/package/README.md) for asset links and protocol details.

## Quick start

```ts
import {loadDecisionModel} from 'web-decision';

const model = await loadDecisionModel({
  model: 'experimental-tamev-micro-minilm-int8',
  assetsUrl: 'https://your-origin.example/models/tamev-micro-minilm/'
});

try {
  const result = await model.decide({
    state: 'A customer reports a duplicate card charge.',
    questions: [{
      instructions: 'Which team should review this?',
      options: ['Billing', 'Fraud']
    }]
  });
  console.log(result.answers[0]?.selected);
} finally {
  await model.dispose();
}
```

Decision requests run in the browser. The configured asset URL is used to fetch model files.

## Verification

Release `0.3.0` was tested with real models in Chromium, Firefox, and WebKit on an Apple M4 Pro. See the [GitHub release](https://github.com/ykoh42/web-decision/releases/tag/v0.3.0) and [verification record](releases/web-decision-0.3.0/verification/package-archive.json).

Project code is Apache-2.0. Model and runtime terms are listed in the package notices.
