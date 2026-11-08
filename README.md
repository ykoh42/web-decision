# web-decision

Latest stable: `0.3.0`. Run decision models locally in the browser.

## Install

```sh
npm install web-decision
```

## Models

Upstream model-card scores appear highest first within the same benchmark. They are not measurements from this library.

### TAMEV held-out set (`n=19,999`)

| Model | Top-1 | Runtime | Model files |
| --- | ---: | --- | ---: |
| [TAMEV Micro MiniLM INT8](https://huggingface.co/Tamkimd/tamev-micro-minilm/blob/c1b95679390911748aad57f634bed70e5f87a084/README.md) | 82.23% | ONNX Runtime WebAssembly CPU | 24.2 MB |
| [TAMEV Nano TinyBERT INT8](https://huggingface.co/Tamkimd/tamev-nano-tinybert/blob/ad15b45f35526070d48fb91f6597d2805a2d80af/README.md) | 80.93% | ONNX Runtime WebAssembly CPU | 15.6 MB |

### Kev transfer-v4 locked test

| Model | Top-1 | Runtime | Model files |
| --- | ---: | --- | ---: |
| [Kev 0.6B Q4F16](https://huggingface.co/jaredpalmer/kev-0.6b/blob/dece6dba8d43f0f7ded45e9f5b9df12474d90843/README.md) | 64.2% | TVM WebGPU (`shader-f16`) | 349.1 MB |

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

## Development direction

This describes ongoing development, not the current `0.3.0` release behavior.

- Prefer TVM WebGPU for browser inference while preserving supported ONNX paths; this is not a TVM-only migration.
- Target model-ID-only loading with pinned artifacts fetched and verified automatically, cached, and available offline after warmup. Keep `assetsUrl` optional for trusted mirrors as routes migrate; not every current route supports ID-only loading.
- Keep each route's backend and model behavior explicit. Never silently substitute a backend, model, or behavior.
- Retain real-model browser verification for backend changes and report the models and environments measured.

## Verification

Release `0.3.0` was tested with real models in Chromium, Firefox, and WebKit on an Apple M4 Pro. See the [GitHub release](https://github.com/ykoh42/web-decision/releases/tag/v0.3.0) and [verification record](releases/web-decision-0.3.0/verification/package-archive.json).

For Python development checks, see [Python development tools](docs/python-quality.md).

See [LICENSE](LICENSE) and the package notices for project, model, and runtime terms.
