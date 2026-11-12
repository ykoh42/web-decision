# web-decision

Latest npm release: `0.3.0`. Run decision models locally in the browser.

The clean model ID interface below is on `main`; it is not included in npm `0.3.0` yet.

## Models in npm 0.3.0

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

The published package includes library code and notices. Its model files must be hosted at an application-controlled URL. See the [0.3.0 setup guide](releases/web-decision-0.3.0/package/README.md) for installation, asset links, and protocol details.

## Kanana candidate on main

Kanana is implemented and verified on `main`; it is not included in npm `0.3.0` yet.

```ts
import {loadDecisionModel} from 'web-decision';

const model = await loadDecisionModel({modelId: 'kanana-2-v-decide-3b-2611'});

try {
  const result = await model.systemOne({
    state: 'A customer reports a duplicate card charge.',
    questions: {
      team: {
        type: 'choice',
        instructions: 'Which team should review this?',
        criteria: {billing: 'Billing', fraud: 'Fraud'}
      }
    }
  });
  console.log(result.answers.team?.choice);
} finally {
  await model.dispose();
}
```

The workspace preview exposes only curated TVM models. `listModels()` and `describeModel(id)` show model capabilities and runtime requirements; `loadDecisionModel({modelId})` loads only a listed ID. Caller-supplied model URLs and unregistered models are not accepted.

The Kanana `0.4.0` candidate uses native TVM text and vision graphs, pinned assets, and only the checkpoint ranges those graphs require. Its exact candidate archive passed text, image, offline-cache, and 20,480-token-boundary WebGPU checks on an Apple M4 Pro with Chromium 157. The archive is bound to the reports in the release verification gate before publication; it is not part of published npm `0.3.0`.

## Verification

Release `0.3.0` was tested with real models in Chromium, Firefox, and WebKit on an Apple M4 Pro. See the [GitHub release](https://github.com/ykoh42/web-decision/releases/tag/v0.3.0) and [verification record](releases/web-decision-0.3.0/verification/package-archive.json).

For Python development checks, see [Python development tools](docs/python-quality.md).

See [LICENSE](LICENSE) and the package notices for project, model, and runtime terms.
