# web-decision

`web-decision@0.3.0` supports three browser-local decision models:

- `kev-0.6b-q4f16` uses TVM WebGPU and requires `shader-f16`.
- `experimental-tamev-nano-tinybert-int8` uses ONNX Runtime WebAssembly on CPU with one inference thread.
- `experimental-tamev-micro-minilm-int8` uses ONNX Runtime WebAssembly on CPU with one inference thread.

Each model uses its own pinned asset directory. The package contains JavaScript, declarations, runtime dependencies and notices; it does not contain model weights.

## Install

```sh
npm install web-decision@0.3.0
```

## Kev 0.6B

Host the exact Kev asset pack at a URL your application can fetch, then pass its directory URL to the loader:

```ts
import {loadDecisionModel} from 'web-decision';

const model = await loadDecisionModel({
  assetsUrl: 'https://your-origin.example/models/kev-0.6b-q4f16/'
});

try {
  const result = await model.decide({
    state: 'A customer requests a refund for a damaged item.',
    questions: [{instructions: 'Should the refund be approved?', options: ['Approve', 'Decline']}]
  });
  console.log(result.answers[0]?.selected);
} finally {
  await model.dispose();
}
```

The Kev loader verifies its pinned manifest and asset hashes. The exact `0.1.0` archive passed real WebGPU tests in Chromium, Firefox and WebKit on one Apple M4 Pro. Those results apply to that archive and test device.

## TAMEV Nano TinyBERT

Copy the four files from the pinned [TAMEV Nano model revision](https://huggingface.co/Tamkimd/tamev-nano-tinybert/tree/ad15b45f35526070d48fb91f6597d2805a2d80af) to a directory your application can fetch. The required files total 15,585,045 bytes: `model_int8.onnx`, `tokenizer.json`, `tokenizer_config.json` and `config.json`.

```ts
import {loadDecisionModel} from 'web-decision';

const model = await loadDecisionModel({
  model: 'experimental-tamev-nano-tinybert-int8',
  assetsUrl: 'https://your-origin.example/models/tamev-nano-tinybert/'
});

try {
  const result = await model.systemOne(
    {payment: 'duplicate card charge'},
    {
      route: {
        type: 'choice',
        instructions: 'Select the handling queue',
        criteria: {billing: 'review a duplicate charge', fraud: 'investigate unauthorized use'}
      },
      urgent: {type: 'noul', instructions: 'Does the customer need immediate help?'},
      severity: {type: 'score', instructions: 'Rate the impact', criteria: ['low', 'medium', 'high']}
    }
  );
  console.log(result.answers.route?.choice);
} finally {
  await model.dispose();
}
```

The TAMEV adapter verifies each downloaded file against its pinned byte length and SHA-256 hash. It preserves option and question order, supports Choice, Noul and Score, and uses the model's calibration for option counts with a configured calibration entry. Noul returns the model's yes probability; it does not abstain.

TAMEV Nano runs through ONNX Runtime WebAssembly, not WebGPU. Its context rows allow 128 tokenizer tokens and each option row allows 48; excess text is right-truncated and reported. The package imposes no fixed option-count or question-count ceiling. Questions run sequentially and practical request size depends on available device memory.

The upstream model card reports its own quality metrics and warns that external tasks may differ. `web-decision` has verified browser execution and input/output behavior; it does not independently validate decision quality.

## TAMEV Micro MiniLM

Copy the four files from the pinned [TAMEV Micro model revision](https://huggingface.co/Tamkimd/tamev-micro-minilm/tree/c1b95679390911748aad57f634bed70e5f87a084) to a directory your application can fetch. The required files total 24,161,003 bytes: `model_int8.onnx`, `tokenizer.json`, `tokenizer_config.json` and `config.json`.

```ts
import {loadDecisionModel} from 'web-decision';

const model = await loadDecisionModel({
  model: 'experimental-tamev-micro-minilm-int8',
  assetsUrl: 'https://your-origin.example/models/tamev-micro-minilm/'
});

try {
  const result = await model.systemOne(
    {payment: 'duplicate card charge'},
    {
      route: {
        type: 'choice',
        instructions: 'Select the handling queue',
        criteria: {billing: 'review a duplicate charge', fraud: 'investigate unauthorized use'}
      },
      urgent: {type: 'noul', instructions: 'Does the customer need immediate help?'},
      severity: {type: 'score', instructions: 'Rate the impact', criteria: ['low', 'medium', 'high']}
    }
  );
  console.log(result.answers.route?.choice);
} finally {
  await model.dispose();
}
```

The adapter verifies each downloaded file against its pinned byte length and SHA-256 hash. It preserves option and question order, supports Choice, Noul and Score, and applies the model's exported probabilities plus its configured per-option-count calibration to `decide()`. Noul returns the model's yes probability; it does not abstain.

TAMEV Micro runs through ONNX Runtime WebAssembly, not WebGPU. Its context rows allow 128 tokenizer tokens and each option row allows 48; excess text is right-truncated and reported. The package imposes no fixed option-count or question-count ceiling. Questions run sequentially and practical request size depends on available device memory. The upstream model card's quality figures are not independently reproduced by this project.

## Asset hosting and safety

The application must host model files under the supplied `assetsUrl` and allow browser requests from its origin. Keep the URL in trusted application configuration: the loader fetches assets from it. Decision inputs are not sent to the asset host. Model assets are not part of the npm archive; the package includes the TAMEV model notices and license files.

The Kev route also excludes its 14 inference assets. Its pinned manifest totals 349,103,828 bytes. `KEV06-NOTICE` and `KEV-NOTICE` cover the route and adapted code; review the terms that apply to the separately hosted asset files.

## API

- `loadDecisionModel({assetsUrl})` loads Kev 0.6B by default.
- `loadDecisionModel({model: 'experimental-tamev-nano-tinybert-int8', assetsUrl})` loads TAMEV Nano TinyBERT.
- `loadDecisionModel({model: 'experimental-tamev-micro-minilm-int8', assetsUrl})` loads TAMEV Micro MiniLM.
- All three sessions provide `decide()`, `releaseCachedMemory()` and `dispose()`. They expose typed `systemOne()` results for their supported protocols.
- Use `listDecisionModels()`, `isDecisionModel()` and `describeModel()` to inspect the three pinned routes.
- Use `clearModelCache()` to clear this library's asset cache for the current origin.

The package code is licensed under Apache-2.0. Model terms and runtime asset terms remain separate; see the included notices.
