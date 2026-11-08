# web-decision

`web-decision` runs curated decision models locally in the browser using TVM. A model ID selects a package-pinned model and its verified assets. Applications cannot register an arbitrary model or asset URL.

This workspace is a preview. The latest published stable package and its supported models are documented in the [repository README](https://github.com/ykoh42/web-decision#readme).

```js
import {describeModel, listModels, loadDecisionModel} from 'web-decision';

console.log(listModels());
console.log(describeModel('kev-0.6b-q4f16'));

const model = await loadDecisionModel({modelId: 'kev-0.6b-q4f16'});
try {
  const result = await model.decide({
    state: 'A customer reports a duplicate charge.',
    questions: [{instructions: 'Which team should review this?', options: ['Billing', 'Support']}]
  });
  console.log(result.answers[0]?.selected);
} finally {
  await model.dispose();
}
```

## Model files and offline use

Kev uses the TVM WebGPU model WASM and metadata included in the package. On first load it fetches the pinned 335,357,952-byte upstream external-data file and reconstructs the TVM weight shards. The loader verifies the source, mapping and each shard against pinned hashes before use. Decision inputs remain local.

Verified shards can be cached for subsequent loads. `offline: true` requires all necessary cached files and does not fetch missing model assets. A successful online load alone does not guarantee that the browser saved the files: storage availability and quota vary. For offline use, the application must also make its own code and pages available offline.

The model requires WebGPU and `shader-f16`. GPU, browser and model-specific limitations and actual verification evidence are described in the model catalog and release record. Model files have separate terms; see the model notices.

## Development

Run from the repository root:

```sh
pnpm install --frozen-lockfile --ignore-scripts
pnpm test
pnpm check:quality
pnpm verify:package
```

The package build uses pinned Apache TVM Web sources and Emscripten setup. Unit tests do not by themselves prove real-model GPU execution. A model is eligible for release only after the exact candidate package and its artifacts pass the release verification gates.
