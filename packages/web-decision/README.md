# web-decision

`web-decision` runs curated decision models locally in the browser using TVM. A model ID selects a package-pinned model and its verified assets. Applications cannot register an arbitrary model or asset URL.

`web-decision` runs decision models in the browser with TVM WebGPU. The repository README lists the supported models and their verification status.

The current package catalog includes `kev-0.6b-q4f16` and `kanana-2-v-decide-3b-2611`.

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

Load Kanana by its catalog ID and submit keyed typed questions with `systemOne()`:

```js
const kanana = await loadDecisionModel({modelId: 'kanana-2-v-decide-3b-2611'});
try {
  const result = await kanana.systemOne({
    state: 'A customer reports a duplicate charge.',
    questions: {
      team: {
        type: 'choice',
        instructions: 'Which team should review this?',
        criteria: {billing: 'Billing', support: 'Support'}
      }
    }
  });
  console.log(result.answers.team?.choice);
} finally {
  await kanana.dispose();
}
```

Kev-only load options (`maxPrefetchBytes`, `maxInputTokens`, and `maxInputCharacters`) are rejected for Kanana with `INVALID_INPUT`; they are not treated as Kanana limits. Kanana follows its pinned 20,480-token model limit, and callers may set the per-request `maxLength` through `systemOne()` options. It accepts up to 10 images per request, counting images in `state` messages and the `images` field together. The pinned model truncates overlong inputs by default; results report the affected query IDs and candidate keys in `inputTruncation`. Set `truncateInputs: false` to reject overlong input instead.

For JSON state and criteria, safe integer `number` values other than negative zero are rendered as Python integers; negative zero, fractional values, and unsafe `number` values are rendered as Python floats. `bigint` values remain exact integers. JavaScript cannot distinguish `2` from `2.0`; use `kananaFloat(2)` from `web-decision` when the Python prompt must contain `2.0`.

## Model files and offline use

Kev uses the TVM WebGPU model WASM and metadata included in the package. On first load it fetches the pinned 335,357,952-byte upstream external-data file and reconstructs the TVM weight shards. The loader verifies the source, mapping and each shard against pinned hashes before use. Decision inputs remain local.

Verified shards can be cached for subsequent loads. `offline: true` requires all necessary cached files and does not fetch missing model assets. A successful online load alone does not guarantee that the browser saved the files: storage availability and quota vary. For offline use, the application must also make its own code and pages available offline.

The model requires WebGPU and `shader-f16`. GPU, browser and model-specific limitations and actual verification evidence are described in the model catalog and release record. Model files have separate terms; see the model notices.

## Kanana model terms

The package project is Apache-2.0, while Kanana model materials use Kakao's Kanana Open License. This package includes the license copied from the pinned model revision in [`KANANA-OPEN-LICENSE.txt`](./KANANA-OPEN-LICENSE.txt) and its required [`KANANA-NOTICE`](./KANANA-NOTICE). The asset manifest identifies web-decision as the generator of the TVM graphs and weight manifests, lists those modified files, and records their source and hashes.

When redistributing Kanana or its derivatives, the agreement requires passing on its use restrictions—including Kakao's [Guidelines For Responsible AI](https://www.kakaocorp.com/page/responsible/detail/guidelinesForResponsibleAI)—and a copy of the agreement, marking modified files, including the notice, and showing a clearly visible “Powered by Kanana” attribution in related interfaces or documentation. The agreement also requires a separate Kakao commercial license for offering or reselling Kanana or derivatives through remote APIs or cloud services, in system-integration or on-premise deployments, or embedded in on-device domains. Check the agreement for your intended use. The basic example displays the attribution when Kanana is selected; other applications must provide their own visible attribution.

## Development

Run from the repository root:

```sh
pnpm install --frozen-lockfile --ignore-scripts
pnpm test
pnpm check:quality
pnpm verify:package
```

The package build uses pinned Apache TVM Web sources and Emscripten setup. Unit tests do not by themselves prove real-model GPU execution. A model is eligible for release only after the exact candidate package and its artifacts pass the release verification gates.
