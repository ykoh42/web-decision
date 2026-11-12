# web-decision

`web-decision` runs curated decision models locally in the browser with TVM WebGPU. A model ID selects a pinned model and its verified assets. Applications cannot register an arbitrary model or asset URL.

The current source catalog and `0.4.0` candidate include verified Kanana. It is not in published npm `0.3.0`. Kev's direct TVM candidate has passed source, conversion, full-input, and exact-package GPU checks. It remains outside the public catalog pending dataset-rights review and asset hosting.

Load Kanana by its catalog ID and submit keyed typed questions with `systemOne()`:

```js
import {loadDecisionModel} from 'web-decision';

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

Kanana's TVM graphs ship with the package, and the loader fetches only the pinned weight ranges required by those graphs. Decision inputs remain local. Verified assets can be cached for subsequent loads. `offline: true` requires all necessary cached files and does not fetch missing model assets. A successful online load alone does not guarantee that the browser saved the files: storage availability and quota vary. For offline use, the application must also make its own code and pages available offline.

The model requires WebGPU. GPU, browser and model-specific limitations and actual verification evidence are described in the model catalog and release record. Model files have separate terms; see the model notices.

## Kanana model terms

The web-decision code is Apache-2.0. Kanana model materials and their derivatives use Kakao's Kanana Open License. This package includes the license copied from the pinned model revision in [`KANANA-OPEN-LICENSE.txt`](./KANANA-OPEN-LICENSE.txt) and its required [`KANANA-NOTICE`](./KANANA-NOTICE). The asset manifest identifies web-decision as the generator of the TVM graphs and weight manifests, lists those modified files, and records their source and hashes.

When redistributing Kanana or its derivatives, the agreement requires passing on its use restrictions, notifying subsequent users, providing a copy of the agreement, marking modified files, including the notice, and showing a clearly visible “Powered by Kanana” attribution in related interfaces or documentation. Section 2.2 incorporates Kakao's **Guidelines For Responsible AI**. Kakao's current [AI Hub](https://www.kakaocorp.com/page/ai?lang=KOR) publishes its responsible-AI guidance; the pinned agreement still contains an older guidance URL. The agreement also requires a separate Kakao commercial license for offering or reselling Kanana or derivatives through remote APIs or cloud services, in system-integration or on-premise deployments, or embedded in on-device domains. This package includes compiled browser/on-device graphs, so whether that clause applies to its distribution needs confirmation from Kakao. The basic example displays the attribution when Kanana is selected; other applications must provide their own visible attribution.

## Development

Run from the repository root:

```sh
pnpm install --frozen-lockfile --ignore-scripts
pnpm test
pnpm check:quality
pnpm verify:package
```

The package build uses pinned Apache TVM Web sources and Emscripten setup. Unit tests do not by themselves prove real-model GPU execution. A model is eligible for release only after the exact candidate package and its artifacts pass the release verification gates.
