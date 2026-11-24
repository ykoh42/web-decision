# web-decision

`web-decision` runs curated decision models locally in the browser with TVM WebGPU. A model ID selects a pinned model and its verified assets. Applications cannot register an arbitrary model or asset URL.

| Model | ID |
| --- | --- |
| Kanana 2 V Decide 3B | `kanana-2-v-decide-3b-2611` |
| Strands Decider 2B Qwen3.5 | `strands-decider-2b-qwen3.5-v1-2610` |
| Mapika Decider 2B v12 | `mapika-decider-2b-v12` |
| Strands Decider E2B Gemma 4 v2 | `strands-decider-e2b-gemma4-v2-2611` |

`listModels()` returns the models in the installed package. `describeModel(modelId)` shows the pinned checkpoint, input limits, and verification metadata.

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

Kanana follows its pinned 20,480-token model limit. Set the per-request `maxLength` through `systemOne()` options. Kanana accepts up to 10 images per request, counting images in `state` messages and the `images` field together. The pinned model truncates overlong inputs by default and reports the affected query IDs and candidate keys in `inputTruncation`. Set `truncateInputs: false` to reject overlong input.

For JSON state and criteria, safe integer `number` values other than negative zero are rendered as Python integers; negative zero, fractional values, and unsafe `number` values are rendered as Python floats. `bigint` values remain exact integers. JavaScript cannot distinguish `2` from `2.0`; use `kananaFloat(2)` from `web-decision` when the Python prompt must contain `2.0`.

## Model files and offline use

The TVM graphs ship with the package. The loader fetches the pinned weight ranges required by the graphs, and decision inputs remain local. Use `cache: true` to save verified model assets and `offline: true` to use cached assets. Available storage and browser quotas determine what the browser can save. The application also needs to make its own code and pages available offline.

The models require WebGPU. The catalog describes model-specific limits and the release record identifies the browser and hardware used for exact-package verification.

Mapika loads the text runner first. It fetches embedding ranges and image assets as requests need them, and caches verified downloads by default. Its `systemOne()` accepts the pinned Decider request format, including `choice`, `score`, `noul`, and PNG or JPEG images.

To prepare Mapika for offline use, call `await model.prepareOffline({signal, onProgress})` while online. Both options are optional. A complete model cache uses about 4.5 GB of browser storage. Check `model.cacheStatus` for `'complete'`, then pass `offline: true` when loading the model offline.

Strands Gemma 4 v2 accepts keyed `choice`, `noul`, and `score` questions and base64 images. Use a `Map` to preserve the order of numeric-looking keys. Use `bigint` for exact large integers in structured content and `kananaFloat(2)` to render the float `2.0`. The default is 32,768 input tokens per question, counting state, image, and question tokens. Set `maxLength` at load time or for an individual `systemOne()` call. Actual capacity depends on the GPU and its buffer limits. Call `prepareOffline()` online and check `isOfflineReady()` before loading with `offline: true`. The pinned checkpoint contains about 10.25 GB of weights. See the packaged [license](./STRANDS-GEMMA4-V2-LICENSE) and [notice](./STRANDS-GEMMA4-V2-NOTICE) for attribution.

## Kanana model terms

The web-decision code is Apache-2.0. Kanana model materials and their derivatives use Kakao's Kanana Open License. This package includes the license copied from the pinned model revision in [`KANANA-OPEN-LICENSE.txt`](./KANANA-OPEN-LICENSE.txt) and its required [`KANANA-NOTICE`](./KANANA-NOTICE). The asset manifest identifies web-decision as the generator of the TVM graphs and weight manifests, lists those modified files, and records their source and hashes.

When redistributing Kanana or its derivatives, the agreement requires passing on its use restrictions, notifying subsequent users, providing a copy of the agreement, marking modified files, including the notice, and showing a clearly visible “Powered by Kanana” attribution in related interfaces or documentation. Section 2.2 incorporates Kakao's **Guidelines For Responsible AI**. Kakao's current [AI Hub](https://www.kakaocorp.com/page/ai?lang=KOR) publishes its responsible-AI guidance; the pinned agreement still contains an older guidance URL. The agreement also requires a separate Kakao commercial license for offering or reselling Kanana or derivatives through remote APIs or cloud services, in system-integration or on-premise deployments, or embedded in on-device domains. This package includes compiled browser/on-device graphs, so whether that clause applies to its distribution needs confirmation from Kakao. The basic example displays the attribution when Kanana is selected; other applications must provide their own visible attribution.

## Strands Qwen3.5 model terms

The Strands checkpoint and Qwen base model use Apache-2.0. The package includes the [Strands license](./STRANDS-QWEN35-LICENSE.md), [Qwen base license](./QWEN-APACHE-LICENSE.txt), and [notice for the TVM adaptation](./STRANDS-QWEN35-NOTICE).
