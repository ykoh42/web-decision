# web-decision

Run curated decision models in the browser with pinned TVM WebGPU assets.

The current `main` API exposes the verified Kanana 2 V Decide model by ID:

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

`listModels()` and `describeModel(id)` expose the curated catalog. Assets are pinned and cached for offline reuse; caller-supplied model URLs are not accepted.

See the [basic example](examples/basic/README.md), [package guide](packages/web-decision/README.md), and [release procedure](.github/RELEASING.md).
