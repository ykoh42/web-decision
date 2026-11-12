# web-decision

Run curated decision models in the browser with TVM WebGPU. This package currently exposes `kanana-2-v-decide-3b-2611`; it loads only pinned model files and verifies their hashes.

## Use Kanana

```ts
import {loadDecisionModel} from 'web-decision';

const model = await loadDecisionModel({modelId: 'kanana-2-v-decide-3b-2611'});
try {
  const result = await model.systemOne({
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
  await model.dispose();
}
```

The model supports text and image decisions, typed `choice`, `noul`, and `score` questions, and explicit abstention. Inputs preserve the model's 20,480-token limit; overlong inputs are truncated by default and reported in `inputTruncation`. Set `truncateInputs: false` to reject them. Up to 10 images are accepted per request across state messages and the `images` field.

Use `offline: true` only after the required pinned weights have been cached. Image URLs must be fetchable by the browser; Blob and File inputs can be used directly.

Applications accepting untrusted inputs should bound image download sizes and concurrent requests before passing them to the model.

## Requirements and terms

The browser must provide compatible WebGPU limits and enough memory for the selected model. The maximum-context run was measured at about 11.09 GB of live WebGPU buffers on an Apple M4 Pro; this does not establish support on other hardware.

The library code is Apache-2.0. Kanana model files and derivatives have separate terms in [`KANANA-OPEN-LICENSE.txt`](./KANANA-OPEN-LICENSE.txt), which contains a copy of the agreement. Use must follow Kakao’s [Guidelines For Responsible AI](https://www.kakaocorp.com/page/responsible/detail/guidelinesForResponsibleAI). The manifest lists the modified files; [`KANANA-NOTICE`](./KANANA-NOTICE) provides the required attribution. Applications using the model must display “Powered by Kanana.” Commercial offerings that embed Kanana or derivatives in on-device domains may require a separate Kakao license. Users are responsible for assessing and obtaining permissions for their use.
