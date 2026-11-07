# web-decision

`web-decision` is a browser JavaScript library for running local decision models. The first public preview, `0.1.0-preview.1`, supports one model: Kev 0.6B Q4F16 through TVM WebGPU.

## Install

```sh
npm install https://github.com/ykoh42/web-decision/releases/download/v0.1.0-preview.1/web-decision-0.1.0-preview.1.tgz
```

The package is code-only. The 14 model runtime assets (about 349 MB) must be prepared separately and hosted by the application. See the [package guide](releases/web-decision-0.1.0-preview.1/package/README.md) and [rebuild instructions](releases/web-decision-0.1.0-preview.1/README.md).

## Verified scope

The exact release archive was tested with real WebGPU inference in Chromium 156.0.8078.4, Firefox 157.0, and WebKit 27.2 on one Apple M4 Pro. Each browser passed 11/11 cases, including 8,192-token generic and typed inputs, token boundaries, ordered options, Choice/Noul/Score outputs, and aggregate requests. These runs do not establish support for other devices, browser versions, model quality, or general decision accuracy. See [release scope and evidence](RELEASE_SCOPE.md).

## Runtime requirements and safety

The Kev route requires browser WebGPU with `shader-f16`. The model accepts up to 8,192 tokens per state and state-plus-question row. The library does not impose a default aggregate token, character, or question-count cap, and does not silently truncate accepted input. Applications handling untrusted input should configure the optional `maxInputTokens` and `maxInputCharacters` limits and apply request-size and concurrency controls themselves. An abort signal can stop queued work but does not preempt GPU inference that has already started.

Use an application-configured asset URL. Do not let untrusted input choose `assetsUrl`; loading begins with a browser request for the manifest. Decision inputs are not sent to the asset host, and pinned asset hashes are verified before use.

## License

Project code is Apache-2.0. Model assets and training data have separate terms; this release does not redistribute them or determine their licensing. Review [third-party notes in the package](releases/web-decision-0.1.0-preview.1/package/KEV06-NOTICE) before preparing or hosting model assets.
