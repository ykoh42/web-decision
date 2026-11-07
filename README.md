# web-decision

`web-decision` is a browser JavaScript library for running decision models locally. The latest published stable release is `0.2.0`, with Kev 0.6B and TAMEV Nano routes.

## Install

```sh
npm install web-decision@0.2.0
```

The npm package contains code and notices, not model weights or runtime assets. Applications host the separately prepared asset packs:

| Model | Runtime | Inference asset bytes |
| --- | --- | ---: |
| Kev 0.6B Q4F16 | TVM WebGPU (`shader-f16`) | 349,103,828 |
| TAMEV Nano TinyBERT INT8 | ONNX Runtime WebAssembly CPU | 15,585,045 |

See the [package guide](releases/web-decision-0.2.0/package/README.md), [asset rebuild instructions](releases/web-decision-0.2.0/README.md), and [release evidence](RELEASE_SCOPE.md).

## Verified scope

The exact `0.2.0` archive was tested with real models on one Apple M4 Pro in Chromium, Firefox, and WebKit. Kev used a non-fallback WebGPU adapter with `shader-f16`; TAMEV Nano used ONNX Runtime WebAssembly on CPU. The tests establish runtime and API behavior on those browsers, not decision quality or compatibility across all devices. Archive SHA-256: `bd29861052f7de6d8a39f60f78253cc36ca6980e7ae0a104f09009354ee798f4`.

## `0.3.0` candidate

The next release adds exactly one route, TAMEV Micro MiniLM INT8, and retains the existing routes. Its exact archive passed nine real-model cases across Chromium 158.0.8127.18, Firefox 159.0, and WebKit 27.2 on one Apple M4 Pro. See the [candidate archive and browser evidence](releases/web-decision-0.3.0/verification/package-archive.json). The candidate is still pending publication.

## Runtime requirements and safety

Kev requires browser WebGPU with `shader-f16`. It preserves input accepted by the model protocol and supports optional token and character limits. TAMEV follows its pinned tokenizer and model-specific truncation behavior, which is reported in the result. Applications that pass untrusted input should set their own request-size and concurrency limits.

Use an application-configured asset URL. Do not let untrusted input choose `assetsUrl`. Decision inputs stay in browser-side tokenization and inference and are not sent to the asset host. The package verifies pinned model asset bytes before use.

## License

Project code is Apache-2.0. Model assets and training data have separate terms; the npm package does not redistribute model weights or establish their broader licensing. See the route notices in the [0.2.0 package](releases/web-decision-0.2.0/package/).
