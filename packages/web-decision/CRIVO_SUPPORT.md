# Crivo-32M v2 support scope

This npm package includes experimental adapters for two pinned Crivo graph variants. It does not include model weights. The model and converted assets carry CC BY-NC 4.0, require attribution, and cannot be used commercially under that license. Read [`CRIVO-NOTICE`](./CRIVO-NOTICE) and the complete upstream terms before hosting assets.

The current FP32 source bundle reports `currentBundleGpu: not-revalidated`. Historical preview.34 results below are scoped to their recorded archive and installed bundle; they do not verify the current source after its request-limit changes.

## Routes

| Model ID                         | Backend                              | Graph                       | Asset bytes | Current verification                                              |
| -------------------------------- | ------------------------------------ | --------------------------- | ----------: | ----------------------------------------------------------------- |
| `experimental-crivo-32m-v2-fp32` | ONNX Runtime WebGPU                  | `onnx/model.onnx`           | 128,385,184 | Current bundle not revalidated; historical preview.34 check below |
| `experimental-crivo-32m-v2-int8` | ONNX Runtime WebAssembly, one thread | `onnx/model_quantized.onnx` |  32,729,616 | Historical preview.34 Chromium WASM fixture check passed          |

The tokenizer and configuration are SHA-256 checked by the loader. The source checkpoint is `tardellirs/crivo-32m@77d61fdb6421838320097d9bcb3e4e42a120323e`. The browser conversion is `dizyann/crivo-web@fb3cf4bc66c06e51e08950399653e54a4ddeb154`, `models/32m/`.

## API behavior and limits

- Choice preserves caller key order and descriptions; ties select the first option.
- Score preserves ordered levels and returns the probability-weighted level index.
- Noul returns the probability that a statement is true; it does not abstain.
- Each question is encoded independently. `tokenCount` is the sum of encoded question rows.
- There is no fixed question-count or option-count ceiling. Choice and Score require at least two options; the row token limit bounds how many can fit. Noul always has two options.
- A complete question row is limited to 2,048 tokens. Overlong rows are rejected without truncation.
- The runner splits requests into batches of at most 32 rows and at most 65,536 padded tokens per inference dispatch. This is a per-dispatch memory budget, not an aggregate request limit. Preparation and result memory grow with the number of rows.
- State objects are serialized as compact JSON with keys sorted by Unicode code point; strings are trimmed.
- The checkpoint is English-only. No independent task-quality evaluation has been completed.

## Historical preview.34 installed-consumer evidence

On 2026-11-04, the npm-packed preview.34 archive was installed in a clean pnpm consumer; its installed runtime bundle SHA-256 was `4f7585dd2f55caf2d89cd625e9808129de84431be702bbc0682d7bd2f8072819`. That historical package ran the three pinned converter cases (six question rows) in Chromium 156 on an Apple M4 Pro. INT8 WASM used one thread and its maximum probability difference from the converter fixture was `0.01438`. FP32 WebGPU used a non-fallback Apple `metal-3` adapter and its maximum probability difference was `0.005071`. Both runs completed all asset downloads and preserved Choice, Score, and Noul outputs. The FP32 reference tolerance is 0.01, with explicit checks for the selected Choice and Score/Noul semantics. Per-run load times, five-sample warm latency, and host measurements are in the consumer reports; these values vary by run. This bundle predates the current request-limit changes and does not verify the current FP32 bundle.

The configured memory and disk guard passed for FP32. ONNX Runtime warned that some shape-related nodes were assigned to CPU; WebGPU selection does not mean every graph node ran on GPU. The npm tarball contains the adapter and notice, not model files. The historical preview.34 run checked 255 accepted and 256 rejected options, plus 256 accepted and 257 rejected questions. Those package count ceilings were removed from current source, so these runs do not verify larger option sets or requests on the current source. These checks cover one browser, one M4 Pro, pinned fixtures, and runtime behavior only. They do not measure independent task quality, other devices, or redistribution rights beyond the stated CC BY-NC 4.0 terms.

See the workspace [candidate record](../../experiments/crivo-32m-v2-candidate-20261104/README.md) for source pins, asset hashes, fixture details, reports, and remaining release gates.
