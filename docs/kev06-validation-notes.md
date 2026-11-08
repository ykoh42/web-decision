# Kev 0.6B validation notes

These observations are exploratory evidence, not a release approval or a statement of model quality. They describe one Windows 11 machine using Chrome 156 and a non-fallback WebGPU adapter with `shader-f16`. The Windows display inventory identified Intel UHD Graphics; the browser did not expose an adapter name.

## TVM artifact reproducibility

The stable package contains a 6,807,315-byte WASM with SHA-256 `68fe0331180dc1adfb3d1e4295c06133815b3f5bfc5ad8dbd73ae66f3a02b8fb`. Independent builds with the pinned TVM revision, Emscripten version, source model and graph produced a 6,807,324-byte WASM with SHA-256 `ecc2bdce793221f4f370890d2ac4ddcb6082b85da05a645f3554903768b03cb0`. The original and refactored build scripts produced the latter hash, including when `PYTHONHASHSEED=0` and `SOURCE_DATE_EPOCH=0` were set. The input specialized ONNX graph and all nine weight shards were byte-identical across those builds.

The two WASMs differ in their code section. The build environment and execution record for the stable WASM were not found, so the cause is not established. Do not claim that the stable binary has been reproduced from the pinned build instructions. The seed rebuild is an experimental candidate and has not replaced the packaged asset.

## Exploratory output comparison

The stable and seed-rebuild WASMs returned identical logits, probabilities and selections on four hand-selected English inputs of 22, 64, 57 and 243 tokens. Both selected the same option as the pinned ONNX model run with ONNX Runtime 1.31.1 CPU execution, graph optimizations enabled and four threads. The rebuild bundle was assembled in an isolated scratch checkout with its own updated integrity metadata; a source-comment build guard was temporarily bypassed for this experiment. This comparison is not evidence for an exact released archive.

| Input | Maximum absolute logit difference from ONNX CPU | Maximum absolute probability difference from ONNX CPU |
| --- | ---: | ---: |
| Double charge | 0.07764 | 0.000101 |
| Account security | 0.05078 | 0.012235 |
| Delayed package | 0.04883 | 0.003878 |
| Long tracking context | 0.08203 | 0.008707 |

The existing upstream reference fixture uses logit and probability limits of `0.15` and `0.001` for **one specific input**. Those limits have not been established as a general acceptance policy for the other inputs; applying the same numeric comparison to the three added cases would exceed the probability limit. Browser and reference token IDs and option indices matched for all four requests.

A separate CPU experiment found identical results with one and four threads at ONNX Runtime's `ORT_ENABLE_ALL` setting, but substantially different logits when graph optimizations were disabled or limited to basic or extended levels. The cause is under investigation; those results must not be treated as interchangeable reference outputs.

Before release, resolve the build provenance and reference behavior, define an evidence-based numerical acceptance policy, and validate the exact release archive against representative inputs. Keep the existing single-input reference check and its limits until a reviewed replacement is available. Revalidate after changing a model artifact, its pins, or its runtime bundle.
