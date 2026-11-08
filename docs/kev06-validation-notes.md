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

A separate CPU experiment found identical results with one and four threads at ONNX Runtime's `ORT_ENABLE_ALL` setting, but substantially different logits when graph optimizations were disabled or limited to basic or extended levels. The output shape, token IDs and option indices remained the same. The first observed intermediate difference was the first layer's MLP `down_proj` `MatMulNBits` output; the preceding attention and MLP inputs were bitwise identical.

In an isolated copy of the extended graph, replacing only the 28 down-projection scale casts with constant initializers containing the same FP32 values made all four outputs match the `ORT_ENABLE_ALL` results bitwise. Disabling `FuseFp16InitializerToFp32NodeTransformer` produced the extended-level output. Disabling `MatmulTransposeFusion`, `QuickGeluFusion` or `NchwcTransformer` did not change the `ORT_ENABLE_ALL` result. The ORT 1.31.1 [Q4 operator implementation](https://github.com/microsoft/onnxruntime/blob/v1.31.1/onnxruntime/contrib_ops/cpu/quantization/matmul_nbits.cc) distinguishes constant scales and includes an INT8 activation computation path; the tests did not directly observe its internal kernel selection. These results do not establish which output is more accurate or whether there is a runtime defect. Treat the optimization settings as part of the reference configuration.

An independent ONNX Runtime Web 1.31.1 session on the same Chrome machine selected the same option for all four inputs, but its largest absolute differences from ORT CPU were 0.3594 for logits and 0.0483 for probabilities. The session used both WebGPU and CPU partitions; the log recorded 1,493 nodes on WebGPU and 1,329 on CPU. This is a comparison result, not a GPU-only reference or an added product runtime.

## Typed and packed question probe

A separate exploratory Windows Chrome WebGPU run exercised one request containing a choice, a yes/no (`noul`) and a five-level score question, then each question separately. All three question types returned results, and the choice was `billing`. For the score question, packed probabilities were `[0.6869, 0.2977, 0.0152, 0.0001, 0]` and separate probabilities were `[0.6902, 0.2945, 0.0152, 0.0001, 0]`; these public values are rounded to four decimal places. This single request is not a general equivalence or quality claim.

The same packed and separate token IDs were independently confirmed against the JavaScript tokenizer and packing implementation. On ONNX Runtime 1.31.1 CPU with all graph optimizations and four threads, the score probabilities were identical for this request, while the choice probabilities differed by up to 0.00638 between packed and separate execution. The GPU choice probabilities were identical to four decimals; the yes/no probability was identical between packed and separate runs on both runtimes. Different execution shapes can therefore affect outputs, and the effect differs between these two runtime configurations. This probe does not identify the cause or establish an acceptable tolerance.

Before release, resolve the build provenance and reference behavior, define an evidence-based numerical acceptance policy, and validate the exact release archive against representative inputs. Keep the existing single-input reference check and its limits until a reviewed replacement is available. Revalidate after changing a model artifact, its pins, or its runtime bundle.
