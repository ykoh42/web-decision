# Kev 0.6B validation notes

These observations are exploratory evidence, not a release approval or a statement of model quality. They describe one Windows 11 machine using Chrome 156 and a non-fallback WebGPU adapter with `shader-f16`. The Windows display inventory identified Intel UHD Graphics; the browser did not expose an adapter name.

## TVM artifact reproducibility

The previously packaged stable WASM was 6,807,315 bytes with SHA-256 `68fe0331180dc1adfb3d1e4295c06133815b3f5bfc5ad8dbd73ae66f3a02b8fb`. Independent builds with the pinned TVM revision, Emscripten version, source model and graph produced a 6,807,324-byte WASM with SHA-256 `ecc2bdce793221f4f370890d2ac4ddcb6082b85da05a645f3554903768b03cb0`. The original and refactored build scripts produced the latter hash, including when `PYTHONHASHSEED=0` and `SOURCE_DATE_EPOCH=0` were set. The input specialized ONNX graph and all nine weight shards were byte-identical across those builds. This branch updates the stable package pins to the reproduced build candidate; the prior binary remains documented here as the historical baseline.

The two WASMs differ in their code section. The build environment and execution record for the historical stable WASM were not found, so the cause of its byte difference is not established. The new pins identify the repeatable candidate build; they do not establish provenance for the historical binary or authenticity of the source data.

## Candidate archive check

At commit `ba6262c7908b3e8b40a75631a00d4fbe0474cfee`, the exact npm archive with SHA-256 `a0958e02f1df70f7058f83af84eb2686cbc7fed22f799854804612632f9d0d8a` passed the Windows release evidence verifier. The installed archive ran one baseline request on Chrome 156 with a non-fallback WebGPU adapter and `shader-f16`; Windows reported an Intel UHD Graphics display device. The test measured all 14 cached assets, confirmed online inference and an offline rerun without network attempts, and matched the pinned historical and one-input ONNX CPU references. The Playwright JSON report SHA-256 was `c1fbe57ac3a9b4a19ceda69a0d7880e35298a80edd7c0dd739f75caceacfd718`.

The verifier also matched the package WASM to the separately built `ecc2bdce793221f4f370890d2ac4ddcb6082b85da05a645f3554903768b03cb0` artifact and its build report. This is evidence for that archive, one browser, one host, and one reference input; later package changes need a new check. It does not resolve the broader numerical and model-quality questions below.

## Exploratory output comparison

The historical stable and reproduced-candidate WASMs returned identical logits, probabilities and selections on four hand-selected English inputs of 22, 64, 57 and 243 tokens. Both selected the same option as the pinned ONNX model run with ONNX Runtime 1.31.1 CPU execution, graph optimizations enabled and four threads. The earlier comparison bundle was assembled in an isolated scratch checkout with its own updated integrity metadata; a source-comment build guard was temporarily bypassed for that experiment. This comparison is not evidence for an exact released archive or proof that this candidate package build has passed the current release gate.

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

## Upstream development sample

An exploratory run used ten records from the upstream [development partition](https://github.com/jaredpalmer/kev/blob/2074cc44256200bb37540103fa4572901faa10f3/evals/v7/decision-v7/development.jsonl) at commit `2074cc44256200bb37540103fa4572901faa10f3` (file SHA-256 `8d5765d7aec4d08c61854f4664ca79ec2ba44ed092e9b967eaf61ed86c496d9c`). Selection was deterministic: the first record for each of eight compositional families, the first legacy-policy yes/no record, and the first SST-5 score record. These were development examples, not an unbiased sample or the upstream locked test.

The ONNX Runtime CPU inputs used token IDs generated by the same tokenizer and packing source as the stable scratch bundle; the browser reported matching token counts. The highest-probability option agreed on all ten records, while the maximum absolute probability difference on an individual option was approximately 0.03434. Public API probabilities were rounded to four decimals. The cases covered 44–163 tokens and all three typed question kinds; the score comparison refers to its highest-probability level, not exact agreement on its expected score. This establishes neither model accuracy nor a general numeric tolerance and does not validate an exact release archive.

Before release, resolve the build provenance and reference behavior, define an evidence-based numerical acceptance policy, and validate the exact release archive against representative inputs. Keep the existing single-input reference check and its limits until a reviewed replacement is available. Revalidate after changing a model artifact, its pins, or its runtime bundle.
