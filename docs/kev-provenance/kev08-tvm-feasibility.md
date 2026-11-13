# Kev 1.0 0.8B TVM feasibility probe

Updated 2026-11-13 KST. This records a small compiler-feasibility probe only. It did not load Kev or Qwen checkpoint weights and is not model evidence.

## Setup

- Kev model-card revision [`bf75a6a`](https://huggingface.co/jaredpalmer/kev-0.8b/blob/bf75a6a8848ea6960ff2ed108d9ed44c2941174f/README.md)
- Qwen base revision [`dc7cdfe`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/tree/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68)
- Python 3.11.11, PyTorch 2.8.0, Transformers 5.22.2, TVM 0.27.0.post1 from the repository's pinned build environment
- Randomly initialized, reduced Qwen3.5 text configurations with a 64-token vocabulary and one or four layers; BF16 hidden-state inputs; no Kev adapter or pointer head

## Results

| Stage | Result |
|---|---|
| PyTorch `torch.export` | Passed for the reduced BF16 text model |
| TVM Relax import with `from_exported_program` | Passed |
| TVM LLVM build | Failed in LLVM code generation with a BF16 type mismatch in `CreateAdd` |
| Default TVM WebGPU build | Failed with `CodeGenWebGPU: Cannot convert type boolx4 to WebGPU type` |
| WebGPU build with `tirx.disable_vectorize` | Passed the bool-vector failure, then failed because WebGPU code generation does not support `i64`; supplying `int32` external position IDs did not remove the internal `i64` |

The vectorization-disabled run is diagnostic only. It is not a performance candidate. No TVM source was patched. These failures show that this direct export does not yet build for either target; they do not prove that a repository-local, model-specific lowering cannot work.

## Scope and release status

The probe used no checkpoint weights, adapter, head, tokenizer, official input encoder, score calibration, or full serving limits. It did not produce a model artifact or run numerical parity, browser GPU inference, package, cache, or offline checks. It does not validate Kev 1.0.

The target remains the publisher's BF16 serving path with its original composition, semantics, and supported inputs, as specified in [`release-direction.md`](release-direction.md). Any further work must use repository-local TVM lowering without patching TVM or changing weights, precision, behavior, or limits. Do not expose the model by ID until full-reference parity and package/browser/offline gates pass.
