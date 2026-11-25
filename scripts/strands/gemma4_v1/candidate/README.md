# Private E2B v1 verification

Use Python 3.11.11, Torch 2.8.0, Transformers 5.22.2, PEFT 0.21.0, Strands Decider 13c0514 and TVM 0.27.0.post1. The original base revision is 3e22461f65e89153144f8adb70e3b8c2cc9845a7. The tokenizer, BF16 base, FP32 adapter and head are read from the exact original files. The adapter revision is d383ecf8534b13d49b3b567a1b66f88fca6d50ba. Set V1_BASE, V1_MODEL, TVM_SOURCE, EMSDK, PYTHON, and a fresh writable OUTPUT directory. Use the unchanged pinned TVM and Emscripten toolchains.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 TVM_NUM_THREADS=2
S=scripts/strands/gemma4_v1
"$PYTHON" -B "$S/source_inventory.py" "$V1_BASE" "$V1_MODEL" "$OUTPUT/source-inventory.json"
"$PYTHON" -B "$S/build_model.py" --base "$V1_BASE" --checkpoint "$V1_MODEL" \
  --inventory "$OUTPUT/source-inventory.json" --tvm-source "$TVM_SOURCE" --emsdk "$EMSDK" --output "$OUTPUT/graph"
node --import tsx "$S/candidate/build.mts" "$OUTPUT/graph" "$OUTPUT/source-inventory.json" "$V1_MODEL" "$OUTPUT/npm"
node --import tsx "$S/candidate/check-installed.mts" "$OUTPUT/npm/report.json"
```

The converter requires the exact range inventory generated from the verified original files. Its SHA-256 covers the repositories, tensor offsets and every chunk digest. The packer requires a clean source revision matching the graph manifest. The exact-installed check verifies archive, installed tree and bundle identity, the five packaged asset hashes, rejection of a modified graph and the missing-device error in Node. It starts no browser and requests no model weight ranges.

```sh
"$PYTHON" -B "$S/build_model.py" --base "$V1_BASE" --checkpoint "$V1_MODEL" \
  --inventory "$OUTPUT/source-inventory.json" --tvm-source "$TVM_SOURCE" --emsdk "$EMSDK" --cpu-wasm --output "$OUTPUT/graph-cpu"
"$PYTHON" -B "$S/capture_cpu_wasm_reference.py" "$V1_BASE" "$V1_MODEL" "$OUTPUT/cpu-reference.json"
node --import tsx "$S/check_cpu_wasm.mts" "$OUTPUT/graph-cpu" "$OUTPUT/source-inventory.json" \
  "$V1_BASE" "$V1_MODEL" "$OUTPUT/cpu-reference.json" --shared
node --import tsx --test packages/web-decision/test/strands-gemma4-v1-private.test.mts
```

The CPU comparison uses a four-token diagnostic input and the original BF16 model with an FP32 pointer. The shared mode exercises two identical branches from a single causal prefix. The test validates the graph ABI and works from verified local ranges. It creates no checkpoint copies. The package builder rejects the CPU-WASM target.

The browser bridge currently uses host transfers and serial layer reads. Actual WebGPU dispatch, runtime memory limits, long-context throughput, online loading and full browser offline restart still require exact-package device verification.

The default 4,096-token serving window comes from the pinned source checkpoint configuration. The private package also accepts an explicit loader option such as `maxLength: 32768` for the separately documented Decision Index window. It preserves the requested length and the model's causal positions across bounded prefill chunks. K/V storage and attention still grow with the visible sequence. With a single K/V head, a full 131,072-token attention tile needs at most a 268,435,456-byte single allocation and binding, in addition to other concurrent allocations. A 32,768-token request is bounded by the 75,497,472-byte static projection. The run checks actual device limits before reading inference weights. The manifest records the original Gemma text position capacity. Full-context end-to-end inference, peak memory and performance require separate measurements on the device.

The official shared `--vision` entry only loads v1 Qwen3.5 checkpoints. It rejects E2B v1's Gemma 4 architecture before loading weights. The Gemma base includes a vision tower, while its official E2B v1 serving integration uses the text decoder. [Official pinned loader](https://github.com/strands-labs/strands-decider/blob/6d85079caf1f1906ebb0f1370b56ef28ba7aa93e/src/strands_decider/vision.py#L89-L107)

The optional loader settings `maxBatch`, `maxBatchTokens` and `usePrefixCache` match the original v1 engine's question batching and usage policy. Defaults are 32 questions, no padded-token budget, and prefix reuse enabled. A nonzero padded-token budget can split an initial question batch. It still allows a question to run alone if the question exceeds the batch budget. The request's context length is checked separately and no question is dropped. Usage counts the state for every actual prefix evaluation. Disabling prefix reuse therefore can increase `input_tokens` for the same answers. [Original batching policy](https://github.com/strands-labs/strands-decider/blob/6d85079caf1f1906ebb0f1370b56ef28ba7aa93e/src/strands_decider/infer.py#L347-L435)
