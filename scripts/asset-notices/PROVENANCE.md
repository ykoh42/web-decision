# Kev 0.6B Q4F16 TVM asset provenance

This notice records the pinned inputs and build controls for the separately hosted asset set prepared by `scripts/setup_preview.sh` and `scripts/prepare_web_decision_assets.py` for model ID `kev-0.6b-q4f16`.

## Model inputs

- The ONNX checkpoint is `onnx-community/kev-0.6b-ONNX` at revision `5a827091b3d58fe4fdfc02a5a5b2ce364047d678`.
- The source ONNX graph SHA-256 is `26e059ddd954efd9897772b098db10b9eeda9b873984dc1c73b32a9ce177c7c8`; its external tensor data SHA-256 is `777335dbc052e2e02ec7591f6eefcfb29e2c28caf52075e492d226fc27e0824b`.
- The pinned tokenizer, tokenizer configuration, and model configuration SHA-256 values are `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`, `3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5`, and `40ce1c875acf016ab7742ecdc45d3bad0fe8c51d3ead0ca2263ec5eb4e96f42d`, respectively.
- The accompanying `MODEL-APACHE-2.0.txt` is the Qwen3-0.6B-Base license text at revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, SHA-256 `832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e`. The model notice records the upstream model-card license statements and their limits.

## Build controls

- `scripts/setup_preview.sh` pins TVM source revision `7b5acb32bd271342a24130f4ef439ff6c2e52c6c`, Emscripten SDK revision `d7b4130688bfa348392b921ab04015aa15c4f154`, and Emscripten `4.0.23`.
- The setup specializes the pinned Q4F16 ONNX graph for dynamic sequence length, builds the TVM WebGPU WASM and tensor cache, then invokes the asset packager. The packager requires a successful build report with input shape `[1, "sequence_length"]` and 394 parameters, verifies the pinned source and cache hashes, and checks each copied file against its SHA-256.
- The canonical asset manifest SHA-256 is `8577804e638051cb056af51fee89eb18ea5de85ccec4f196594d830c16fdc767`, taken from `packages/web-decision/src/kev-preview-model.json`.
- The asset manifest lists inference files only. Distribution notices are copied beside them and are not fetched by the browser inference loader.

This record documents source and build pins. It does not determine redistribution rights, establish model quality, or guarantee results across hardware and browsers. Model terms and the intended hosting use require separate review.
