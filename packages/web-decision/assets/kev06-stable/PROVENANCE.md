# Stable Kev 0.6B TVM bundle provenance

The stable Kev `modelId` uses the TVM WebGPU model WASM and metadata from the current pinned 0.6B Q4F16 asset manifest. The package bundles the model WASM and small runtime metadata in a lazy chunk; it fetches the pinned ONNX external-data file at runtime and reconstructs the nine parameter shards. This public route does not accept `assetsUrl` and always uses the source revision below. The legacy `model` asset-pack route is separate.

- Canonical manifest SHA-256: `57c554e757a96093669c1555e9e2564780c7eaba5e102b4fcd9ad0c535878fe2`.
- Stable model WASM: 6,807,324 bytes, SHA-256 `ecc2bdce793221f4f370890d2ac4ddcb6082b85da05a645f3554903768b03cb0`.
- Source: `onnx-community/kev-0.6b-ONNX`, revision `5a827091b3d58fe4fdfc02a5a5b2ce364047d678`, external-data file `onnx/model_q4f16.onnx_data`, 335,357,952 bytes, SHA-256 `777335dbc052e2e02ec7591f6eefcfb29e2c28caf52075e492d226fc27e0824b`.
- Source-to-shard map SHA-256: `bbe4279b99d995770bd3ba681d17cec07e4f083aeaaddffb0a3ce26d07e81065`.

The loader validates the source hash and map coverage, reconstructs the pinned shard bytes, and verifies every reconstructed shard against the manifest before caching or uploading it to TVM. The raw ONNX external-data file is not cached. The experimental q2-k128 candidate retains its separate WASM, manifest, and route.

This provenance records pins and byte handling. It does not establish decision quality, broad hardware compatibility, or licensing terms for the source data and transformed weights.
