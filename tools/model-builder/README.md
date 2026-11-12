# Model building and reference checks

The developer preview build is `scripts/setup_preview.sh`. It downloads the official Kev adapter and Qwen base at the pinned revisions, checks them with the converter's `INPUT_HASHES`, and imports the PyTorch model directly through `torch.export` into TVM Relax. ONNX is used only as a numerical reference; it is not a conversion input.

The converter dependencies are pinned in `tools/kev-converter/uv.lock`. The setup pins the Kev checkpoint at `abe7110dedfd68a185839550079bcf7c26f561ac`, Qwen base at `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, Kev source at `37cb2f6155afec1054de2cc78dfd7c7dd24a956d`, TVM at `7b5acb32bd271342a24130f4ef439ff6c2e52c6c`, and Emscripten SDK at `d7b4130688bfa348392b921ab04015aa15c4f154`. Before the TVM build, it verifies the downloaded file hashes from `INPUT_HASHES` and the Kev source revision; the converter also verifies the Web runtime libraries and `libatomic.so.1` hashes. On Linux, the default library path is `/usr/lib/x86_64-linux-gnu/libatomic.so.1`. Set `WEB_DECISION_LIBATOMIC_PATH` when the matching library is elsewhere.

Run the setup from a fresh checkout with Python 3.12, `uv`, Git, `curl`, `make`, and the matching `libatomic.so.1` installed. The default output is `models/web-decision-assets/direct-kev-0.6b-q4f16/`; an optional first argument selects another path inside the checkout. It contains a local `browser-package/` and `direct-tvm-report.json`. The build does not update the curated catalog or host the external weight shards. The r9 exact-package Chrome 157 WebGPU run passed its 4-case online matrix, offline reuse, and the full 8,192-token boundary on Intel Gen-12LP; see [`direct-tvm-q4-webgpu-real-chrome-r9.json`](../kev-converter/reports/direct-tvm-q4-webgpu-real-chrome-r9.json). Kev remains outside the public catalog pending dataset-rights review and hosting.

The previous ONNX-based preview scripts remain for historical and release-source-lock purposes. `setup_preview.sh` no longer invokes that conversion path.

To reproduce the single-input upstream Kev CPU reference, obtain the exact source model and tokenizer identified in `tests/fixtures/kev06-upstream-onnx-reference.json`, then run:

```sh
cd tools/model-builder
uv sync --locked
uv run --locked python verify_kev_reference.py /path/to/kev-model-directory
```

The directory must contain `tokenizer.json`, `onnx/model_q4f16.onnx`, and `onnx/model_q4f16.onnx_data`. The command verifies their hashes, uses the separately pinned Python tokenizer and ONNX Runtime CPU provider, and checks token IDs and logits against the recorded fixture. The browser test compares the direct TVM result against this reference within the documented tolerances. One example does not establish full numerical equivalence or decision quality.
