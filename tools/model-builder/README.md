# Model building and reference checks

The current Kev build recipe remains in `scripts/setup_preview.sh` and its pinned inputs. It is not yet a generic model converter. Keep its outputs separate from tracked source and do not treat a successful build as release verification.

To reproduce the single-input upstream Kev CPU reference, obtain the exact source model and tokenizer identified in `tests/fixtures/kev06-upstream-onnx-reference.json`, then run:

```sh
cd tools/model-builder
uv sync --locked
uv run --locked python verify_kev_reference.py /path/to/kev-model-directory
```

The directory must contain `tokenizer.json`, `onnx/model_q4f16.onnx`, and `onnx/model_q4f16.onnx_data`. The command verifies their hashes, uses the separately pinned Python tokenizer and ONNX Runtime CPU provider, and checks token IDs and logits against the recorded fixture. The browser test compares its TVM result against this reference within the documented tolerances. One example does not establish full numerical equivalence or decision quality.
