# Kev 0.6B provenance review

Reviewed 2026-11-09. The catalog pins the following public upstream asset:

- Repository: [onnx-community/kev-0.6b-ONNX](https://huggingface.co/onnx-community/kev-0.6b-ONNX/tree/5a827091b3d58fe4fdfc02a5a5b2ce364047d678), revision `5a827091b3d58fe4fdfc02a5a5b2ce364047d678`.
- File: `onnx/model_q4f16.onnx_data`, 335,357,952 bytes, SHA-256 `777335dbc052e2e02ec7591f6eefcfb29e2c28caf52075e492d226fc27e0824b`.
- The [pinned upstream model card](https://huggingface.co/onnx-community/kev-0.6b-ONNX/blob/5a827091b3d58fe4fdfc02a5a5b2ce364047d678/README.md) declares Apache-2.0 and identifies its ONNX files as conversions of `jaredpalmer/kev-0.6b`. The card's SHA-256 is `c1e475f5e8642ed37005c58b15b09f7e5a92d20a91ab37e96e4e683126c7d214`.
- A [published Kev model card](https://huggingface.co/jaredpalmer/kev-0.6b/blob/ac5bda89ec391ee5fe0c17dd44cae01815dbffcd/README.md) declares Apache-2.0 and identifies `Qwen/Qwen3-0.6B-Base` as its base model. Its SHA-256 is `8d5642bb152900d912d4e822e7fdf910a8879f42383505b1458fc01fd96aa0c2`.
- The [Qwen base model license](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/da87bfb608c14b7cf20ba1ce41287e8de496c0cd/LICENSE) contains Apache-2.0; that file's SHA-256 is `832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e`.

The pinned `onnx/model_q4f16.onnx` graph is 3,338,522 bytes and has SHA-256 `26e059ddd954efd9897772b098db10b9eeda9b873984dc1c73b32a9ce177c7c8`. Its metadata reports `producer_name=pytorch` and `producer_version=2.13.0`; model and graph metadata properties are empty. These values are claims embedded in the graph, not a verified conversion log or a complete toolchain record.

The ONNX card does not pin the original Kev adapter revision or Qwen base revision used for conversion. The cited Kev and Qwen revisions document their publishers' license declarations; they do not establish the exact conversion inputs. The catalog therefore retains `separate-review-required`. Before release, verify the conversion lineage and all obligations for the embedded assets and redistributed artifacts. The release evidence must also match the exact final package and model artifacts.

This unresolved state is recorded in `packages/web-decision/src/catalog/model-provenance.json` and the hash-pinned evidence file beside it. `pnpm run check:model-provenance` fails while any public model is pending, and the release verifier independently requires the committed manifest and evidence bytes. The pending Kev record is intentional; it does not approve or infer missing lineage or redistribution rights.
