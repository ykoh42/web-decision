# web-decision development

- Ship only verified TVM models with curated IDs.
- Prioritize direct TVM import and TVM WebGPU for every model; re-review existing paths and never use ONNX as an intermediate.
- Never carry local source patches to TVM. Use pinned upstream TVM unchanged and keep model-specific lowering in this repository. If supported APIs and model-specific graph rewrites cannot work around a reproduced upstream defect, document it and defer that model path until an upstream TVM release fixes it. Then update the pinned dependency and validate it.
- Pin and verify assets; support offline reuse.
- Preserve model semantics and limits.
- Keep code simple; enforce complexity gates.
- Test the exact package and report hardware limits. Follow the [release procedure](.github/RELEASING.md).
