# web-decision development

- Ship only verified TVM models with curated IDs.
- Prioritize direct TVM import and TVM WebGPU for every model; re-review existing paths and never use ONNX as an intermediate.
- Use pinned upstream TVM unchanged; never patch its source or private APIs. Use supported APIs and keep model-specific lowering in this repository. If these cannot work around a reproduced upstream defect, document it and defer that model path until an upstream TVM release fixes it.
- Pin and verify assets; support offline reuse.
- Preserve model semantics and limits.
- Keep code simple; enforce complexity gates.
- Test the exact package and report hardware limits. Follow the [release procedure](.github/RELEASING.md).
