# web-decision development

- Ship only verified TVM models with curated IDs.
- Prioritize direct TVM import and TVM WebGPU for every model; re-review existing paths and never use ONNX as an intermediate.
- Pin and verify assets; support offline reuse.
- Preserve model semantics and limits.
- Keep code simple; enforce complexity gates.
- Test the exact package and report hardware limits. Follow the [release procedure](.github/RELEASING.md).
