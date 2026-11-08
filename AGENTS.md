# web-decision development

- Build local decision inference. Prefer TVM; preserve supported backends.
- Aim for model-ID-only loading. Pin, verify, and cache assets for offline reuse. Allow trusted mirrors.
- Preserve model semantics, pins, and protocol limits. Never silently change backends or data.
- Use supported APIs and pnpm. Keep code simple.
- Test changes and report real-model and hardware limits. Follow the [release procedure](.github/RELEASING.md).
