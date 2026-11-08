# web-decision development

- Write simple, textbook-quality code with plain names, short functions, and flat control flow.
- Prefer the latest stable TypeScript and JavaScript syntax supported by the declared Node and browser targets.
- Build browser-local decision models with TVM WebGPU as the primary direction while preserving supported ONNX paths; do not force a TVM-only migration.
- Target model-ID-only loading through pinned sources, automatic download and integrity checks, caching, and offline use after warmup. Keep `assetsUrl` optional for trusted mirrors, and make each route's backend and model behavior explicit. Never silently substitute a backend, model, or behavior. This is a direction, not a claim that every route already supports ID-only loading.
- TAMEV's TVM transition is separately owned and is not complete. Keep real-model browser verification for backend changes, and report its measured scope clearly.
- Integrate TVM WebGPU and any supported runtime or framework through their supported APIs. Keep fixes focused, and use WebLLM as a reference when it improves decision-model support.
- Use pnpm for the workspace. Keep packages independent and examples self-contained.
- Preserve each model's input and output semantics, including tokenization, option order, scores, and abstention. Add only content required by its protocol.
- Derive accepted inputs from each model's pinned protocol, configuration, tokenizer, and validated runtime. Do not impose adapter-only caps below those limits or silently truncate, clip, reorder, or drop input; report device-resource limits as runtime constraints.
- Run applicable package, type, and browser checks. Distinguish mocks and synthetic inputs from real-model GPU results; report measured scope and limits without silent fallbacks.
- Before publishing a GitHub Release, pass local real-model browser tests for every included model and follow the [release procedure](.github/RELEASING.md). Store results in that version's `verification/` directory.
