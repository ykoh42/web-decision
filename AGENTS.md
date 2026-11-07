# web-decision development

- Write simple, textbook-quality code with plain names, short functions, and flat control flow.
- Prefer the latest stable TypeScript and JavaScript syntax supported by the declared Node and browser targets.
- Integrate TVM WebGPU and any supported runtime or framework through their supported APIs. Keep fixes focused, and use WebLLM as a reference when it improves decision-model support.
- Use pnpm for the workspace. Keep packages independent and examples self-contained.
- Preserve each model's input and output semantics, including tokenization, option order, scores, and abstention. Add only content required by its protocol.
- Derive accepted inputs from each model's pinned protocol, configuration, tokenizer, and validated runtime. Do not impose adapter-only caps below those limits or silently truncate, clip, reorder, or drop input; report device-resource limits as runtime constraints.
- Run applicable package, type, and browser checks. Distinguish mocks and synthetic inputs from real-model GPU results; report measured scope and limits without silent fallbacks.
- Before publishing a GitHub Release, pass local real-model browser tests for every included model and follow the [release procedure](.github/RELEASING.md). Store results in that version's `verification/` directory.
