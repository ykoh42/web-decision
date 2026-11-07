# Rebuild `web-decision@0.3.0`

This isolated release snapshot carries forward Kev 0.6B and TAMEV Nano TinyBERT from `0.2.0` and adds exactly one model route: TAMEV Micro MiniLM INT8. It contains package code and notices, not model weights.

## Rebuild the package

From a clean checkout with the pinned Node and pnpm versions:

```sh
pnpm install --frozen-lockfile --ignore-scripts
node releases/web-decision-0.3.0/build.mjs
node releases/web-decision-0.3.0/pack.mjs
```

The build checks the release source lock, the existing Kev asset manifest and the pinned TVM WebGPU source. It produces one browser bundle and records the complete source inputs and bundle hash. The pack step checks the archive file allowlist, route notices, package declarations, `publint`, TypeScript package exports, and the pinned bundle hash.

The npm archive excludes model assets. Kev's existing 14-file asset pack and each TAMEV model's four files must be hosted separately by the application. TAMEV Nano's files total 15,585,045 bytes; TAMEV Micro's files total 24,161,003 bytes. Both are checked against the sizes and hashes in the bundled adapter.

Real-model browser and exact-archive results belong under `verification/`. TAMEV Nano and Micro use ONNX Runtime WebAssembly on CPU, so their GPU verification status is not applicable. The included model routes must be checked against this version's exact bundle before release.
