# Rebuild `web-decision@0.2.0`

This isolated release snapshot carries forward Kev 0.6B from `0.1.0` and adds exactly one model route: TAMEV Nano TinyBERT INT8. It contains package code and notices, not either model's weights.

## Rebuild the package

From a clean checkout with the pinned Node and pnpm versions:

```sh
pnpm install --frozen-lockfile --ignore-scripts
node releases/web-decision-0.2.0/build.mjs
node releases/web-decision-0.2.0/pack.mjs
```

The build checks the release source lock, the existing Kev asset manifest and the pinned TVM WebGPU source. It produces one browser bundle and records the complete source inputs and bundle hash. The pack step checks the archive file allowlist, route notices, package declarations, `publint`, TypeScript package exports, and the pinned bundle hash.

The npm archive excludes model assets. Kev's existing 14-file asset pack and TAMEV Nano's four model files must be hosted separately by the application. The TAMEV files total 15,585,045 bytes and are checked against the sizes and hashes in the bundled adapter.

Real-model browser and exact-archive results belong under `verification/`. TAMEV Nano uses ONNX Runtime WebAssembly on CPU, so its GPU verification status is not applicable. Kev's real-WebGPU evidence for the earlier exact `0.1.0` archive remains in [`../web-decision-0.1.0/verification`](../web-decision-0.1.0/verification); it does not automatically validate the new `0.2.0` bundle.
