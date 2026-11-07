# Releasing web-decision

The current preview is `web-decision@0.1.0-preview.1` and supports exactly one route, `kev-0.6b-q4f16`. Keep each Git tag, npm archive, and verification record tied to the same package archive and runtime bundle hashes.

## Before publishing

From a clean checkout, run:

```sh
pnpm install --frozen-lockfile --ignore-scripts
pnpm test:release
pnpm build:release
pnpm pack:release
```

Then verify the resulting package archive in a clean npm consumer and run `npm audit --omit=dev`. Use `npm publish --dry-run --ignore-scripts --tag preview` to inspect the registry payload before any registry write.

Run real-model WebGPU tests locally for the exact archive and every included model. Record browser versions, device, non-fallback adapter, input and output coverage, archive and bundle hashes, failures, and host resource limits in `releases/web-decision-0.1.0-preview.1/verification/`. Standard GitHub-hosted runners do not provide the local GPU evidence required for this release. The current preview's receipts are summarized in `RELEASE_SCOPE.md` and `releases/web-decision-0.1.0-preview.1/verification/real-model-gpu.json`.

Review the exact staged file list, dependencies, notices, asset pins, model-data terms, and security findings before tagging. Do not include model weights unless redistribution rights and required notices are established. The current package excludes all model weights.

Create release tags only from the reviewed source commit and attach the exact tested archive. A GitHub Release and an npm registry publication are separate operations; neither implies the other.
