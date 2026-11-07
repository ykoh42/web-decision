# Releasing web-decision

The current stable release is `web-decision@0.1.0` and supports exactly one route, `kev-0.6b-q4f16`. Keep each Git tag, npm archive, and verification record tied to the same package archive and runtime bundle hashes.

## Before publishing

From a clean checkout, run:

```sh
pnpm install --frozen-lockfile --ignore-scripts
pnpm test:release
pnpm build:release
pnpm pack:release
```

Then verify the resulting package archive in a clean npm consumer and run `npm audit --omit=dev`. Use `npm publish --dry-run --ignore-scripts --access public` to inspect the registry payload before publishing the stable version.

Run real-model WebGPU tests locally for the exact archive and every included model. Record browser versions, device, non-fallback adapter, input and output coverage, archive and bundle hashes, failures, and host resource limits in `releases/web-decision-0.1.0/verification/`. Standard GitHub-hosted runners do not provide the local GPU evidence required for this release. Stable receipts are summarized in `RELEASE_SCOPE.md` and `releases/web-decision-0.1.0/verification/real-model-gpu.json`.

Review the exact staged file list, dependencies, notices, asset pins, model-data terms, and security findings before tagging. Do not include model weights unless redistribution rights and required notices are established. The current package excludes all model weights.

Create the `v0.1.0` tag only from the reviewed source commit and attach the exact tested archive. Publish the same archive to npm with the `latest` tag. A GitHub Release and an npm registry publication are separate operations; neither implies the other.
