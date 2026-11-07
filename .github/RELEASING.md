# Releasing web-decision

Keep each Git tag, npm archive, and verification record tied to the same package archive and runtime bundle hashes.

## Release cadence

After `0.1.0`, release each newly completed model in its own version. Carry forward the previous stable model routes and add exactly one new route per release; do not wait to bundle several completed candidates together. A route is eligible only after its exact package archive passes the required package, type, and real-model browser checks, and its model and asset rights have been reviewed. If a candidate is blocked, keep its route out of the release and continue with the next candidate only as a separate release once that candidate is complete.

## Before publishing

From a clean checkout, run the scripts for the candidate version. For example, for `0.3.0`:

```sh
pnpm install --frozen-lockfile --ignore-scripts
pnpm test:release:0.3.0
pnpm build:release:0.3.0
pnpm pack:release:0.3.0
```

Then verify the resulting package archive in a clean npm consumer and run `npm audit --omit=dev`. Use `npm publish --dry-run --ignore-scripts --access public` to inspect the registry payload before publishing the stable version.

Run real-model browser tests locally against the exact archive for every included model, using each model's supported runtime. For WebGPU routes, record browser versions, device, non-fallback adapter and required features. For CPU/WASM routes, record browser versions and execution provider; GPU verification is not applicable. Record input and output coverage, archive and bundle hashes, failures, and host resource limits in that version's `releases/web-decision-<version>/verification/` directory. Standard GitHub-hosted runners do not provide the local GPU evidence required for WebGPU routes.

Review the exact staged file list, dependencies, notices, asset pins, model-data terms, and security findings before tagging. Do not include model weights unless redistribution rights and required notices are established. The current package excludes all model weights.

Create a `vX.Y.Z` tag from the reviewed source commit and attach the exact tested archive. Publish the same archive to npm with the `latest` tag. Verify both publications and record their hashes in that version's verification directory.
