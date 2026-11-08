# Releasing web-decision

Release only curated models whose TVM artifacts and behavior have been verified. A release must identify its exact source commit, npm archive, model artifacts, and verification evidence.

## Candidate checks

- Build one candidate archive from a clean checkout and install that archive in a clean consumer.
- Check the public catalog, TypeScript API, package contents, dependency audit, pinned assets, notices, and model-data rights.
- Run the repository's formatting, lint, cognitive complexity, type, unit, and package checks.
- Pass `pnpm run check:typescript:complete` and `pnpm run check:complexity:complete`, plus the Python complexity completion check in CI.
- For every included model and claimed environment, run real inference against the installed archive. Record the browser, hardware, WebGPU adapter, required features, input and output coverage, and reference comparison.
- Verify online loading, artifact hashes, cached offline loading and inference without network access, cleanup, and failure paths. Missing or skipped required evidence fails the release.
- Tie the evidence to the archive and artifact hashes. A change after verification requires a new candidate and new checks.
- Resolve the recorded [Kev TVM validation findings](../docs/kev06-validation-notes.md) before releasing that model.

Review the candidate and evidence before creating a tag or publishing. Publish the exact archive that passed verification.

## Historical releases

To reproduce a previous release, check out its own Git tag and use the instructions and source lock in that checkout. The `0.2.0` tag is missing its TVM source overlay. Recover `releases/web-decision-0.2.0/source/overlay/.toolchains` from commit `baa3c144d23e1882ba29084c9f95897b8c56accb`, then verify it against the tag's source lock. Historical verification is not evidence for a new archive.
