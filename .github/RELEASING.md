# Releasing web-decision

Release only curated models whose TVM artifacts and behavior have been verified. A release must identify its exact source commit, npm archive, model artifacts, and verification evidence.

## Candidate checks

- Build one candidate archive from a clean checkout and install that archive in a clean consumer.
- Check the public catalog, TypeScript API, package contents, dependency audit, pinned assets, notices, and model-data rights.
- Pass `pnpm run check:model-provenance`; every public model must have a pinned, committed provenance record with resolved source lineage and redistribution evidence. Main CI accepts pending models excluded from the public catalog, and release verification rejects pending records.
- Run the repository's formatting, lint, cognitive complexity, type, unit, and package checks.
- Pass `pnpm run check:typescript:complete` and `pnpm run check:complexity:complete`, plus the Python complexity completion check in CI.
- For every included model and claimed environment, run real inference against the installed archive. Record the browser, hardware, WebGPU adapter, required features, input and output coverage, and reference comparison.
- Verify online loading, artifact hashes, cached offline loading and inference without network access, cleanup, and failure paths. Missing or skipped required evidence fails the release.
- Supply independently rebuilt TVM WASM files and successful build reports to the [release evidence gate](../scripts/release/RELEASE-EVIDENCE.md); their bytes must match every curated manifest.
- Tie the evidence to the archive and artifact hashes. A change after verification requires a new candidate and new checks.
- Resolve the recorded [Kev TVM validation findings](../docs/kev06-validation-notes.md) before releasing that model.

Review the candidate and evidence before creating a tag or publishing. Publish the exact archive that passed verification.

## 0.8.0 archive identity

The GPU-tested package was built from `4bf2d93e9234bebb8007e6256e0bfa1d822bc0d5`. Before publishing, verify the original archive itself:

```sh
archive=/path/to/web-decision-0.8.0.tgz
printf '%s  %s\n' '69574de135185770ba825f58b4ce294b0d2e0189a72a22e4dd8ba7e030b6a923' "$archive" | shasum -a 256 -c
```

CI builds and tests the source and its own packed archive; it does not receive or verify the original archive. Review the original archive's model evidence and its source identity separately.

## Historical releases

For 0.1.0–0.3.0, check out the corresponding release tag and follow its build instructions and source lock. The 0.2.0 tag lacks its TVM overlay; restore it before checking the lock:

```sh
git restore --source=baa3c144d23e1882ba29084c9f95897b8c56accb -- releases/web-decision-0.2.0/source/overlay/.toolchains
```

The tags and overlay commit must be available locally; a shallow clone may need its history fetched. The tagged source locks have been checked against their inputs, including the restored 0.2.0 overlay. This does not by itself establish a reproducible archive build. The later [b0f66fb snapshot](https://github.com/ykoh42/web-decision/tree/b0f66fb34b96887a75f7708b3944ff1b62b7560a/releases) preserves additional historical sources and evidence, but its checkout alone does not satisfy those release source locks. Historical verification is not evidence for a new archive.
