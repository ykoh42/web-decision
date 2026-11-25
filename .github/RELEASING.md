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

## E2B v1 promotion to 0.8.0

The package source is commit `0ddee0f7560015c1d19dc462c9ebfa774afb98df`.
Keep the tested checkout at that commit throughout the GPU runs.
Run the model-specific gates with the original GPU reports, start and completion records,
pinned reference data, independent TVM outputs, and the exact installed archive.
The five models must pass their own gates. Use the committed Kanana evidence-child
mode for its maximum-context and browser reports.

Preserve the original reports and local verifier output outside the npm package.
Review the complete local verifier output and its source, archive, installed-tree,
bundle and model evidence. Commit only the reviewed portable receipt at
`scripts/release/evidence/gemma4-v1-promotion-0.8.0.json`. The receipt contains
each required report digest, each independently verified WASM and build report
digest, the Kanana evidence commit and the complete local verification digest.
The receipt belongs in a descendant commit containing only the reviewed release
procedure changes. Its source field still names the GPU-tested package commit.
The archive-specific Kanana evidence child must be integrated into that history.
The hosted check verifies its direct parent, single added evidence path, source
provenance, archive and report digests, and identical blob in the tagged tree.

GitHub Release checks read the committed receipt and validate its required evidence
fields and the exact SHA-256 of the packed archive. This hosted CPU check relies on
the local gate and the release review for the original GPU observations.
The digest fields identify the locally verified outputs retained by the release
owner. The workflow has no access to the local originals or a GPU.
A receipt alone gives no independent assurance about how an observation was obtained.

Run the same receipt and archive check before publishing the tested archive.

```sh
node --import tsx scripts/release/verify-gemma4-v1-promotion.mts /path/to/web-decision-0.8.0.tgz
```

Require the tagged GitHub Release checks to pass before publishing that exact
archive. If a CI rebuild produces different bytes, resolve the discrepancy and
verify the candidate again before publication.

## Historical releases

For 0.1.0–0.3.0, check out the corresponding release tag and follow its build instructions and source lock. The 0.2.0 tag lacks its TVM overlay; restore it before checking the lock:

```sh
git restore --source=baa3c144d23e1882ba29084c9f95897b8c56accb -- releases/web-decision-0.2.0/source/overlay/.toolchains
```

The tags and overlay commit must be available locally; a shallow clone may need its history fetched. The tagged source locks have been checked against their inputs, including the restored 0.2.0 overlay. This does not by itself establish a reproducible archive build. The later [b0f66fb snapshot](https://github.com/ykoh42/web-decision/tree/b0f66fb34b96887a75f7708b3944ff1b62b7560a/releases) preserves additional historical sources and evidence, but its checkout alone does not satisfy those release source locks. Historical verification is not evidence for a new archive.
