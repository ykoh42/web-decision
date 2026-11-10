# Kev provenance archive

These immutable source snapshots support the direct-TVM candidate review. Verify them with `sha256sum -c UPSTREAM-SHA256SUMS.txt` from this directory.

- `upstream/` contains the pinned Kev and Qwen model cards, file-list API responses, checkpoint training/provenance records, the exact Kev code license, and the v7 training manifest.
- [`kev-v7-training-data-snapshot.md`](kev-v7-training-data-snapshot.md) summarizes the exact dataset revisions and observed license metadata. It is not a legal conclusion.
- [`../kev06-provenance.md`](../kev06-provenance.md) records the converter inputs, single-fixture CPU evidence, remaining release gates, and current catalog status.
- [`../../tools/kev-converter/reports/direct-tvm-q4-cpu-r6.json`](../../tools/kev-converter/reports/direct-tvm-q4-cpu-r6.json) records the direct TVM Q4 CPU run and source/build hashes.

The model card declares Apache-2.0 for the Kev adapter/head and Qwen base; the pinned Qwen license text is archived here. The Kev model repository has no standalone license or notice file. Training dataset terms remain partly unknown, mixed, missing, or governed by separate agreements. Do not treat this archive as redistribution approval.
