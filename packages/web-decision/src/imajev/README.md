# Imajev internal integration candidate

The private preview registers `experimental-imajev-4b-candidate` as an
online-only text-choice route using separately supplied pinned assets. An
installed preview.33 package ran three real-model cases on one Apple GPU; see
[the evidence](../../../../experiments/imajev-range-hashes-20261101/README.md).
The graph's forced index narrowing, broader model quality and device
compatibility remain unverified. Session unit tests use fake devices, scores
and weights.
The session reuses the Strands pinned-TVM readback lease, including quarantine
when a drain cannot be confirmed. Its owner must install the Strands disposal
guard after runtime tracking and before WebGPU initialization; checking the
guard's shape at construction does not prove that installation happened. A
pending readback can delay cancellation cleanup indefinitely, and uncertain
resources are intentionally retained until the JavaScript realm is discarded.
The session also retains itself and asks the guard to retain the runtime if
root scope closure or runtime disposal fails, or device loss is observed
during disposal. The failed disposal rejects and is not retried. This is a
conservative fake-tested policy, not proof of safety on an actual device.

The text-only, single-rotation protocol and result processor are adapted from
the isolated `experiments/imajev-integration-20261030` work. They derive from
Imajev reference source at commit `318e8dad55c12a1c6faddbec5f30489d2ee529ff`,
copyright 2026 Mohit Garg, Apache-2.0. The protocol rejects invalid UTF-16 and
reserved control tokens and is narrower than the upstream implementation.

The unknown readout is at the index immediately after the supplied options.
If it wins, the internal result sets `selected` and `selectedIndex` to `null`,
and records the highest-probability known choice separately as `bestKnown`.
Known-choice probabilities are conditional on excluding unknown. Any future
public API must retain this distinction without changing legacy model results.
They are normalized directly from the known logits, so an overwhelming unknown
score does not turn underflowed known probabilities into a uniform tie. This
intentionally differs from the upstream uniform fallback in that edge case.

Before any public release, broaden the asset, graph, device and model-quality
checks and review model and asset distribution terms. See the experiment
README and its linked evidence for the current limitations.

The current preview.33 installed bundle was also checked on one Apple M4 Pro
using a fixed answerable request and a fixed abstention request. Both matched
the previous GPU scores exactly; this two-request check does not clear the
forced graph index narrowing or establish representative quality. See the
[current support record](../../../../docs/model-support/IMAJEV_SUPPORT.md).

The isolated disposal regression checks are included in the package's default
test glob. To run them separately:

```sh
node --test test/imajev-disposal.test.mjs
```

Run this command from `packages/web-decision`; it uses synthetic CPU-backed
fixtures and does not execute a model or WebGPU.
