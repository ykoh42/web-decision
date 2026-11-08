# H2O experimental preview adapter

This experimental route connects the pinned H2O plan, tokenizer, range source,
incremental hashes, TVM owner/upload/runner and choice session through
`loadDecisionModel`. It uses the path-free pinned projection and inventory and
supports online HTTP(S) range loading only, without cache, offline loading,
progress callbacks or prefetch. Asset servers
must support exact byte ranges and expose `Content-Range` across origins.
The public browser API checks adapter requirements but cannot measure available
system RAM or reproduce the development host's external resource guard.

The proposed separately hosted asset directory has the following relative paths:

| Path                                                                                     | Pinned source                                                                           |
| ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `evidence/inventory.json`                                                                | `h2o-inventory-31d7b5f89e2c41a6/inventory.pending.json`                                 |
| `h2o-public-metadata.json`                                                               | `experiments/h2o-public-evidence-preparation-20261031/h2o-public-metadata.pending.json` |
| `manifest.pending.json`, `tensor-*.pending`, `head-f32.pending`                          | `h2o-run-c476df0e9ff43007/candidate/`                                                   |
| `model.pending.wasm`                                                                     | `full-split-webgpu-20261030-v2/model.pending.wasm`                                      |
| `tokenizer/tokenizer.json`, `tokenizer/tokenizer_config.json`, `tokenizer_contract.json` | corresponding paths under `checkpoint/`                                                 |

Unless otherwise shown, paths above are under `models/h2o-integration-20261030/`.
The 681 inventory files total 2,382,544,174 bytes, including 2,369,736,192
bytes of converted weights and the tokenizer; the WASM, config, contract and
metadata are additional. Inventory ranges and whole-file hashes
must remain byte-for-byte unchanged. `source-evidence.mts` retrieves the pinned
inventory, and `source.mjs` verifies the path-free projection and manifest
before opening the assets. Every requested range
requires a correct HTTP 206 and `Content-Range`; cross-origin deployments must
allow the request and expose that header. No cache or offline mode is supported.

The original stage log and monitor contain local absolute paths and are not in
this proposed directory. The projection was derived only after checking the
original pinned stage, manifest, inventory and receipt chain. It retains source
hashes and the required ordered inputs, but it does not independently prove
the hidden records' contents, and neither the projection nor the old receipts
are signed. Its SHA pin trusts the preparation and package release. The model
and converted artifacts still need a distribution and license review. The
proposed directory is a technical contract, not a public deployment.

`test/h2o-packaged-source.test.mjs` packs the actual npm artifact, imports its
internal modules from an extracted installed layout, and uses a local fake fetch
backed by the real pinned inventory and small files plus one 64-byte weight.
It compares both metadata paths' validated plans, checks tampering, and exercises
the actual tokenizer, loader orchestration and session. Device, runtime, upload
and inference are simulated. It does not prove the whole model downloads,
executes, or fits a browser device.
The session has no fixed request-wide question-count cap. Each question is
separately packed and executed in its own full graph call, sequentially. The
per-question 512-token and 255-logit limits still apply. Preparation and result
memory grow with the number of questions; this is not a memory-fit guarantee
or a bound on the number of queued requests.
The loader accepts `experimental-h2o-lightning-4b-q4f16` and the separate
`-dlight` and `-dual` graph variants. Each variant requires its own matching
pinned graph and, for the optimized variants, candidate metadata.
The session has the common `decide`, `releaseCachedMemory` and `dispose` shape,
but this route rejects caching, offline mode, progress callbacks and nonzero
prefetch. Its adapter checks cannot measure available host memory.
During loading, cancellation stops waiting for asynchronous metadata, graph and
tokenizer verification; an already-started browser hash may finish afterward.
Synchronous JSON parsing and tokenizer construction cannot be interrupted until
they return control to the event loop.
`createH2oBrowserFactories` accepts the
entry's existing Tokenizer and TVM imports, avoiding checkout-relative imports
when supplied internally.

The ESM hash module contains fast-sha256 1.3.0 with its UMD wrapper replaced by
an ESM wrapper. The upstream file SHA-256 is
`08abbd6528bcc26760f49399b9064b87199b9245ad22eea283f30fcffd79c8b3`;
its license is in `H2O-SHA256-LICENSE` at the package root. This does not
resolve the distribution terms for the model and generated assets.

The private preview route is not a production release. Numerical acceptance
criteria, representative inputs and lengths, model quality, more devices and
real-device abort, device-loss, readback and disposal remain to be evaluated.
The preliminary GPU limits are not a memory-fit guarantee. Asset distribution
and license obligations need review before publishing the converted files.
