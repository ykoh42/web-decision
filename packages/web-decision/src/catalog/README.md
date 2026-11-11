# Model catalog

`model-spec.json` is the curated source for public model IDs, adapters, descriptions, capabilities, and verification state. Artifact hashes live in the generated `artifacts.lock.json`.

This package candidate exposes `kanana-2-v-decide-3b-2611`. Its entry pins the Kakao checkpoint, native TVM text and vision graphs, graph-required source ranges, model limits, and license. The exact npm archive passed text, image, offline, and maximum-context WebGPU checks on Apple M4 Pro / Chromium 157.

`packages/web-decision/scripts/build-model-catalog.mts` validates each adapter's pinned artifacts and regenerates the lock, runtime catalog, and declaration files. Run it with `--check` to confirm generated files match their sources. Kanana ships native TVM text and vision graphs and their manifests; tokenizer/configuration files and only graph-required Safetensors ranges load from its pinned Hugging Face revision. It supports choice, Noul, and ordered score questions, image choice, a 20,480-token limit per query and candidate prompt, and up to 10 images per request. Images in state messages and the `images` field count together.

When Kev is pinned to its direct source checkpoint, the generator reads the
self-converted package from `assets/kev06-direct`. Use
`--kev-direct-artifact-dir` to select another directory inside this package;
the generator checks its manifest against every local file. At runtime the
catalog's pinned `assets.directArtifactRoot` supplies the whole package,
including its manifest, TVM module, tokenizer, configuration, and weight
shards. The cache uses a local model-specific namespace, so verified files can
be reused offline after an online load.

Artifact integrity, model provenance, license terms, and runtime evidence are
separate records. The artifact lock records pinned hashes; it does not establish
hardware verification or determine how license terms apply to a user's
activities. Users and distributors are responsible for meeting the license
terms described in the package notice and agreement.

To add a model, add its curated ID and TVM WebGPU description, then add an adapter-specific artifact validation recipe and source-integrity/capability assertions. Unknown adapters and non-TVM backends fail closed.
