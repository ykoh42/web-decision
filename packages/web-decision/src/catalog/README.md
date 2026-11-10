# Model catalog

`model-spec.json` is the manually curated source for model IDs, adapters, public
descriptions, capabilities, and verification evidence. Artifact hashes live in
the generated `artifacts.lock.json`, not in the spec.

The curated IDs are `kev-0.6b-q4f16` and `kanana-2-v-decide-3b-2611`.

`packages/web-decision/scripts/build-model-catalog.mts` validates each adapter's
pinned artifacts and regenerates the lock, catalog, and declaration files. Use
`--check` to confirm generated files match their sources. Kev packages its
stable TVM runtime and reconstructs its pinned upstream weights. Kanana ships
native TVM text and vision graphs; it loads pinned tokenizer/configuration files
and only graph-required Safetensors ranges from its pinned Hugging Face
revision. Kanana supports choice, Noul, and ordered score questions, image
choice, a 20,480-token limit for each query and candidate prompt, and up to 10
images per request. Images in state messages and the `images` field count
together.

When Kev is pinned to its direct source checkpoint, the generator reads the
self-converted package from `assets/kev06-direct`. Use
`--kev-direct-artifact-dir` to select another directory inside this package;
the generator checks its manifest against every local file. Configure the
catalog's `assets.directArtifactRoot` only when that package has a real hosted
directory.

Artifact integrity and runtime evidence are separate gates. The lock records
pinned hashes; it does not establish current hardware verification or infer a
release state. Keep evidence and capability status model-specific in the spec.

To add a model, append its curated ID and TVM WebGPU description, then add an
adapter-specific artifact validation recipe and source-integrity/capability
assertions. Unknown adapters and non-TVM backends fail closed.
