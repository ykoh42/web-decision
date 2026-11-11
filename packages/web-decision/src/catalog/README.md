# Model catalog

`model-spec.json` is the full candidate source for model IDs, adapters,
descriptions, capabilities, and verification evidence. Artifact hashes live in
the generated `artifacts.lock.json`, not in the spec. `model-provenance.json`
tracks the approval state for every candidate.

The public catalog includes models with verified provenance.
Kanana's current `verified` record keeps its existing public entry; pending
candidates are excluded. Kev's direct TVM conversion stays pending until its
source, conversion, rights, and exact-package gates pass.

`packages/web-decision/scripts/build-model-catalog.mts` validates each adapter's
pinned artifacts and provenance, then regenerates the candidate lock/catalog,
public catalog, and declaration files. Use `--check` to confirm generated
files match their sources. Kanana ships
native TVM text and vision graphs; it loads pinned tokenizer/configuration files
and only graph-required Safetensors ranges from its pinned Hugging Face
revision. Kanana supports choice, Noul, and ordered score questions, image
choice, a 20,480-token limit for each query and candidate prompt, and up to 10
images per request. Images in state messages and the `images` field count
together.

When Kev is pinned to its direct source checkpoint, the generator reads the
self-converted package from `assets/kev06-direct`. Use
`--kev-direct-artifact-dir` to select another directory inside this package;
the generator checks its manifest against every local file. At runtime the
catalog's pinned `assets.directArtifactRoot` supplies the whole package,
including its manifest, TVM module, tokenizer, configuration, and weight
shards. The cache uses a local model-specific namespace, so verified files can
be reused offline after an online load.

Artifact integrity and runtime evidence are separate gates. The lock records
pinned hashes; it does not establish current hardware verification or infer a
release state. Keep evidence and capability status model-specific in the spec.

To add a model, append its curated ID and TVM WebGPU description, then add an
adapter-specific artifact validation recipe and source-integrity/capability
assertions. Unknown adapters and non-TVM backends fail closed.
