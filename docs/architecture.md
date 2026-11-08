# Architecture

This document describes the curated TVM package and its release requirements.

## Public surface

`web-decision` is a browser library. `listModels()` and `describeModel(id)` expose the models included in the installed package. `loadDecisionModel({modelId})` accepts only an included ID. An application cannot choose a model artifact URL, register a model, or replace the execution engine through the public interface.

The package includes only models selected by the project and admitted through the release verification process. A model's metadata describes what it can do and the environments actually checked; it does not guarantee output quality or universal hardware support. GPU support is model-specific. In particular, TVM does not imply that every model requires `shader-f16` or that a browser will offer the necessary GPU features.

## Responsibilities

- The model catalog owns public IDs, capability metadata and pins to model artifacts. Its generated files must match the checked-in source definitions and artifact records.
- An artifact record identifies the model, source revision, transformation inputs, TVM execution contract and verified file hashes. Source claims, build-derived facts and release evidence have distinct owners; do not treat a previous build's test result as evidence for a new package.
- The asset manager downloads from pinned sources, checks size and digest before use and manages cached copies. A verified cache hit is checked before use. Online success is not a guarantee that caching succeeded. An offline request must not fetch missing artifacts.
- The TVM runtime manages device selection, execution, cancellation, device loss and cleanup. It must not silently switch execution engines.
- A model adapter owns model-specific input rendering, tokenization, numerical semantics, output interpretation and limits. Two models may use similar output names with different meanings; an adapter must not invent capabilities that the model does not have.
- Model-building tools compile and package model artifacts. Pin source inputs, toolchains, build options and output hashes; record the relation between the TVM executable, parameter layout and runtime contract.

`modelId` identifies a model and its public behavior. A package release pins the exact artifact it uses. Changing the checkpoint, quantization or output semantics requires a distinct model identity or an explicitly revised public contract. A rebuild of an equivalent artifact requires a new artifact digest and fresh verification.

## Artifact trust

The installed package contains the trusted expected manifest digest. An external server supplies bytes, not authority to change the model: validate the manifest against the package pin, then validate each listed file against that manifest. Hashes detect mismatched bytes; they do not prove model quality, rights, or that a conversion preserved semantics.

Large weight files remain outside the npm package. Their pinned public location must remain available for supported releases, permit browser CORS, and be covered by applicable license and redistribution review. Download and memory limits, browser quota and eviction, interruption, corrupt cached files and concurrent requests must be tested. The application is responsible for making its own pages and code available offline.

## Release verification

Build a final candidate archive once. Install that exact archive in a clean consumer and verify its package and asset digests. Run the required tests, including actual inference for every included model on each claimed browser/device configuration; missing or skipped required evidence is not a pass. Verify model-specific reference behavior, pinned artifacts, online load and inference, a cached offline load and inference without network requests, cleanup and failure paths. Keep the external verification record tied to the candidate archive and artifact hashes. If code, catalog, assets or the archive change after verification, create and verify a new candidate. Publish the exact archive that passed.

Existing release snapshots and their original pins remain historical records; they are not evidence that a new package has passed. The current source tree is responsible for future packages. Prior GPU measurements must name the exact tested package, browser and device rather than imply broad support.

Reproduce a historical release from its own Git tag and verify its source lock. The `0.2.0` tag omits its TVM source overlay; recover that overlay from commit `baa3c144d23e1882ba29084c9f95897b8c56accb` and check every file against the tag's lock before use. The release scripts in a later checkout are not a substitute for the tagged source.

## Quality gates

New JavaScript and TypeScript use the repository's SonarJS cognitive complexity check, and Python uses complexipy. The limit is 15. Existing baseline findings remain visible and cannot worsen; new findings are not added as an automatic escape hatch. Formatting, lint, type, unit, package, model and release checks apply alongside these limits.

The TypeScript migration policy tracks remaining directly maintained JavaScript by exact path. CI requires the complete migration check; immutable historical releases and documented external or generated outputs have separate exact-path exceptions.
