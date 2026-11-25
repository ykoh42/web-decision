# Release browser evidence gate

The gate connects a candidate npm archive, the package installed by the browser test, curated catalog inputs, static asset manifests, independently built TVM artifacts, and one real Chromium WebGPU test per curated model. It accepts only the dedicated test file pinned by [release-evidence-policy.json](./release-evidence-policy.json). The policy bytes are pinned in the verifier and must also match the committed `HEAD` copy.

Before collecting evidence, run `pnpm run check:model-provenance`. The verifier independently applies the same fail-closed check to the committed model spec, artifact lock, provenance manifest, and evidence files. The manifest has a source-code-pinned SHA-256; changing only a model status cannot approve it. Every entry must match the complete public model ID set and exact artifact-lock bytes. The verifier accepts only the canonical `packages/web-decision` source root, so the explicitly synthetic release-test fixture cannot be used for a release. A `pending` record produces an error that names the unresolved lineage or rights.

## Mapika exact-package GPU evidence

From a clean checkout, build and pack once. Install that tarball in a clean consumer and record its absolute paths using `WEB_DECISION_RELEASE_ARCHIVE`, `WEB_DECISION_RELEASE_PACKAGE_DIR`, and `WEB_DECISION_INSTALLED_BUNDLE`. Set `WEB_DECISION_MAPIKA2B_PUBLIC_REPORT` to a JSON path outside the checkout. Leave `WEB_DECISION_MAPIKA2B_SNAPSHOT` unset for release verification.

```sh
WEB_DECISION_REAL_GPU=1 WEB_DECISION_BROWSER_CHANNEL=chrome \
  WEB_DECISION_MAPIKA2B_PUBLIC_GPU=1 WEB_DECISION_MAPIKA2B_RELEASE_GPU=1 \
  pnpm exec playwright test tests/browser/mapika2b-public-real-webgpu.spec.ts \
  --grep 'loads Mapika by model ID'

# With the pinned native TVM Python compiler and Emscripten installed:
python scripts/mapika/build_release_tvm.py --output-dir /release/mapika-tvm

node --import tsx scripts/release/mapika-release-evidence.mts \
  "$WEB_DECISION_RELEASE_ARCHIVE" "$WEB_DECISION_RELEASE_PACKAGE_DIR" \
  "$WEB_DECISION_MAPIKA2B_PUBLIC_REPORT" /release/mapika-tvm/build-evidence.json
```

This gate checks the committed catalog and exact archive, both independent WASM rebuilds, seven text and image reference cases, and cached offline parity. It records that the first text result precedes any vision weight requests and that `prepareOffline()` completes the full cache separately. The browser reads from the pinned Hugging Face checkpoint and writes about 4.5 GB to the test profile by the end of preparation. The Mapika gate covers only Mapika. Collect same-archive GPU evidence for the other included models before publishing.

The expected model IDs and artifact digests come from the committed `packages/web-decision/src/catalog/model-spec.json` and `artifacts.lock.json`. The gate checks that the model spec, artifact lock, and pinned provenance manifest have the same non-empty ID set, that each manifest digest and static manifest path match the lock, and that the built and installed public APIs (`listModels()` and `describeModel(id)`) match those committed inputs. It does not read the legacy model registry. The source API is built from `src/curated/index.ts`; the verifier imports its built `dist/index.js` so the CLI also runs on Node 20, which cannot execute TypeScript source directly.

The provenance gate checks committed evidence hashes and binds any declared upstream source to the artifact lock. It does not independently authenticate the reviewer, verify remote source claims, or prove that the reviewed documents are true; those remain human review responsibilities.

## Gemma 4 v2 exact-package evidence preparation

Gemma source provenance records the pinned checkpoint lineage, redistribution terms, and independently rebuilt graph. The verified provenance status refers to those inputs. Browser verification uses the combined archive built from that committed candidate and records results outside the source tree. This permits the same package bytes to proceed from candidate verification to release.

The public producer requires a clean source checkout and an installed combined archive with verified provenance. It records the archive, source, graph response, physical hardware, live checkpoint range, six CPU-reference cases, image recovery, and offline inference. The official subset includes rows 0–299 and 430–489 and excludes rows 300–429. The producer embeds both the record and the code probabilities in the Playwright JSON report.

With the release archive installed in a clean consumer and the exact upstream `golden_vectors.jsonl` available, run the producer with the paths set below. Save the JSON report outside the checkout.

```bash
WEB_DECISION_REAL_GPU=1 \
WEB_DECISION_BROWSER_NAME=chromium \
WEB_DECISION_GEMMA_V2_PUBLIC_EVIDENCE=1 \
WEB_DECISION_GEMMA_V2_RAW_CAPTURE=1 \
WEB_DECISION_RELEASE_ARCHIVE="$archive" \
WEB_DECISION_RELEASE_ARCHIVE_SHA256="$archive_sha256" \
WEB_DECISION_INSTALLED_BUNDLE="$installed_package/dist/index.js" \
WEB_DECISION_GEMMA_V2_OFFICIAL_GOLDEN="$golden" \
PLAYWRIGHT_JSON_OUTPUT_NAME="$playwright_json" \
pnpm exec playwright test --config=playwright.config.ts --reporter=json \
  tests/browser/gemma-v2-public-real-webgpu.spec.ts

node --import tsx scripts/release/strands-gemma4-v2-release-evidence.mts \
  "$archive" "$installed_package" "$playwright_json" "$tvm_build_evidence" "$golden"
```

Chromium’s debugger observes unrounded values in the installed scorer. The producer serves archive JavaScript and the pinned tokenizer without development-server transforms. It fails if Chromium parses different bytes, moves the exact scorer breakpoint, or records a different set of questions. The Gemma verifier requires raw evidence for the 1,066 pinned questions and recomputes the upstream criterion. It requires at least 1,000 questions, at least 98% top-1 agreement, and at most 0.02 mean per-question maximum absolute probability difference. Ties use the first code. It checks the exact requests, token counts, output types, choice labels, and structure for all six CPU cases, and records their numerical differences. The Gemma result covers this model on the exact combined archive. The remaining models have separate checks.

## Gemma split evidence for the reviewed 0.7.0 archive

The original single-producer procedure above remains supported. The split gate uses two Playwright reports for the same tested source and installed archive. The 1,066-question corpus and five CPU-reference cases use byte-preserving archive JavaScript and the pinned tokenizer, with raw scorer capture, image recovery, offline inference, and cleanup. The separate 32,768-token case exercises the installed archive through a Vite consumer. That run did not record browser HTTP response bytes.

The reviewed CPU Vite replay checks the exact archive and source, 27 archive JavaScript modules plus the tokenizer, and 17 assets. It is a later reproduction of the Vite setup. The replay does not attest to the live response bytes or its complete inherited environment. Its report and script pins apply to the reviewed 0.7.0 candidate. A different candidate requires a newly reviewed replay and pins.

The persistent Chromium producer uses one of these explicit checkpoint evidence scopes.

| Producer setting                                                     | Observation                                                                                                                                                                                                                  |
| -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `WEB_DECISION_SPLIT_PERSISTENT_CHECKPOINT_MODE=online`               | With a fresh profile, the verifier checks the original public checkpoint range observed during model loading.                                                                                                                |
| `WEB_DECISION_SPLIT_PERSISTENT_CHECKPOINT_MODE=local`                | The producer verifies the local checkpoint and each range it serves. This mode records the local route and range digests.                                                                                                    |
| `online` with `WEB_DECISION_SPLIT_PERSISTENT_REUSE_FAILED_PROFILE=1` | The reviewed rerun uses a clone of the pinned failed-run profile at the original loopback origin. Before model load it separately checks the first pinned public Hugging Face 206 response and the installed graph response. |

The reused-profile evidence records `independent-online-range-probe` and `independent-archive-response`. It verifies the exact requested range, bytes and digest and binds the old start receipt digest. Model loading can use verified cached assets. A probe records one observed public range. None of the persistent modes verifies a whole-checkpoint online download.

Supply `WEB_DECISION_SPLIT_TESTED_COMMIT`, `WEB_DECISION_SPLIT_TESTED_ROOT`, `WEB_DECISION_SPLIT_PRODUCER_DIR`, `WEB_DECISION_SPLIT_VITE_AUDIT`, and `WEB_DECISION_SPLIT_VITE_AUDIT_SCRIPT`. For persistent evidence, also supply the settings recorded by the producer. Run the normal Gemma verifier command above with the completed raw corpus report as `"$playwright_json"` and append `--split "$independent_32k_prefix"`.

The verifier checks the original reports and their start and completion receipts independently. It records the tested source commit separately from the verifier commit and rejects intervening changes outside release verification. Raw probability, CPU references, matching devices, offline replay, cleanup, and native TVM build checks remain required. The output is model evidence with `releaseApproved: false`.

## Gemma split evidence for e729bb94

The e729 verifier binds the recorded tested source, exact npm archive, installed package tree, and bundle to the new candidate. It reads the original reports and receipts for that candidate. The 32K receipts preserve their installed-package Vite consumer scope and the completion receipt binds the start and Playwright report digests.

The Vite replay must use the new installation and actual e729 start receipt. Its script and report require an archive-specific reviewed pin. The previous 7c3 replay continues to identify the older archive. The later Vite replay remains CPU evidence and the live browser HTTP response bytes remain outside its scope.

The fresh persistent Chromium producer captures the byte-preserving raw 1,066-question corpus and five CPU-reference cases, then checks offline replay. Its model-load range is checked immediately against the pinned inventory. The previous candidate's warm profile receipt does not establish e729 provenance. Admission of a future candidate-specific profile clone requires verified seed receipts, the same storage origin, e729's complete offline asset rehash, and a separately reported public range probe. The source profile remains read-only.

The split procedure uses the model-specific Gemma verifier. The historical generic command still expects the single Kev Playwright policy and cannot accept the four model-specific current reports. The current native TVM checks pin the toolchain separately from the browser transport or scorer outcome.

## Post-build Kanana evidence for the current archive

Build and test the archive on the clean source commit. Use `--kanana-current-candidate` with the archive pin to check both Kanana reports and the independent TVM rebuilds. Keep the test checkout on that source commit so the Gemma and Mapika report checks continue to bind the same source. The candidate verifier prints the archive, installed package tree, bundle and report digests, as well as the verified context length.

In a separate worktree at the tested commit, add `scripts/release/evidence/kanana-current/<archiveSha256>.json` and commit it as a direct child of that commit. The child commit may add exactly this one regular file. Record the candidate verification values and the SHA-256 of each original Kanana provenance file from the tested source. The two provenance files are `packages/web-decision/src/catalog/provenance/kanana-2-v-decide-3b-2611.json` and `packages/web-decision/src/catalog/provenance/kanana-conversion-config.json`.

```json
{
  "schemaVersion": 1,
  "sourceCommit": "<tested git commit>",
  "packageVersion": "<verified archive version>",
  "archiveSha256": "<archive sha256>",
  "packageTreeSha256": "<installed package tree sha256>",
  "bundleSha256": "<installed entrypoint sha256>",
  "playwrightReportSha256": "<browser report sha256>",
  "maxContextReportSha256": "<context report sha256>",
  "maxContextTokens": 20480,
  "maxContextPassed": true,
  "modelProvenanceSha256": "<tested model provenance file sha256>",
  "conversionConfigSha256": "<tested conversion config sha256>"
}
```

Run the final gate from the original clean source worktree. Provide the full ID of the post-build evidence commit and the same archive, installation, reports and build evidence.

```sh
node --import tsx scripts/release/verify-release-evidence.ts \
  --kanana-current-evidence-commit "$evidence_commit" \
  --archive "$archive" --installed-package "$installed_package" \
  --playwright-report "$browser_report" --max-context-report "$context_report" \
  --build-evidence "$build_evidence"
```

The verifier reads the added file directly from Git, applies the current-release archive, provenance, test fixture, hardware and TVM rebuild checks, and verifies the two exact report byte strings against the post-build commit. Keep the evidence commit ID with the release record. Historical exact-package provenance stays at its original paths and the existing historical commands use it. The Git commit pins the evidence bytes. Reviewer identity and release authorization follow the release procedure.

## Kanana exact-package GPU evidence

The Kanana gate is separate from the Kev matrix and accepts only the npm-packed `web-decision@0.4.0` archive pinned in [release-evidence-policy.json](./release-evidence-policy.json). The policy binds the archive SHA-256, Kanana static manifest, checkpoint revision and digest, all five manifest file hashes, and the text and image reference fixtures. The text test checks its recorded TVM WebGPU output against the pinned Python/PyTorch fixture. The image test checks the selected option and probabilities against its pinned Python/PyTorch fixture. Every record carries the installed archive and entrypoint digests, exact text and vision Wasm hashes, the `navigator.gpu.requestAdapter().info` observation, adapter features and limits, and the model output. The policy pins the source hashes of all four GPU test files and their evidence helpers; release verification requires those files to match committed `HEAD` bytes.

Kanana candidate evidence accepts host data from Windows WMI or macOS `system_profiler`. The verifier cross-checks reported host and OS values against their source fields and rejects known virtual-machine signatures. The catalog's M4 Pro entry identifies the hardware used for its recorded run.

The offline case loads the text model with `cache: true`, completes a reference inference, then loads with `offline: true` while Playwright routes abort requests. The record must show positive priming bytes, no attempted requests, and matching online/offline outputs. A fourth pinned test exercises the 20,480-token boundary and validates the exact archive, truncation result, probability distribution, and WebGPU limits. All four GPU tests remain opt-in.

The Kanana gate also requires independent rebuild evidence for both packaged TVM modules: `kanana-text-model.wasm` and `kanana-vision-model.wasm`. Each evidence entry points to a successful `web-decision-tvm-build-v1` report and a rebuilt WASM inside a build directory outside both `packages/web-decision` and the installed archive. The verifier checks each artifact's byte count and SHA-256 against the archive-pinned manifest, checks the report's TVM version and output hash, and requires its repository, revision, checkpoint digest, graph name, `buildBackend: native-tvm`, `weightSource: pinned-safetensors`, and `onnxImportOrExportUsed: false` fields to match the Kanana pins. One record per artifact is required. Direct package paths and package hardlinks are rejected. Separate byte-for-byte copies cannot be distinguished from independent deterministic build outputs by hashes alone; retain the underlying build logs and review them with the report.

For a local exact-archive inference run before the policy is committed, set `WEB_DECISION_KANANA_LOCAL_GPU=1`. This mode still checks the current policy bytes against the verifier pin, verifies the exact archive against its installed package, and runs the snapshot, GPU, and inference assertions. Set `WEB_DECISION_KANANA_RELEASE_EVIDENCE=1` to emit structured records that the verifier can check. For strict release evidence, leave `WEB_DECISION_KANANA_LOCAL_GPU` unset and set both `WEB_DECISION_KANANA_RELEASE_EVIDENCE=1` and `WEB_DECISION_KANANA_CURATED_REAL_GPU=1`.

```sh
export WEB_DECISION_RELEASE_ARCHIVE=/release/web-decision-0.4.0.tgz
export WEB_DECISION_RELEASE_PACKAGE_DIR=/release/consumer/node_modules/web-decision
export WEB_DECISION_INSTALLED_BUNDLE=/release/consumer/node_modules/web-decision/dist/index.js
export WEB_DECISION_REAL_GPU=1
export WEB_DECISION_KANANA_RELEASE_EVIDENCE=1
export WEB_DECISION_KANANA_CURATED_REAL_GPU=1
export WEB_DECISION_BROWSER_NAME=chromium
export WEB_DECISION_BROWSER_CHANNEL=chrome
export PLAYWRIGHT_JSON_OUTPUT_NAME=/release/kanana-playwright-report.json
export WEB_DECISION_KANANA_MAX_CONTEXT_REPORT=/release/kanana-max-context-playwright-report.json
export WEB_DECISION_TVM_BUILD_EVIDENCE=/release/kanana-tvm-build-evidence.json
pnpm exec playwright test \
  tests/browser/kanana-curated-text-real-webgpu.spec.ts \
  tests/browser/kanana-curated-real-webgpu.spec.ts \
  tests/browser/kanana-curated-offline-text-real-webgpu.spec.ts \
  --reporter=json
WEB_DECISION_REAL_GPU=1 WEB_DECISION_KANANA_MAX_CONTEXT_GPU=1 \
  pnpm exec playwright test tests/browser/kanana-max-context-real-webgpu.spec.ts \
  --reporter=json > "$WEB_DECISION_KANANA_MAX_CONTEXT_REPORT"
node --import tsx scripts/release/verify-release-evidence.ts --kanana \
  --archive "$WEB_DECISION_RELEASE_ARCHIVE" \
  --installed-package "$WEB_DECISION_RELEASE_PACKAGE_DIR" \
  --playwright-report "$PLAYWRIGHT_JSON_OUTPUT_NAME" \
  --max-context-report "$WEB_DECISION_KANANA_MAX_CONTEXT_REPORT" \
  --build-evidence "$WEB_DECISION_TVM_BUILD_EVIDENCE"
```

The Kanana build-evidence file has one entry for each packaged WASM. `buildRoot` is resolved relative to the evidence file; `wasm` and `report` are resolved inside that root. Keep the independent compiler outputs and reports outside the source and installed package trees. The following is the record shape; replace paths and report values with those emitted by the independent builds:

```json
{
  "schemaVersion": 1,
  "builds": [
    {
      "modelId": "kanana-2-v-decide-3b-2611",
      "artifact": "kanana-text-model.wasm",
      "buildRoot": "kanana-build/text",
      "wasm": "kanana-text-model.wasm",
      "report": "release-build-report.json"
    },
    {
      "modelId": "kanana-2-v-decide-3b-2611",
      "artifact": "kanana-vision-model.wasm",
      "buildRoot": "kanana-build/vision",
      "wasm": "kanana-vision-model.wasm",
      "report": "release-build-report.json"
    }
  ]
}
```

Each report uses `evidenceSchema: web-decision-tvm-build-v1` and records `modelId`, `artifact`, `wasm`, `status: success`, `target: webgpu / wasm32`, `tvmVersion`, `embeddedBuildRoot: false`, `wasmBytes`, and `wasmSha256`. It also records the pinned `repository`, `modelRevision`, `checkpointSha256`, `graph` (`text` or `vision`), `buildBackend: native-tvm`, `weightSource: pinned-safetensors`, and `onnxImportOrExportUsed: false`. Build reports and WASM bytes are checked twice during release verification to detect changes during the run.

The archive must be an npm tarball with only regular files. The verifier compares every archive member byte-for-byte with the installed package directory, computes the archive SHA-256, package tree digest, and public bundle digest, then checks the static manifest at the path pinned in `artifacts.lock.json`. The manifest's complete `files` map is checked against the committed artifact lock. The public `assets.requiredFiles` list is derived as `['manifest.json', ...Object.keys(manifest.files)]`; the verifier derives the same full set from the committed lock and rejects a separately maintained source list. The bundled runtime files are a subset of this complete list and live in the lazy browser chunk. The record carries SHA-256 digests of the canonical installed package and entrypoint paths, not personal absolute paths; the gate recomputes both from its CLI input.

Nine shard files are reconstructed by the browser loader from the pinned upstream source and are not separate tarball members. For that reason, the browser record must include a `verifiedFiles` item for every manifest entry. The browser spec independently reads the complete CacheStorage entries after a successful load, measures their byte lengths, and computes SHA-256 values. The gate compares that measured list against all 14 entries in the archive-pinned manifest. It also requires online inference, an offline repeat with identical results and zero network attempts, and matches against both the pinned Windows baseline fixture and the one-input upstream CPU reference fixture.

Release evidence schema 2 also requires the committed [expanded inference fixture](../../tests/fixtures/kev06-expanded-inference.json), pinned by the policy and verifier. The real browser test passes its four decision requests (22, 64, 57 and 243 reference tokens) through `decide()`, then runs one `systemOne()` request containing a choice, yes/no (`noul`) and score question, followed by each question separately. The report carries each request, token count, option order, output, and descriptive maximum differences from the pinned ONNX CPU observations (ORT 1.31.1, CPUExecutionProvider, `ORT_ENABLE_ALL`, four threads). It checks required case IDs, exact fixture inputs, result shapes, finite logits, selected-option argmax, probability ranges and normalization. Each decision and typed question must select the same highest-probability option as its CPU reference, including packed and individual runs. Typed choice distributions must match their criterion keys/order; `noul` and score outputs must be in their declared ranges; score legends and levels must match the fixture. Typed probabilities are rounded to four decimal places by the public API, so their normalization check allows only the per-value rounding error plus floating-point summation error.

The expanded probability differences are recorded without a new numerical tolerance. Matching the reference selection for these inputs is not a general accuracy or model-quality claim. The browser API does not expose token IDs or option indices; those fields in the fixture describe the CPU reference packing only and are not claimed as current browser measurements. The test does not require packed and separate `systemOne()` probabilities to be equal. The original single-input baseline checks and their existing tolerances remain unchanged.

The release matrix contains only Chromium on Windows hardware. It makes no Firefox or WebKit verification claim. The gate requires the actual WebGPU adapter to be non-fallback and to report the catalog's required features and limits. Adapter Info can be empty; the gate requires the value and source to be recorded but does not invent adapter fields. GPU manufacturer, device name, driver, and PNP ID are collected separately from `Win32_VideoController`. The host check requires exactly one visible WMI video controller and rejects known virtual host or display-adapter signatures. It records `Win32_ComputerSystem.HypervisorPresent` as reported but does not treat `true` alone as proof of a virtual machine: Windows VBS can use the hypervisor on a physical device ([`Win32_ComputerSystem`](https://learn.microsoft.com/en-us/windows/win32/cimwin32prov/win32-computersystem), [VBS](https://learn.microsoft.com/en-us/windows-hardware/design/device-experiences/oem-vbs)). Accordingly, `device.isVirtual: false` is an observation-based screen for known signatures, not an absolute proof of physical hardware. WMI describes host-visible inventory and does not prove which adapter Chromium selected; browser Adapter Info remains a separate `navigator.gpu.requestAdapter().info` observation and may be empty. Host and OS metadata must likewise match their declared source data.

## Run

Pack and install the exact candidate archive first. Set the paths to that archive and the package directory that the test will load, then run the TypeScript Playwright spec and retain its JSON report:

```powershell
$env:WEB_DECISION_REAL_GPU = '1'
$env:WEB_DECISION_KEV06_DEFAULT_GPU = '1'
$env:WEB_DECISION_BROWSER_NAME = 'chromium'
$env:WEB_DECISION_BROWSER_CHANNEL = 'chrome'
$env:WEB_DECISION_RELEASE_ARCHIVE = 'C:\release\web-decision-x.y.z.tgz'
$env:WEB_DECISION_RELEASE_PACKAGE_DIR = 'C:\release\consumer\node_modules\web-decision'
$env:WEB_DECISION_INSTALLED_BUNDLE = 'C:\release\consumer\node_modules\web-decision\dist\index.js'
$env:PLAYWRIGHT_JSON_OUTPUT_NAME = 'C:\release\playwright-report.json'
$env:WEB_DECISION_TVM_BUILD_EVIDENCE = 'C:\release\tvm-build-evidence.json'
pnpm exec playwright test tests/browser/kev06-default-source-real-webgpu.spec.ts --reporter=json
```

`WEB_DECISION_KEV06_DEFAULT_GPU=1` is required to enable this model test; without it, Playwright marks the case skipped. `WEB_DECISION_REAL_GPU=1` launches Chromium with the real-GPU settings. On Windows, the example selects installed Chrome; use the browser channel that matches the host being recorded.

The Playwright project must be `chromium`. Its report must contain exactly one passing execution per curated model, with no skip, retry, flaky result, global error, or unrelated test. The browser spec must load the package at `WEB_DECISION_RELEASE_PACKAGE_DIR`.

For Kanana, invoke the separate mode with the same exact archive, installed package, Playwright report, and required rebuild evidence:

```powershell
./scripts/release/run-release-evidence.ps1 -Mode verify-kanana `
  -Archive $env:WEB_DECISION_RELEASE_ARCHIVE `
  -InstalledPackage $env:WEB_DECISION_RELEASE_PACKAGE_DIR `
  -PlaywrightReport $env:PLAYWRIGHT_JSON_OUTPUT_NAME `
  -MaxContextReport $env:WEB_DECISION_KANANA_MAX_CONTEXT_REPORT `
  -BuildEvidence $env:WEB_DECISION_TVM_BUILD_EVIDENCE
```

Supply a JSON build-evidence file with one entry for every `.wasm` in every curated model manifest. `buildRoot` is absolute or relative to the evidence file; `wasm` and `report` are resolved from that root. Both files must be inside the build root and outside the source and installed packages. Build the artifact independently from the pinned source and toolchain; do not copy the packaged WASM as a substitute. For the current build script, an entry has this shape:

```json
{
  "schemaVersion": 1,
  "builds": [
    {
      "modelId": "kev-0.6b-q4f16",
      "artifact": "model.wasm",
      "buildRoot": "path/to/independent-build",
      "wasm": "public/tvm-kev-q4-dynamic.wasm",
      "report": "reports/tvm-q4-build-dynamic.json"
    }
  ]
}
```

The verifier checks the actual rebuilt bytes, their size and SHA-256, and the successful build report against the committed manifest. The report must identify the model, artifact and built WASM, and record `target: webgpu / wasm32`, the manifest's TVM version, and no embedded build root. The verifier rejects the package artifact itself, including a hardlink where the filesystem reports inode identity; copies and fabricated reports cannot be detected as such. This is an integrity check, not a signature or independent proof of the reported build environment. Retain and review the source and toolchain records separately.

Run the gate using the same archive and installation paths:

```powershell
./scripts/release/run-release-evidence.ps1 -Mode verify `
  -Archive $env:WEB_DECISION_RELEASE_ARCHIVE `
  -InstalledPackage $env:WEB_DECISION_RELEASE_PACKAGE_DIR `
  -PlaywrightReport $env:PLAYWRIGHT_JSON_OUTPUT_NAME `
  -BuildEvidence $env:WEB_DECISION_TVM_BUILD_EVIDENCE
```

The PowerShell runner compiles the TypeScript verifier to a temporary directory before launching it, so Node 20 can run the CLI. Run the verifier's strict typecheck with:

```powershell
node node_modules/typescript/bin/tsc --project scripts/release/tsconfig.json
```

Run its tests with the same Node 20-compatible runner:

```powershell
./scripts/release/run-release-evidence.ps1 -Mode test
```

On Node 20 or newer, the tests can also be run with the project TypeScript loader:

```sh
node --import tsx --test scripts/release/tvm-build-evidence.test.ts scripts/release/verify-release-evidence.test.ts
```

## Evidence record

The evidence prefix and browser record use schema 2. The existing `inference.online` and `inference.offline` fields retain their one-input meanings and thresholds; the added observations are recorded separately as `expandedInference`.

Import the helper from the dedicated Playwright spec using `../../scripts/release/verify-release-evidence.ts`. Call it after the test has installed and exercised the candidate package. The helper returns the archive, sorted curated model requirements, catalog digest, and each exact static manifest. The `verifiedFiles` values below must come from the test's independent CacheStorage reads and SHA-256 calculations, not from copying the expected manifest values.

The `*ComparisonPassed` and `observedOfflineNetworkAttempts` names in the example stand for values produced by the browser test's real comparisons and request monitor; they are not constants to set to passing values. The observed model ID, descriptor digest, inference arrays, adapter values, host metadata, and cache measurements must come from the running test.

```ts
import assert from 'node:assert/strict';
import {realpathSync} from 'node:fs';
import {resolve} from 'node:path';
import {collectReleaseEvidenceInputs} from '../../scripts/release/verify-release-evidence.ts';

function observedInference(result) {
  const actualProbabilities = result.probabilities;
  return {
    completed: true,
    modelId: result.modelId,
    backend: result.backend,
    assetManifestSha256: result.assetManifestSha256,
    selected: result.selected,
    optionOrder: actualProbabilities.map(({option}) => option),
    logits: actualProbabilities.map(({logit}) => logit),
    probabilities: actualProbabilities.map(({probability}) => probability)
  };
}

const inputs = await collectReleaseEvidenceInputs({
  archivePath: process.env.WEB_DECISION_RELEASE_ARCHIVE!,
  installedPackagePath: process.env.WEB_DECISION_RELEASE_PACKAGE_DIR!,
  sourcePackageRoot: resolve(workspaceRoot, 'packages/web-decision')
});
assert.equal(realpathSync(resolve(process.env.WEB_DECISION_RELEASE_PACKAGE_DIR!)), inputs.archive.installedRoot);
assert.equal(realpathSync(resolve(process.env.WEB_DECISION_INSTALLED_BUNDLE!)), inputs.archive.bundlePath);
const model = inputs.models.find(({modelId}) => modelId === testedModelId)!;
const manifest = inputs.manifests.get(testedModelId)!;
assert.equal(online.modelId, testedModelId);
assert.equal(online.describedId, testedModelId);
assert.equal(online.assetManifestSha256, model.manifestSha256);
const baselineReferenceMatched = baselineFixtureComparisonPassed;
const upstreamReferenceMatched = upstreamFixtureComparisonPassed;
const offlineMatchesOnline = offlineOutputComparisonPassed;
const offlineNetworkAttempts = observedOfflineNetworkAttempts;
const record = {
  schemaVersion: 2,
  testKind: 'real-model-webgpu',
  catalogSha256: inputs.catalogSha256,
  model: {
    requestedId,
    loadedId: online.modelId,
    describedId: online.describedId,
    assetManifestSha256: online.assetManifestSha256
  },
  artifact: {
    modelId: testedModelId,
    manifestPath: manifest.path,
    manifestSha256: manifest.manifestSha256,
    filesSha256: manifest.filesSha256,
    manifest: manifest.manifest,
    verifiedFiles: cacheMeasurements.map(({path, bytes, sha256}) => ({path, bytes, sha256}))
  },
  archive: {
    sha256: inputs.archive.archiveSha256,
    packageTreeSha256: inputs.archive.packageTreeSha256,
    bundleSha256: inputs.archive.bundleSha256,
    installedPackagePathSha256: inputs.archive.installedPackagePathSha256,
    entryPathSha256: inputs.archive.entryPathSha256,
    packageName: inputs.archive.packageName,
    packageVersion: inputs.archive.packageVersion
  },
  browser: {name, version, userAgent},
  device: {
    manufacturer: hostManufacturer,
    model: hostModel,
    os,
    osVersion,
    architecture,
    isVirtual: false,
    evidence: {
      host: {source: 'windows-wmi/Win32_ComputerSystem', raw: {Manufacturer: hostManufacturer, Model: hostModel}},
      os: {source: 'node:os', raw: {version: os, release: osVersion, architecture}}
    }
  },
  gpu: {
    api: 'webgpu',
    isFallbackAdapter,
    adapterInfo,
    hardware: {manufacturer: gpuManufacturer, model: gpuModel, driverVersion, pnpDeviceId},
    hardwareEvidence: {
      source: 'windows-wmi/Win32_VideoController',
      raw: {
        AdapterCompatibility: gpuManufacturer,
        Name: gpuModel,
        DriverVersion: driverVersion,
        PNPDeviceID: pnpDeviceId
      }
    },
    features,
    limits
  },
  inference: {
    online: {...observedInference(online), referenceMatched: baselineReferenceMatched},
    offline: {
      ...observedInference(offline),
      matchesOnline: offlineMatchesOnline,
      networkAttempts: offlineNetworkAttempts
    },
    reference: {
      packageSha256: '5aef3331e41a67e20fac9fbc00e47074d02e1feeef4e0c7e6f4b13fdc87bca6f',
      fixtureSha256: '91297c6ebf012170b95c61bb5a053b6ed86e4e47a129fe5eaaa681d4ec8c74cd',
      scope: 'baseline-preservation-only'
    },
    upstreamReference: {
      matched: upstreamReferenceMatched,
      fixtureSha256: '656e8403984e57f24dd38f08d461188bde7802288c369a88dac0b29319bada10',
      modelId: 'kev-0.6b-q4f16',
      graphSha256: '26e059ddd954efd9897772b098db10b9eeda9b873984dc1c73b32a9ce177c7c8',
      weightsSha256: '777335dbc052e2e02ec7591f6eefcfb29e2c28caf52075e492d226fc27e0824b',
      scope: 'single-input-upstream-onnx-cpu-reference'
    }
  },
  expandedInference: expandedInferenceEvidence
};
console.log(`WEB_DECISION_RELEASE_EVIDENCE_V2:${JSON.stringify(record)}`);
```

The measured `verifiedFiles` list must contain exactly one `{path, bytes, sha256}` item for each path in `manifest.files`, including all reconstructed shards. `online` and `offline` each need `{completed: true, modelId, backend: 'tvm-webgpu', assetManifestSha256, selected, optionOrder, logits, probabilities}`. Derive `logits` and `probabilities` from the actual browser result's ordered probability entries. Do not include `optionIndices`: the public browser API does not expose the original token indices. The verifier compares the option order and each output value against both pinned references using their declared tolerances, then requires offline output to equal online output. Online must set `referenceMatched: true`; offline must set `matchesOnline: true` and `networkAttempts: []`.

The gate reads both fixture files from the committed tree, checks their exact SHA-256 values, and checks the baseline package and upstream ONNX/weight digests inside those files. The old-main hashes establish preservation against the old Windows archive and baseline fixture. A separate upstream block pins the one-input CPU reference produced from the independently hash-verified ONNX and weight source. This compares one input only and is not a general model quality claim. The verifier checks that the report text is internally consistent with the exact archive, installed files, catalog, manifest, browser, and hardware fields. It does not cryptographically authenticate the browser run, the machine, or the text report.
