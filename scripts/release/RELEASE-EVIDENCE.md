# Release browser evidence gate

The gate connects a candidate npm archive, the package installed by the browser test, curated catalog inputs, static asset manifests, independently built TVM artifacts, and one real Chromium WebGPU test per curated model. It accepts only the dedicated test file pinned by [release-evidence-policy.json](./release-evidence-policy.json). The policy bytes are pinned in the verifier and must also match the committed `HEAD` copy.

Before collecting evidence, run `pnpm run check:model-provenance`. The verifier independently applies the same fail-closed check to the committed model spec, artifact lock, provenance manifest, and evidence files. The manifest has a source-code-pinned SHA-256; changing only a model status cannot approve it. Every entry must match the complete public model ID set and exact artifact-lock bytes. The verifier accepts only the canonical `packages/web-decision` source root, so the explicitly synthetic release-test fixture cannot be used for a release. A `pending` record produces an error that names the unresolved lineage or rights.

The expected model IDs and artifact digests come from the committed `packages/web-decision/src/catalog/model-spec.json` and `artifacts.lock.json`. The gate checks that the model spec, artifact lock, and pinned provenance manifest have the same non-empty ID set, that each manifest digest and static manifest path match the lock, and that the built and installed public APIs (`listModels()` and `describeModel(id)`) match those committed inputs. It does not read the legacy model registry. The source API is built from `src/curated/index.ts`; the verifier imports its built `dist/index.js` so the CLI also runs on Node 20, which cannot execute TypeScript source directly.

The provenance gate checks committed evidence hashes and binds any declared upstream source to the artifact lock. It does not independently authenticate the reviewer, verify remote source claims, or prove that the reviewed documents are true; those remain human review responsibilities.

## Kanana exact-package GPU evidence

The Kanana gate is separate from the Kev matrix and accepts only the npm-packed `web-decision@0.4.0` archive pinned in [release-evidence-policy.json](./release-evidence-policy.json). The policy binds the archive SHA-256, Kanana static manifest, checkpoint revision and digest, all five manifest file hashes, and the text and image reference fixtures. The text test checks its recorded TVM WebGPU output against the pinned Python/PyTorch fixture. The image test checks the selected option and probabilities against its pinned Python/PyTorch fixture. Every record carries the installed archive and entrypoint digests, exact text and vision Wasm hashes, the `navigator.gpu.requestAdapter().info` observation, adapter features and limits, and the model output. The policy pins the source hashes of all four GPU test files and their evidence helpers; release verification requires those files to match committed `HEAD` bytes.

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
  "builds": [{
    "modelId": "kev-0.6b-q4f16",
    "artifact": "model.wasm",
    "buildRoot": "path/to/independent-build",
    "wasm": "public/tvm-kev-q4-dynamic.wasm",
    "report": "reports/tvm-q4-build-dynamic.json"
  }]
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
    manufacturer: hostManufacturer, model: hostModel, os, osVersion, architecture, isVirtual: false,
    evidence: {
      host: {source: 'windows-wmi/Win32_ComputerSystem', raw: {Manufacturer: hostManufacturer, Model: hostModel}},
      os: {source: 'node:os', raw: {version: os, release: osVersion, architecture}}
    }
  },
  gpu: {
    api: 'webgpu', isFallbackAdapter, adapterInfo,
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
    features, limits
  },
  inference: {
    online: {...observedInference(online), referenceMatched: baselineReferenceMatched},
    offline: {...observedInference(offline), matchesOnline: offlineMatchesOnline, networkAttempts: offlineNetworkAttempts},
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
