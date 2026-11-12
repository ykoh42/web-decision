import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {readFile} from 'node:fs/promises';
import {relative, resolve, sep} from 'node:path';
import type {TestInfo} from '@playwright/test';
import {
  canonicalSha256,
  loadInstalledCatalog,
  sha256,
  verifyArchiveInstall
} from '../../scripts/release/verify-release-evidence.ts';
import {collectHardwareEvidence} from './kev06-release-evidence.ts';
import {
  collectInstalledKananaReleaseInputs,
  createKananaReleaseEvidenceEnvelope,
  KANANA_RELEASE_EVIDENCE_PREFIX
} from './kanana-release-evidence.ts';

export {KANANA_RELEASE_EVIDENCE_PREFIX};

type TestKind = 'text' | 'image' | 'offline';

type KananaModelObservation = Readonly<{
  requestedId: string;
  modelId: string;
  descriptionModelId: string;
  checkpointRevision: string;
  checkpointSha256: string;
  assetManifestSha256: string;
  backend: string;
}>;

type AdapterObservation = Readonly<{
  isFallback?: boolean | null;
  isFallbackAdapter?: boolean | null;
  vendor: string;
  architecture: string;
  device?: string;
  features: readonly string[];
  limits: Readonly<Record<string, number>>;
}>;

function requiredEnvironment(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`Set ${name} to the exact Kanana candidate package.`);
  return resolve(value);
}

function kananaPolicy() {
  const policyPath = new URL('../../scripts/release/release-evidence-policy.json', import.meta.url);
  const policy = JSON.parse(readFileSync(policyPath, 'utf8')) as {
    kanana?: {
      model?: {
        modelId?: string;
        repository?: string;
        revision?: string;
        checkpointSha256?: string;
        manifestPath?: string;
        manifestRawSha256?: string;
        manifestSha256?: string;
      };
      tests?: Partial<Record<TestKind, Readonly<{file: string; title: string}>>>;
    };
  };
  if (!policy.kanana) throw new Error('The release policy has no Kanana evidence block.');
  if (!policy.kanana.model) throw new Error('The Kanana evidence policy has no model pin.');
  return {...policy.kanana.model, tests: policy.kanana.tests};
}

export async function createKananaReleaseEvidence(
  input: Readonly<{
    kind: TestKind;
    testInfo: TestInfo;
    browserName: string;
    browserVersion: string;
    userAgent: string;
    model: KananaModelObservation;
    adapter: AdapterObservation;
    observation: unknown;
  }>
): Promise<Readonly<Record<string, unknown>> | undefined> {
  if (process.env.WEB_DECISION_KANANA_RELEASE_EVIDENCE !== '1') return undefined;

  const policy = kananaPolicy();
  const provenancePath = new URL(
    '../../packages/web-decision/src/catalog/provenance/kanana-2-v-decide-3b-2611.json',
    import.meta.url
  );
  const provenance = JSON.parse(readFileSync(provenancePath, 'utf8')) as {status?: string};
  assert.equal(provenance.status, 'verified', 'Kanana provenance must pass source and rights checks.');
  const testRequirement = policy.tests?.[input.kind];
  assert.ok(testRequirement, `The policy does not require a Kanana ${input.kind} test.`);
  const workspaceRoot = resolve(process.cwd());
  const testPath = relative(workspaceRoot, input.testInfo.file).split(sep).join('/');
  assert.equal(testPath, testRequirement.file, 'Kanana candidate evidence came from an unpinned test file.');
  assert.equal(input.testInfo.title, testRequirement.title, 'Kanana candidate test title differs from policy.');

  const archive = await verifyArchiveInstall(
    requiredEnvironment('WEB_DECISION_RELEASE_ARCHIVE'),
    requiredEnvironment('WEB_DECISION_RELEASE_PACKAGE_DIR')
  );
  assert.equal(
    requiredEnvironment('WEB_DECISION_INSTALLED_BUNDLE'),
    archive.bundlePath,
    'The browser bundle must be the entrypoint installed from the candidate archive.'
  );
  const catalog = await loadInstalledCatalog(archive.installedRoot);
  const descriptor = catalog.find(({modelId}) => modelId === policy.modelId);
  assert.ok(descriptor, `The installed candidate package does not expose ${policy.modelId}.`);
  assert.equal(descriptor.backend.name, 'tvm-webgpu');
  assert.equal(descriptor.manifestSha256, policy.manifestSha256);

  const manifestBytes = await readFile(resolve(archive.installedRoot, policy.manifestPath!));
  assert.equal(sha256(manifestBytes), policy.manifestRawSha256);
  const manifest = JSON.parse(manifestBytes.toString('utf8')) as {
    modelId?: string;
    repository?: string;
    revision?: string;
    checkpointSha256?: string;
    files?: Record<string, {bytes: number; sha256: string}>;
  };
  assert.equal(canonicalSha256(manifest), policy.manifestSha256);
  assert.equal(manifest.modelId, policy.modelId);
  assert.equal(manifest.repository, policy.repository);
  assert.equal(manifest.revision, policy.revision);
  assert.equal(manifest.checkpointSha256, policy.checkpointSha256);
  assert.deepEqual(input.model, {
    requestedId: policy.modelId,
    modelId: policy.modelId,
    descriptionModelId: policy.modelId,
    checkpointRevision: policy.revision,
    checkpointSha256: policy.checkpointSha256,
    assetManifestSha256: policy.manifestSha256,
    backend: 'tvm-webgpu'
  });

  const packageFiles: Record<string, {bytes: number; sha256: string}> = {};
  for (const [name, metadata] of Object.entries(manifest.files ?? {})) {
    const path = resolve(archive.installedRoot, 'assets/kanana', name);
    const bytes = await readFile(path);
    const fileDigest = sha256(bytes);
    assert.equal(bytes.byteLength, metadata.bytes, `Kanana candidate asset size differs for ${name}.`);
    assert.equal(fileDigest, metadata.sha256, `Kanana candidate asset digest differs for ${name}.`);
    packageFiles[name] = {bytes: bytes.byteLength, sha256: fileDigest};
  }

  const hardware = collectHardwareEvidence();
  const hostEvidence = hardware.device.evidence.host;
  assert.equal(hostEvidence.source, 'macos-system_profiler/SPHardwareDataType');
  if (hostEvidence.source !== 'macos-system_profiler/SPHardwareDataType')
    throw new Error('Kanana M4 release evidence requires macOS host data.');
  assert.match(hostEvidence.raw.chip_type, /Apple M4 Pro/u);
  const isFallbackAdapter = input.adapter.isFallbackAdapter ?? input.adapter.isFallback;
  assert.equal(isFallbackAdapter, false);
  assert.equal(input.browserName, 'chromium');

  const releaseInputs = await collectInstalledKananaReleaseInputs();
  const evidenceKind = {
    text: 'text-choice',
    image: 'synthetic-image-choice',
    offline: 'cached-offline-text-choice'
  } as const;
  const envelope = createKananaReleaseEvidenceEnvelope({
    inputs: releaseInputs,
    kind: evidenceKind[input.kind],
    runtimeArtifactNames: releaseInputs.policy.tests[input.kind].runtimeArtifacts,
    browserName: input.browserName,
    browserVersion: input.browserVersion,
    userAgent: input.userAgent,
    model: {
      loadedId: input.model.modelId,
      describedId: input.model.descriptionModelId,
      checkpointRevision: input.model.checkpointRevision,
      checkpointSha256: input.model.checkpointSha256
    },
    gpu: {
      isFallbackAdapter,
      adapterInfo: {
        vendor: input.adapter.vendor,
        architecture: input.adapter.architecture,
        isFallbackAdapter
      },
      features: input.adapter.features,
      limits: input.adapter.limits
    }
  });
  const observation = input.observation as Record<string, any>;
  const device = {
    manufacturer: hardware.device.manufacturer,
    model: hardware.device.model,
    host: hostEvidence.raw.chip_type,
    os: hardware.device.os,
    osVersion: hardware.device.osVersion,
    architecture: hardware.device.architecture,
    isVirtual: hardware.device.isVirtual,
    evidence: hardware.device.evidence
  };
  if (input.kind === 'text')
    return {
      ...envelope,
      device,
      referenceFixtureSha256: observation.referenceFixtureSha256,
      probabilityTolerance: observation.probabilityTolerance,
      inference: observation.result
    };
  if (input.kind === 'image')
    return {
      ...envelope,
      device,
      input: observation.input,
      referenceFixtureSha256: observation.referenceFixtureSha256,
      inference: observation.result
    };
  return {
    ...envelope,
    device,
    referenceFixtureSha256: observation.referenceFixtureSha256,
    probabilityTolerance: releaseInputs.policy.textProbabilityTolerance,
    cache: {
      primedOnline: true,
      requestedOffline: true,
      blockedNetwork: true,
      fetchedRanges: observation.fetchedRanges,
      fetchedBytes: observation.fetchedBytes
    },
    networkAttempts: observation.networkAttempts,
    inference: {online: observation.online, offline: observation.offline}
  };
}
