import assert from 'node:assert/strict';
import {realpathSync} from 'node:fs';
import {resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {
  collectKananaReleaseEvidenceInputs,
  type KananaReleaseEvidenceInputs
} from '../../scripts/release/verify-release-evidence.ts';

export const KANANA_RELEASE_EVIDENCE_PREFIX = 'WEB_DECISION_KANANA_RELEASE_EVIDENCE_V1:';
export const localKananaCandidateMode = process.env.WEB_DECISION_KANANA_LOCAL_GPU === '1';

export function requiredKananaEnvironment(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`${name} is required for Kanana release evidence.`);
  return value;
}

export async function collectInstalledKananaReleaseInputs(): Promise<KananaReleaseEvidenceInputs> {
  const archivePath = requiredKananaEnvironment('WEB_DECISION_RELEASE_ARCHIVE');
  const installedPackagePath = requiredKananaEnvironment('WEB_DECISION_RELEASE_PACKAGE_DIR');
  const installedBundlePath = requiredKananaEnvironment('WEB_DECISION_INSTALLED_BUNDLE');
  const workspaceRoot = resolve(fileURLToPath(new URL('../..', import.meta.url)));
  const inputs = await collectKananaReleaseEvidenceInputs({
    archivePath,
    installedPackagePath,
    workspaceRoot,
    localCandidate: localKananaCandidateMode
  });
  assert.equal(realpathSync(resolve(installedPackagePath)), inputs.archive.installedRoot);
  assert.equal(realpathSync(resolve(installedBundlePath)), inputs.archive.bundlePath);
  return inputs;
}

export function createKananaReleaseEvidenceEnvelope({
  inputs,
  kind,
  runtimeArtifactNames,
  browserName,
  browserVersion,
  userAgent,
  model,
  gpu
}: {
  inputs: KananaReleaseEvidenceInputs;
  kind: 'text-choice' | 'synthetic-image-choice' | 'cached-offline-text-choice';
  runtimeArtifactNames: readonly string[];
  browserName: string;
  browserVersion: string;
  userAgent: string;
  model: Readonly<{
    loadedId: string;
    describedId: string;
    checkpointRevision: string;
    checkpointSha256: string;
  }>;
  gpu: Readonly<{
    isFallbackAdapter: boolean;
    adapterInfo: Readonly<{vendor: string; architecture: string; isFallbackAdapter: boolean}>;
    features: readonly string[];
    limits: Readonly<Record<string, number>>;
  }>;
}) {
  const {archive, manifest, policy} = inputs;
  return {
    schemaVersion: 1,
    kind,
    archive: {
      sha256: archive.archiveSha256,
      packageTreeSha256: archive.packageTreeSha256,
      bundleSha256: archive.bundleSha256,
      installedPackagePathSha256: archive.installedPackagePathSha256,
      entryPathSha256: archive.entryPathSha256,
      packageName: archive.packageName,
      packageVersion: archive.packageVersion
    },
    artifact: {
      modelId: policy.model.modelId,
      manifestPath: manifest.path,
      manifestRawSha256: manifest.rawSha256,
      manifestSha256: manifest.manifestSha256,
      filesSha256: manifest.filesSha256,
      files: manifest.files,
      runtimeArtifacts: runtimeArtifactNames.map((path) => {
        const metadata = manifest.files[path];
        if (!metadata) throw new Error(`Kanana release policy omits runtime artifact ${path}.`);
        return {path, ...metadata};
      })
    },
    model: {
      requestedId: policy.model.modelId,
      loadedId: model.loadedId,
      describedId: model.describedId,
      checkpointRevision: model.checkpointRevision,
      checkpointSha256: model.checkpointSha256
    },
    browser: {name: browserName, version: browserVersion, userAgent},
    gpu: {
      api: policy.gpu.api,
      isFallbackAdapter: gpu.isFallbackAdapter,
      adapterInfo: {
        source: 'navigator.gpu.requestAdapter() and GPUAdapterInfo',
        value: gpu.adapterInfo
      },
      features: [...gpu.features],
      limits: {...gpu.limits}
    }
  };
}
