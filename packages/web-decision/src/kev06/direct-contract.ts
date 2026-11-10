import {validateManifestFiles, type KevAssetManifest} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {readPinnedManifest} from '../pinned-manifest.js';

export const DIRECT_KEV06_MODEL = 'kev-0.6b-q4f16';
export const DIRECT_KEV06_PARAMETER_COUNT = 511;
export const DIRECT_KEV06_CHECKPOINT = Object.freeze({
  repository: 'jaredpalmer/kev-0.6b',
  revision: 'abe7110dedfd68a185839550079bcf7c26f561ac'
});
export const DIRECT_KEV06_BASE = Object.freeze({
  repository: 'Qwen/Qwen3-0.6B-Base',
  revision: 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
});

const REQUIRED_FILES = [
  'model.wasm',
  'tokenizer.json',
  'tokenizer_config.json',
  'model_config.json',
  'weights/tensor-cache.json'
] as const;

type DirectKev06Manifest = KevAssetManifest & {
  readonly model: typeof DIRECT_KEV06_MODEL;
  readonly checkpoint: typeof DIRECT_KEV06_CHECKPOINT;
  readonly base: typeof DIRECT_KEV06_BASE;
};

export interface DirectKev06CatalogPin {
  readonly modelId: string;
  readonly checkpoint: {readonly repository: string; readonly revision: string};
  readonly assets: {readonly manifestSha256: string; readonly directArtifactRoot?: string};
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function invalidManifest(): never {
  throw new WebDecisionError('INVALID_ASSET', 'The manifest does not match the direct Kev TVM package.');
}

/** Validate the self-converted Kev package, pinned by the curated catalog digest. */
export async function readDirectKev06ManifestResponse(
  response: Response,
  expectedSha256: string,
  signal?: AbortSignal
): Promise<DirectKev06Manifest> {
  if (!/^[0-9a-f]{64}$/.test(expectedSha256)) invalidManifest();
  const value = await readPinnedManifest(response, signal, expectedSha256, 'direct Kev');
  if (!isRecord(value) || !isRecord(value.checkpoint) || !isRecord(value.base)) invalidManifest();

  const checkpoint = value.checkpoint;
  const base = value.base;
  if (
    value.formatVersion !== 1 ||
    value.model !== DIRECT_KEV06_MODEL ||
    value.backend !== 'tvm-webgpu' ||
    Object.hasOwn(value, 'maxInputTokens') ||
    value.parameterCount !== DIRECT_KEV06_PARAMETER_COUNT ||
    value.maxStateTokens !== 8192 ||
    value.maxBranchTokens !== 8192 ||
    checkpoint.repository !== DIRECT_KEV06_CHECKPOINT.repository ||
    checkpoint.revision !== DIRECT_KEV06_CHECKPOINT.revision ||
    base.repository !== DIRECT_KEV06_BASE.repository ||
    base.revision !== DIRECT_KEV06_BASE.revision
  )
    invalidManifest();

  validateManifestFiles(value.files, REQUIRED_FILES, 'direct Kev');
  return value as unknown as DirectKev06Manifest;
}

/** Select direct loading only after the curated catalog pins this checkpoint. */
export function hasDirectKev06CatalogPin(description: DirectKev06CatalogPin): boolean {
  const artifactRoot = description.assets.directArtifactRoot;
  let validArtifactRoot = false;
  if (typeof artifactRoot === 'string') {
    try {
      const url = new URL(artifactRoot);
      validArtifactRoot =
        url.protocol === 'https:' &&
        url.pathname.endsWith('/') &&
        !url.search &&
        !url.hash &&
        !url.username &&
        !url.password;
    } catch {
      validArtifactRoot = false;
    }
  }
  return (
    description.modelId === DIRECT_KEV06_MODEL &&
    description.checkpoint.repository === DIRECT_KEV06_CHECKPOINT.repository &&
    description.checkpoint.revision === DIRECT_KEV06_CHECKPOINT.revision &&
    /^[0-9a-f]{64}$/.test(description.assets.manifestSha256) &&
    validArtifactRoot
  );
}
