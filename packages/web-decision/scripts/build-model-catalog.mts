import {createHash} from 'node:crypto';
import {lstat, readdir, readFile, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import type {
  CatalogModelDescription,
  ModelAdapterId,
  ModelArtifactLock,
  ModelCatalogEntry,
  ModelId
} from '../src/catalog/types.generated.js';
import {KEV06_MANIFEST_SHA256} from '../src/kev06/contract.js';
import {
  DIRECT_KEV06_BASE,
  DIRECT_KEV06_CHECKPOINT,
  DIRECT_KEV06_PARAMETER_COUNT,
  readDirectKev06ManifestResponse
} from '../src/kev06/direct-contract.js';
import {KANANA_MODEL_ID, KANANA_REPOSITORY, KANANA_REVISION} from '../src/kanana/contract.mjs';
import {KANANA_SAFETENSORS_SPEC} from '../src/kanana/safetensors-spec.mjs';
import {
  EXPERIMENTAL_KEV_UPSTREAM_MAP_SHA256,
  EXPERIMENTAL_KEV_UPSTREAM_REPOSITORY,
  EXPERIMENTAL_KEV_UPSTREAM_REVISION,
  EXPERIMENTAL_KEV_UPSTREAM_SOURCE_BYTES,
  EXPERIMENTAL_KEV_UPSTREAM_SOURCE_PATH,
  EXPERIMENTAL_KEV_UPSTREAM_SOURCE_SHA256
} from '../src/kev06/upstream-contract.js';

type CuratedModel = Omit<ModelCatalogEntry, 'description' | 'artifacts'> & {
  readonly artifactUpstreams?: readonly DirectKevUpstream[];
  readonly description: Omit<CatalogModelDescription, 'assets'> & {
    readonly adapter: ModelAdapterId;
    readonly assets: Omit<CatalogModelDescription['assets'], 'manifestSha256' | 'requiredFiles'>;
  };
};

interface ManifestFile {
  readonly sha256: string;
  readonly bytes: number;
}

interface DirectKevUpstream {
  readonly role: string;
  readonly repository: string;
  readonly revision: string;
  readonly sourcePath: string;
  readonly sourceBytes: number;
  readonly sourceSha256: string;
}

type DirectModelArtifactLock = ModelArtifactLock & {
  readonly upstreams: readonly DirectKevUpstream[];
};

type CatalogArtifactLock = ModelArtifactLock & {
  readonly upstreams?: readonly DirectKevUpstream[];
};

interface KevManifest {
  readonly model: string;
  readonly backend: string;
  readonly checkpoint: Readonly<{repository: string; revision: string; weightSha256: string}>;
  readonly experimental: boolean;
  readonly files: Readonly<Record<string, ManifestFile>>;
}

interface DirectKevManifest {
  readonly model: string;
  readonly backend: string;
  readonly experimental: boolean;
  readonly files: Readonly<Record<string, ManifestFile>>;
  readonly formatVersion: number;
  readonly maxStateTokens: number;
  readonly maxBranchTokens: number;
  readonly parameterCount: number;
  readonly checkpoint: Readonly<{repository: string; revision: string}>;
  readonly base: Readonly<{repository: string; revision: string}>;
}

interface ArtifactLockFile {
  readonly schemaVersion: 1;
  readonly byModelId: Readonly<Record<string, CatalogArtifactLock>>;
}

interface JsonFile {
  readonly bytes: Buffer;
  readonly value: unknown;
}

interface CatalogGenerationOptions {
  readonly directKevArtifactDirectory?: string;
}

const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repositoryRoot = path.resolve(packageRoot, '../..');
const catalogRoot = path.join(packageRoot, 'src/catalog');
const specPath = path.join(catalogRoot, 'model-spec.json');
const manifestPath = 'assets/kev06-stable/manifest.json';
const directKevArtifactDirectory = path.join(packageRoot, 'assets/kev06-direct');
const directKevManifestSha256 = '78988880bd42241c01729bed42c768b81a0cbc48d51e45de64845c64ea18b93a';
const mapPath = 'assets/experimental-kev-upstream/source-shard-map.json';
const bundledFiles: readonly (readonly [string, string])[] = [
  ['model.wasm', 'assets/kev06-stable/model.wasm'],
  ['tokenizer.json', 'assets/experimental-kev-upstream/tokenizer.json'],
  ['tokenizer_config.json', 'assets/experimental-kev-upstream/tokenizer_config.json'],
  ['model_config.json', 'assets/experimental-kev-upstream/model_config.json'],
  ['weights/tensor-cache.json', 'assets/experimental-kev-upstream/weights/tensor-cache.json']
];
const kananaManifestPath = 'assets/kanana/manifest.json';

function failUnless(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(
    Object.keys(value)
      .sort()
      .map((key) => [key, canonical((value as Record<string, unknown>)[key])])
  );
}

function canonicalManifestBytes(value: unknown): Buffer {
  const serialized = JSON.stringify(canonical(value));
  failUnless(typeof serialized === 'string', 'Manifest could not be serialized.');
  return Buffer.from(serialized + '\n');
}

function isDirectKevCheckpoint(model: CuratedModel): boolean {
  const checkpoint = model.description.checkpoint;
  return (
    checkpoint.repository === DIRECT_KEV06_CHECKPOINT.repository ||
    checkpoint.revision === DIRECT_KEV06_CHECKPOINT.revision
  );
}

function isHttpsDirectory(value: unknown): value is string {
  if (typeof value !== 'string') return false;
  try {
    const url = new URL(value);
    return (
      url.protocol === 'https:' &&
      url.pathname.endsWith('/') &&
      !url.search &&
      !url.hash &&
      !url.username &&
      !url.password
    );
  } catch {
    return false;
  }
}

function safeManifestPath(value: string): boolean {
  return (
    value.length > 0 &&
    value !== 'manifest.json' &&
    !value.startsWith('/') &&
    !value.includes(':') &&
    !/[\\?#%]/u.test(value) &&
    !containsControlCharacter(value) &&
    value.split('/').every((part) => part !== '' && part !== '.' && part !== '..')
  );
}

function containsControlCharacter(value: string): boolean {
  return Array.from(value).some((character) => {
    const codePoint = character.codePointAt(0) ?? 0;
    return codePoint < 0x20 || codePoint === 0x7f;
  });
}

function isDirectKevUpstream(value: unknown): value is DirectKevUpstream {
  if (!isRecord(value)) return false;
  const keys = Object.keys(value).sort();
  return (
    JSON.stringify(keys) ===
      JSON.stringify(['repository', 'revision', 'role', 'sourceBytes', 'sourcePath', 'sourceSha256']) &&
    typeof value.role === 'string' &&
    /^[a-z0-9][a-z0-9-]*$/u.test(value.role) &&
    typeof value.repository === 'string' &&
    /^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/u.test(value.repository) &&
    typeof value.revision === 'string' &&
    /^[0-9a-f]{40}$/u.test(value.revision) &&
    typeof value.sourcePath === 'string' &&
    safeManifestPath(value.sourcePath) &&
    Number.isSafeInteger(value.sourceBytes) &&
    (value.sourceBytes as number) > 0 &&
    typeof value.sourceSha256 === 'string' &&
    /^[0-9a-f]{64}$/u.test(value.sourceSha256)
  );
}

function validateDirectKevUpstreams(value: unknown): readonly DirectKevUpstream[] {
  failUnless(Array.isArray(value) && value.length > 0, 'Direct Kev catalog must pin its conversion inputs.');
  const roles = new Set<string>();
  const sources = new Set<string>();
  for (const candidate of value) {
    failUnless(isDirectKevUpstream(candidate), 'Direct Kev conversion input pin is invalid.');
    const source = [candidate.repository, candidate.revision, candidate.sourcePath].join('\0');
    failUnless(!roles.has(candidate.role), 'Direct Kev conversion input roles must be unique.');
    failUnless(!sources.has(source), 'Direct Kev conversion input sources must be unique.');
    roles.add(candidate.role);
    sources.add(source);
  }
  return value as readonly DirectKevUpstream[];
}

function validatePrivateArtifactUpstreams(model: Record<string, unknown>): void {
  const description = model.description;
  const checkpoint = isRecord(description) ? description.checkpoint : undefined;
  failUnless(
    model.modelId === 'kev-0.6b-q4f16' &&
      isRecord(description) &&
      description.adapter === 'kev06' &&
      isRecord(checkpoint) &&
      checkpoint.repository === DIRECT_KEV06_CHECKPOINT.repository &&
      checkpoint.revision === DIRECT_KEV06_CHECKPOINT.revision,
    'Artifact upstream pins are direct Kev-only.'
  );
  validateDirectKevUpstreams(model.artifactUpstreams);
}

function isDirectKevSpecModel(model: Record<string, unknown>): boolean {
  const description = model.description;
  const checkpoint = isRecord(description) ? description.checkpoint : undefined;
  return (
    isRecord(description) &&
    isRecord(checkpoint) &&
    (checkpoint.repository === DIRECT_KEV06_CHECKPOINT.repository ||
      checkpoint.revision === DIRECT_KEV06_CHECKPOINT.revision)
  );
}

function validateModelPrivateInputs(model: Record<string, unknown>): void {
  if (isDirectKevSpecModel(model)) {
    validatePrivateArtifactUpstreams(model);
    return;
  }
  failUnless(!Object.hasOwn(model, 'artifactUpstreams'), 'Artifact upstream pins are direct Kev-only.');
}

function validateDirectKevManifestFiles(files: unknown): asserts files is Readonly<Record<string, ManifestFile>> {
  failUnless(isRecord(files), 'Direct Kev manifest must contain a file inventory.');
  const names = Object.keys(files).sort();
  failUnless(names.length > 0, 'Direct Kev manifest must contain a file inventory.');
  for (const name of names) {
    failUnless(safeManifestPath(name), 'Direct Kev manifest contains an unsafe file path: ' + name + '.');
    const metadata = files[name];
    const metadataKeys = isRecord(metadata) ? Object.keys(metadata).sort() : [];
    failUnless(
      isRecord(metadata) &&
        JSON.stringify(metadataKeys) === JSON.stringify(['bytes', 'sha256']) &&
        typeof metadata.sha256 === 'string' &&
        /^[0-9a-f]{64}$/u.test(metadata.sha256) &&
        Number.isSafeInteger(metadata.bytes) &&
        (metadata.bytes as number) > 0,
      'Direct Kev manifest file metadata is invalid: ' + name + '.'
    );
  }
}

async function listLocalFiles(directory: string): Promise<string[]> {
  const files: string[] = [];
  const visit = async (current: string, relative: string): Promise<void> => {
    for (const entry of await readdir(current, {withFileTypes: true})) {
      const name = relative ? `${relative}/${entry.name}` : entry.name;
      const fullPath = path.join(current, entry.name);
      if (entry.isSymbolicLink()) throw new Error('Direct Kev package must not contain symbolic links.');
      if (entry.isDirectory()) await visit(fullPath, name);
      else if (entry.isFile()) files.push(name);
      else throw new Error('Direct Kev package contains a non-regular entry: ' + name + '.');
    }
  };
  await visit(directory, '');
  return files.sort();
}

function validateDirectKevModel(model: CuratedModel): readonly DirectKevUpstream[] {
  failUnless(model.modelId === 'kev-0.6b-q4f16', 'Direct Kev package model ID differs from the curated model.');
  failUnless(model.description.adapter === 'kev06', 'Direct Kev package requires the Kev adapter.');
  failUnless(
    model.description.checkpoint.repository === DIRECT_KEV06_CHECKPOINT.repository &&
      model.description.checkpoint.revision === DIRECT_KEV06_CHECKPOINT.revision,
    'Direct Kev catalog checkpoint differs from the direct source pin.'
  );
  return validateDirectKevUpstreams(model.artifactUpstreams);
}

function validateDirectKevManifestMetadata(model: CuratedModel, manifest: DirectKevManifest): void {
  failUnless(typeof manifest.experimental === 'boolean', 'Direct Kev manifest must preserve its experimental flag.');
  failUnless(manifest.model === model.modelId, 'Direct Kev manifest model differs from the curated model.');
  failUnless(
    manifest.backend === model.description.backend.name,
    'Direct Kev manifest backend differs from the catalog.'
  );
  failUnless(
    manifest.parameterCount === DIRECT_KEV06_PARAMETER_COUNT,
    'Direct Kev parameter count differs from its pin.'
  );
  failUnless(
    manifest.checkpoint.repository === model.description.checkpoint.repository &&
      manifest.checkpoint.revision === model.description.checkpoint.revision,
    'Direct Kev checkpoint differs from the catalog.'
  );
  failUnless(
    manifest.base.repository === DIRECT_KEV06_BASE.repository && manifest.base.revision === DIRECT_KEV06_BASE.revision,
    'Direct Kev base checkpoint differs from its pin.'
  );
  validateDirectKevManifestFiles(manifest.files);
}

async function validateAvailableDirectKevFiles(directory: string, manifest: DirectKevManifest): Promise<void> {
  const actualNames = await listLocalFiles(directory);
  const localAssets = actualNames.filter((name) => name !== 'manifest.json');
  for (const name of localAssets) {
    const metadata = manifest.files[name];
    failUnless(metadata, 'Direct Kev package contains an undeclared file: ' + name + '.');
    const bytes = await readFile(path.join(directory, ...name.split('/')));
    failUnless(metadata.bytes === bytes.byteLength, 'Direct Kev file size differs from manifest: ' + name + '.');
    failUnless(metadata.sha256 === sha256(bytes), 'Direct Kev file hash differs from manifest: ' + name + '.');
  }
}

export async function createDirectKev06ArtifactLockFromDirectory(
  model: CuratedModel,
  artifactDirectory: string,
  expectedManifestSha256: string = directKevManifestSha256
): Promise<DirectModelArtifactLock> {
  const upstreams = validateDirectKevModel(model);
  const absoluteDirectory = path.resolve(artifactDirectory);
  const relativeDirectory = path.relative(packageRoot, absoluteDirectory);
  failUnless(
    relativeDirectory !== '' && !relativeDirectory.startsWith('..') && !path.isAbsolute(relativeDirectory),
    'Direct Kev artifacts must be inside the web-decision package.'
  );
  const directoryInfo = await lstat(absoluteDirectory);
  failUnless(
    directoryInfo.isDirectory() && !directoryInfo.isSymbolicLink(),
    'Direct Kev artifact directory is invalid.'
  );

  const sourceManifestPath = path.join(absoluteDirectory, 'manifest.json');
  const manifestInfo = await lstat(sourceManifestPath);
  failUnless(manifestInfo.isFile() && !manifestInfo.isSymbolicLink(), 'Direct Kev manifest must be a regular file.');
  const manifestFile = await readJson(sourceManifestPath);
  failUnless(isRecord(manifestFile.value), 'Direct Kev manifest must be an object.');
  const canonicalBytes = canonicalManifestBytes(manifestFile.value);
  const manifest = (await readDirectKev06ManifestResponse(
    new Response(manifestFile.bytes.toString('utf8')),
    expectedManifestSha256
  )) as DirectKevManifest;
  validateDirectKevManifestMetadata(model, manifest);
  await validateAvailableDirectKevFiles(absoluteDirectory, manifest);
  const sourcePath = path.posix.join(relativeDirectory.split(path.sep).join('/'), 'manifest.json');
  return {
    schemaVersion: 1,
    manifest: {
      sourcePath,
      rawSha256: sha256(manifestFile.bytes),
      canonicalSha256: sha256(canonicalBytes),
      canonicalBytes: canonicalBytes.byteLength,
      experimental: manifest.experimental,
      files: manifest.files
    },
    bundledRuntimeFiles: [],
    upstreams
  };
}

async function readJson(filePath: string): Promise<JsonFile> {
  const bytes = await readFile(filePath);
  return {bytes, value: JSON.parse(bytes.toString('utf8'))};
}

export function validateSpec(spec: unknown): CuratedModel[] {
  failUnless(isRecord(spec), 'Catalog spec must be an object.');
  failUnless(spec.schemaVersion === 1, 'Unsupported catalog spec version.');
  failUnless(Array.isArray(spec.models) && spec.models.length > 0, 'Catalog must contain at least one curated model.');
  const modelIds = new Set<string>();
  for (const candidate of spec.models) {
    failUnless(isRecord(candidate), 'Catalog model entries must be objects.');
    const model = candidate;
    failUnless(
      typeof model.modelId === 'string' && /^[a-z0-9][a-z0-9.-]*$/u.test(model.modelId),
      'Model spec needs a valid modelId.'
    );
    failUnless(isRecord(model.description), 'Catalog model descriptions must be objects.');
    const description = model.description;
    failUnless(isRecord(description.backend), 'Catalog backends must be objects.');
    failUnless(isRecord(description.assets), 'Catalog assets must be objects.');
    validateModelPrivateInputs(model);
    failUnless(!modelIds.has(model.modelId), 'Catalog modelIds must be unique.');
    modelIds.add(model.modelId);
    failUnless(description.backend.name === 'tvm-webgpu', 'Catalog models must use TVM WebGPU.');
    failUnless(description.assets.integrity === 'pinned-manifest', 'Catalog models must pin their manifests.');
    failUnless(!Object.hasOwn(model, 'model'), 'Use modelId as the catalog identifier.');
    failUnless(!Object.hasOwn(description, 'model'), 'Model descriptions must not duplicate modelId.');
    for (const key of ['assetsUrl', 'assetUrl', 'baseUrl']) {
      failUnless(!Object.hasOwn(description, key), 'Catalog descriptions must not accept custom asset URLs.');
      failUnless(!Object.hasOwn(description.assets, key), 'Catalog assets must not accept custom asset URLs.');
    }
    for (const key of ['manifestSha256', 'wasmSha256', 'sourceDataSha256', 'requiredFiles']) {
      failUnless(!Object.hasOwn(description.assets, key), 'Artifact hashes must be generated, not curated.');
    }
    if (Object.hasOwn(description.assets, 'directArtifactRoot')) {
      const checkpoint = description.checkpoint;
      failUnless(
        model.modelId === 'kev-0.6b-q4f16' && description.adapter === 'kev06',
        'Direct artifact roots are Kev-only.'
      );
      failUnless(
        isRecord(checkpoint) &&
          checkpoint.repository === DIRECT_KEV06_CHECKPOINT.repository &&
          checkpoint.revision === DIRECT_KEV06_CHECKPOINT.revision,
        'Direct artifact roots require the direct Kev checkpoint.'
      );
      failUnless(
        isHttpsDirectory(description.assets.directArtifactRoot),
        'Direct artifact root must be an HTTPS directory.'
      );
    }
  }
  return spec.models as CuratedModel[];
}

function buildBundledFileRecords(
  manifest: KevManifest,
  inputs: readonly Readonly<{manifestName: string; sourcePath: string; bytes: Buffer}>[]
): ModelArtifactLock['bundledRuntimeFiles'] {
  return inputs.map(({manifestName, sourcePath, bytes}) => {
    const declared = manifest.files?.[manifestName];
    const digest = sha256(bytes);
    failUnless(declared, 'Manifest is missing bundled file ' + manifestName + '.');
    failUnless(declared.sha256 === digest, 'Bundled file hash differs from manifest: ' + manifestName + '.');
    failUnless(declared.bytes === bytes.byteLength, 'Bundled file size differs from manifest: ' + manifestName + '.');
    return {manifestName, sourcePath, bytes: bytes.byteLength, sha256: digest};
  });
}

function validateManifest(
  model: CuratedModel,
  manifest: KevManifest,
  manifestBytes: Buffer
): ModelArtifactLock['manifest'] {
  const digest = sha256(canonicalManifestBytes(manifest));
  failUnless(digest === KEV06_MANIFEST_SHA256, 'Canonical Kev manifest digest differs from the runtime pin.');
  failUnless(manifest.model === model.modelId, 'Kev manifest model does not match the curated modelId.');
  failUnless(
    manifest.backend === model.description.backend.name,
    'Kev manifest backend does not match the curated backend.'
  );
  failUnless(
    manifest.checkpoint?.repository === model.description.checkpoint.repository,
    'Kev repository differs from the manifest.'
  );
  failUnless(
    manifest.checkpoint?.revision === model.description.checkpoint.revision,
    'Kev revision differs from the manifest.'
  );
  failUnless(
    manifest.checkpoint?.repository === EXPERIMENTAL_KEV_UPSTREAM_REPOSITORY,
    'Kev manifest repository differs from the upstream pin.'
  );
  failUnless(
    manifest.checkpoint?.revision === EXPERIMENTAL_KEV_UPSTREAM_REVISION,
    'Kev manifest revision differs from the upstream pin.'
  );
  failUnless(
    manifest.checkpoint?.weightSha256 === EXPERIMENTAL_KEV_UPSTREAM_SOURCE_SHA256,
    'Kev source data hash differs from the upstream pin.'
  );
  failUnless(typeof manifest.experimental === 'boolean', 'Kev manifest must preserve its experimental flag.');
  return {
    sourcePath: manifestPath,
    rawSha256: sha256(manifestBytes),
    canonicalSha256: digest,
    canonicalBytes: canonicalManifestBytes(manifest).byteLength,
    experimental: manifest.experimental,
    files: manifest.files
  };
}

async function createKev06ArtifactLock(
  model: CuratedModel,
  localDirectArtifactDirectory: string = directKevArtifactDirectory
): Promise<CatalogArtifactLock> {
  if (isDirectKevCheckpoint(model)) {
    return createDirectKev06ArtifactLockFromDirectory(model, localDirectArtifactDirectory);
  }
  const [manifestFile, runtimeBytes, mapBytes] = await Promise.all([
    readJson(path.join(packageRoot, manifestPath)),
    Promise.all(bundledFiles.map(([, sourcePath]) => readFile(path.join(packageRoot, sourcePath)))),
    readFile(path.join(packageRoot, mapPath))
  ]);
  const {bytes: manifestBytes, value: rawManifest} = manifestFile;
  failUnless(isRecord(rawManifest), 'Kev manifest must be an object.');
  const manifest = rawManifest as unknown as KevManifest;
  const manifestRecord = validateManifest(model, manifest, manifestBytes);
  const bundledRuntimeFiles = buildBundledFileRecords(
    manifest,
    bundledFiles.map(([manifestName, sourcePath], index) => ({
      manifestName,
      sourcePath,
      bytes: runtimeBytes[index] as Buffer
    }))
  );
  failUnless(
    sha256(mapBytes) === EXPERIMENTAL_KEV_UPSTREAM_MAP_SHA256,
    'Upstream shard-map hash differs from its pin.'
  );
  return {
    schemaVersion: 1,
    manifest: manifestRecord,
    bundledRuntimeFiles,
    upstream: {
      repository: EXPERIMENTAL_KEV_UPSTREAM_REPOSITORY,
      revision: EXPERIMENTAL_KEV_UPSTREAM_REVISION,
      sourcePath: EXPERIMENTAL_KEV_UPSTREAM_SOURCE_PATH,
      sourceBytes: EXPERIMENTAL_KEV_UPSTREAM_SOURCE_BYTES,
      sourceSha256: EXPERIMENTAL_KEV_UPSTREAM_SOURCE_SHA256,
      shardMapPath: mapPath,
      shardMapSha256: sha256(mapBytes)
    }
  };
}

async function createKananaArtifactLock(model: CuratedModel): Promise<ModelArtifactLock> {
  const manifestFile = await readJson(path.join(packageRoot, kananaManifestPath));
  failUnless(isRecord(manifestFile.value), 'Kanana package manifest must be an object.');
  const manifest = manifestFile.value;
  failUnless(manifest.format === 'web-decision-kanana-tvm-artifacts-v1', 'Kanana package manifest format is invalid.');
  failUnless(model.modelId === KANANA_MODEL_ID, 'Kanana catalog model ID differs from the pinned model.');
  failUnless(manifest.modelId === KANANA_MODEL_ID, 'Kanana artifact model ID differs from the catalog.');
  failUnless(manifest.repository === KANANA_REPOSITORY, 'Kanana repository differs from the pinned model.');
  failUnless(manifest.revision === KANANA_REVISION, 'Kanana revision differs from the pinned model.');
  failUnless(
    manifest.checkpointSha256 === KANANA_SAFETENSORS_SPEC.fileSha256,
    'Kanana checkpoint hash differs from the pin.'
  );
  failUnless(isRecord(manifest.files), 'Kanana package manifest has no file inventory.');
  const runtime = isRecord(manifest.runtime) ? manifest.runtime : {};
  const textGraph = isRecord(runtime.textGraph) ? runtime.textGraph : {};
  const visionGraph = isRecord(runtime.visionGraph) ? runtime.visionGraph : {};
  failUnless(textGraph.mode === 'checkpoint_full', 'Kanana text graph must use the full checkpoint.');
  failUnless(textGraph.graphWeights === 354, 'Kanana text graph must contain 354 graph weights.');
  failUnless(visionGraph.graphWeights === 333, 'Kanana vision graph must contain 333 graph weights.');

  const files: Record<string, ManifestFile> = {};
  const bundledRuntimeFiles = [];
  for (const name of Object.keys(manifest.files)) {
    const declared = manifest.files[name];
    failUnless(isRecord(declared), 'Kanana package file metadata is invalid: ' + name + '.');
    const sourcePath = `assets/kanana/${name}`;
    const bytes = await readFile(path.join(packageRoot, sourcePath));
    const digest = sha256(bytes);
    failUnless(
      declared.bytes === bytes.byteLength,
      'Kanana package file size differs from its manifest: ' + name + '.'
    );
    failUnless(declared.sha256 === digest, 'Kanana package file hash differs from its manifest: ' + name + '.');
    files[name] = {bytes: bytes.byteLength, sha256: digest};
    bundledRuntimeFiles.push({manifestName: name, sourcePath, bytes: bytes.byteLength, sha256: digest});
  }
  const rangeManifestPath = 'assets/kanana/safetensors-range-manifest.json';
  const rangeManifest = await readJson(path.join(packageRoot, rangeManifestPath));
  failUnless(isRecord(rangeManifest.value), 'Kanana range manifest must be an object.');
  failUnless(
    rangeManifest.value.model === KANANA_MODEL_ID &&
      rangeManifest.value.repository === KANANA_REPOSITORY &&
      rangeManifest.value.revision === KANANA_REVISION &&
      rangeManifest.value.path === KANANA_SAFETENSORS_SPEC.path &&
      rangeManifest.value.totalBytes === KANANA_SAFETENSORS_SPEC.totalBytes &&
      rangeManifest.value.headerSha256 === KANANA_SAFETENSORS_SPEC.headerSha256 &&
      rangeManifest.value.tensorCount === KANANA_SAFETENSORS_SPEC.tensorCount,
    'Kanana range manifest differs from the pinned checkpoint.'
  );
  const canonicalBytes = canonicalManifestBytes(manifest);
  const rangeManifestRecord = files['safetensors-range-manifest.json'];
  failUnless(rangeManifestRecord, 'Kanana range manifest is missing from the package file inventory.');

  return {
    schemaVersion: 1,
    manifest: {
      sourcePath: kananaManifestPath,
      rawSha256: sha256(manifestFile.bytes),
      canonicalSha256: sha256(canonicalBytes),
      canonicalBytes: canonicalBytes.byteLength,
      experimental: false,
      files
    },
    bundledRuntimeFiles,
    upstream: {
      repository: KANANA_REPOSITORY,
      revision: KANANA_REVISION,
      sourcePath: KANANA_SAFETENSORS_SPEC.path,
      sourceBytes: KANANA_SAFETENSORS_SPEC.totalBytes,
      sourceSha256: KANANA_SAFETENSORS_SPEC.fileSha256,
      rangeManifestPath,
      rangeManifestSha256: rangeManifestRecord.sha256
    }
  };
}

export async function createModelArtifactLock(
  model: CuratedModel,
  options: Readonly<{directKevArtifactDirectory?: string}> = {}
): Promise<CatalogArtifactLock> {
  if (model.description.adapter === 'kev06')
    return createKev06ArtifactLock(model, options.directKevArtifactDirectory ?? directKevArtifactDirectory);
  if (model.description.adapter === 'kanana') return createKananaArtifactLock(model);
  throw new Error('No artifact recipe is registered for adapter ' + model.description.adapter + '.');
}

async function createArtifactLock(
  models: readonly CuratedModel[],
  options: CatalogGenerationOptions
): Promise<ArtifactLockFile> {
  const records = await Promise.all(
    models.map(async (model): Promise<readonly [ModelId, CatalogArtifactLock]> => [
      model.modelId,
      await createModelArtifactLock(model, options)
    ])
  );
  return {schemaVersion: 1, byModelId: Object.fromEntries(records)};
}

function readDirectKevArtifactDirectory(args: readonly string[]): string {
  const option = '--kev-direct-artifact-dir';
  const index = args.indexOf(option);
  if (index < 0) return directKevArtifactDirectory;
  failUnless(args.indexOf(option, index + 1) < 0, 'Direct Kev artifact directory option must appear once.');
  const value = args[index + 1];
  failUnless(
    typeof value === 'string' && value.length > 0 && !value.startsWith('--'),
    'Direct Kev artifact directory is missing.'
  );
  failUnless(!value.includes('://'), 'Direct Kev artifact directory must be a local filesystem path.');
  return path.resolve(packageRoot, value);
}

function renderCatalogModule(): string {
  return `import modelSpec from './model-spec.json' with {type: 'json'};
import artifactLock from './artifacts.lock.json' with {type: 'json'};

import type {
  CatalogModelDescription,
  ModelArtifactLock,
  ModelCatalogEntry,
  ModelId,
  ModelIdTuple
} from './types.generated.js';

type CuratedModelSpec = Readonly<{
  modelId: ModelId;
  description: Omit<CatalogModelDescription, 'assets'> & {
    assets: Omit<CatalogModelDescription['assets'], 'manifestSha256' | 'requiredFiles'>;
  };
}>;
const models = modelSpec.models as unknown as readonly CuratedModelSpec[];
const artifactsByModelId = artifactLock.byModelId as unknown as Readonly<Record<ModelId, ModelArtifactLock>>;

function deepFreeze<T>(value: T): Readonly<T> {
  if (!value || typeof value !== 'object' || Object.isFrozen(value)) return value;
  for (const child of Object.values(value)) deepFreeze(child);
  return Object.freeze(value);
}

const catalog: ModelCatalogEntry[] = models.map((model) => {
  const artifacts = artifactsByModelId[model.modelId];
  return {
    modelId: model.modelId,
    description: {
      ...model.description,
      assets: {
        ...model.description.assets,
        requiredFiles: ['manifest.json', ...Object.keys(artifacts.manifest.files)],
        manifestSha256: artifacts.manifest.canonicalSha256
      }
    },
    artifacts
  };
});

export const MODEL_CATALOG = deepFreeze(catalog);
export const MODEL_IDS = Object.freeze(MODEL_CATALOG.map((entry) => entry.modelId)) as ModelIdTuple;
export const MODEL_CATALOG_BY_ID = Object.freeze(
  Object.fromEntries(MODEL_CATALOG.map((entry) => [entry.modelId, entry])) as Record<ModelId, ModelCatalogEntry>
);
`;
}

function renderTypeFiles(models: readonly CuratedModel[]): {types: string; entry: string} {
  const ids = models.map((model) => "'" + model.modelId + "'");
  const idTypes = ids.join(' | ');
  const adapterTypes = [...new Set(models.map((model) => "'" + model.description.adapter + "'"))].join(' | ');
  const tupleTypes = ids.join(', ');
  const typeSource =
    [
      'export type ModelId = ' + idTypes + ';',
      'export type ModelIdTuple = readonly [' + tupleTypes + '];',
      'export type ModelAdapterId = ' + adapterTypes + ';',
      'export interface CatalogModelLimit {',
      "  readonly resource: 'tokens' | 'options' | 'questions' | 'images';",
      "  readonly scope: 'state' | 'question' | 'request';",
      '  readonly min?: number;',
      '  readonly max?: number;',
      '  readonly description?: string;',
      '}',
      'export interface CatalogModelDescription {',
      '  readonly adapter: ModelAdapterId;',
      '  readonly precision: string;',
      '  readonly checkpoint: Readonly<{repository: string; revision: string; weightSha256?: string}>;',
      '  readonly assets: Readonly<{',
      "    integrity: 'pinned-manifest';",
      '    manifestSha256: string;',
      '    location: string;',
      '    directArtifactRoot?: string;',
      '    requiredFiles: readonly string[];',
      '    additionalFiles: string;',
      '  }>;',
      '  readonly licenses: Readonly<Record<string, string>>;',
      '  readonly backend: Readonly<{',
      "    name: 'tvm-webgpu';",
      '    requiredFeatures: readonly string[];',
      '    requiredLimits: Readonly<Record<string, number>>;',
      '    requiresNonFallbackAdapter: boolean;',
      '  }>;',
      '  readonly features: Readonly<{',
      '    textChoice: Readonly<{',
      "      status: 'implemented';",
      '      abstention: string;',
      '      limits: readonly CatalogModelLimit[];',
      '      handling: string;',
      '      tokenCount: string;',
      '      score: Readonly<Record<string, unknown>>;',
      '    }>;',
      '    typedTextQuestionTypes?: readonly string[];',
      "    imageChoice?: Readonly<{status: 'implemented' | 'not-implemented'; limits?: readonly CatalogModelLimit[]}>;",
      "    rating?: Readonly<{status: 'not-implemented'}>;",
      '  }>;',
      '  readonly verification: Readonly<Record<string, string>>;',
      '}',
      'export interface ModelArtifactLock {',
      '  readonly schemaVersion: 1;',
      '  readonly manifest: Readonly<{',
      '    sourcePath: string;',
      '    rawSha256: string;',
      '    canonicalSha256: string;',
      '    canonicalBytes: number;',
      '    experimental: boolean;',
      '    files: Readonly<Record<string, Readonly<{sha256: string; bytes: number}>>>;',
      '  }>;',
      '  readonly bundledRuntimeFiles: readonly Readonly<{',
      '    manifestName: string;',
      '    sourcePath: string;',
      '    bytes: number;',
      '    sha256: string;',
      '  }>[];',
      '  readonly upstream?: Readonly<{',
      '    repository: string;',
      '    revision: string;',
      '    sourcePath: string;',
      '    sourceBytes: number;',
      '    sourceSha256: string;',
      '    shardMapPath?: string;',
      '    shardMapSha256?: string;',
      '    rangeManifestPath?: string;',
      '    rangeManifestSha256?: string;',
      '  }>;',
      '  readonly upstreams?: readonly Readonly<{',
      '    role: string;',
      '    repository: string;',
      '    revision: string;',
      '    sourcePath: string;',
      '    sourceBytes: number;',
      '    sourceSha256: string;',
      '  }>[];',
      '}',
      'export interface ModelCatalogEntry {',
      '  readonly modelId: ModelId;',
      '  readonly description: CatalogModelDescription;',
      '  readonly artifacts: ModelArtifactLock;',
      '}',
      'export declare const MODEL_IDS: ModelIdTuple;',
      'export declare const MODEL_CATALOG: readonly ModelCatalogEntry[];',
      'export declare const MODEL_CATALOG_BY_ID: Readonly<Record<ModelId, ModelCatalogEntry>>;'
    ].join('\n') + '\n';
  return {
    types: typeSource,
    entry: "export * from './types.generated.js';\n"
  };
}

function renderJson(value: unknown): string {
  return JSON.stringify(value, null, 2) + '\n';
}

async function writeOutputs(outputs: readonly (readonly [string, string])[], checkOnly: boolean): Promise<void> {
  for (const [filePath, expected] of outputs) {
    if (!checkOnly) {
      await writeFile(filePath, expected);
      continue;
    }
    let actual;
    try {
      actual = await readFile(filePath, 'utf8');
    } catch {
      actual = '';
    }
    failUnless(actual === expected, 'Generated file is stale: ' + path.relative(repositoryRoot, filePath));
  }
}

async function main(): Promise<void> {
  const {value: rawSpec} = await readJson(specPath);
  const models = validateSpec(rawSpec);
  const artifactLock = await createArtifactLock(models, {
    directKevArtifactDirectory: readDirectKevArtifactDirectory(process.argv.slice(2))
  });
  const types = renderTypeFiles(models);
  const outputs: readonly (readonly [string, string])[] = [
    [path.join(catalogRoot, 'artifacts.lock.json'), renderJson(artifactLock)],
    [path.join(catalogRoot, 'index.generated.ts'), renderCatalogModule()],
    [path.join(catalogRoot, 'types.generated.d.ts'), types.types],
    [path.join(catalogRoot, 'index.generated.d.ts'), types.entry]
  ];
  await writeOutputs(outputs, process.argv.includes('--check'));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
