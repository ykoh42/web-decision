import {createHash} from 'node:crypto';
import {readFile, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {KEV06_MANIFEST_SHA256} from '../src/kev06/contract.js';
import {
  EXPERIMENTAL_KEV_UPSTREAM_MAP_SHA256,
  EXPERIMENTAL_KEV_UPSTREAM_REPOSITORY,
  EXPERIMENTAL_KEV_UPSTREAM_REVISION,
  EXPERIMENTAL_KEV_UPSTREAM_SOURCE_BYTES,
  EXPERIMENTAL_KEV_UPSTREAM_SOURCE_PATH,
  EXPERIMENTAL_KEV_UPSTREAM_SOURCE_SHA256
} from '../src/kev06/upstream-contract.js';

const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repositoryRoot = path.resolve(packageRoot, '../..');
const catalogRoot = path.join(packageRoot, 'src/catalog');
const specPath = path.join(catalogRoot, 'model-spec.json');
const manifestPath = 'assets/kev06-stable/manifest.json';
const mapPath = 'assets/experimental-kev-upstream/source-shard-map.json';
const bundledFiles = [
  ['model.wasm', 'assets/kev06-stable/model.wasm'],
  ['tokenizer.json', 'assets/experimental-kev-upstream/tokenizer.json'],
  ['tokenizer_config.json', 'assets/experimental-kev-upstream/tokenizer_config.json'],
  ['model_config.json', 'assets/experimental-kev-upstream/model_config.json'],
  ['weights/tensor-cache.json', 'assets/experimental-kev-upstream/weights/tensor-cache.json']
];

function failUnless(condition, message) {
  if (!condition) throw new Error(message);
}

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(
    Object.keys(value)
      .sort()
      .map((key) => [key, canonical(value[key])])
  );
}

function canonicalManifestBytes(value) {
  return Buffer.from(JSON.stringify(canonical(value)) + '\n');
}

async function readJson(filePath) {
  const bytes = await readFile(filePath);
  return {bytes, value: JSON.parse(bytes.toString('utf8'))};
}

export function validateSpec(spec) {
  failUnless(spec.schemaVersion === 1, 'Unsupported catalog spec version.');
  failUnless(Array.isArray(spec.models) && spec.models.length > 0, 'Catalog must contain at least one curated model.');
  const modelIds = new Set();
  for (const model of spec.models) {
    failUnless(model && typeof model === 'object', 'Catalog model entries must be objects.');
    failUnless(
      typeof model.modelId === 'string' && /^[a-z0-9][a-z0-9.-]*$/u.test(model.modelId),
      'Model spec needs a valid modelId.'
    );
    failUnless(!modelIds.has(model.modelId), 'Catalog modelIds must be unique.');
    modelIds.add(model.modelId);
    failUnless(model.description?.backend?.name === 'tvm-webgpu', 'Catalog models must use TVM WebGPU.');
    failUnless(model.description?.assets?.integrity === 'pinned-manifest', 'Catalog models must pin their manifests.');
    failUnless(!Object.hasOwn(model, 'model'), 'Use modelId as the catalog identifier.');
    failUnless(!Object.hasOwn(model.description, 'model'), 'Model descriptions must not duplicate modelId.');
    for (const key of ['assetsUrl', 'assetUrl', 'baseUrl']) {
      failUnless(!Object.hasOwn(model.description, key), 'Catalog descriptions must not accept custom asset URLs.');
      failUnless(!Object.hasOwn(model.description.assets, key), 'Catalog assets must not accept custom asset URLs.');
    }
    for (const key of ['manifestSha256', 'wasmSha256', 'sourceDataSha256']) {
      failUnless(!Object.hasOwn(model.description.assets, key), 'Artifact hashes must be generated, not curated.');
    }
  }
  return spec.models;
}

function buildBundledFileRecords(manifest, inputs) {
  return inputs.map(({manifestName, sourcePath, bytes}) => {
    const declared = manifest.files?.[manifestName];
    const digest = sha256(bytes);
    failUnless(declared, 'Manifest is missing bundled file ' + manifestName + '.');
    failUnless(declared.sha256 === digest, 'Bundled file hash differs from manifest: ' + manifestName + '.');
    failUnless(declared.bytes === bytes.byteLength, 'Bundled file size differs from manifest: ' + manifestName + '.');
    return {manifestName, sourcePath, bytes: bytes.byteLength, sha256: digest};
  });
}

function validateManifest(model, manifest, manifestBytes) {
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

async function createKev06ArtifactLock(model) {
  const inputs = await Promise.all([
    readJson(path.join(packageRoot, manifestPath)),
    ...bundledFiles.map(([, sourcePath]) => readFile(path.join(packageRoot, sourcePath))),
    readFile(path.join(packageRoot, mapPath))
  ]);
  const {bytes: manifestBytes, value: manifest} = inputs[0];
  const runtimeBytes = inputs.slice(1, 1 + bundledFiles.length);
  const mapBytes = inputs[1 + bundledFiles.length];
  const manifestRecord = validateManifest(model, manifest, manifestBytes);
  const bundledRuntimeFiles = buildBundledFileRecords(
    manifest,
    bundledFiles.map(([manifestName, sourcePath], index) => ({
      manifestName,
      sourcePath,
      bytes: runtimeBytes[index]
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

export async function createModelArtifactLock(model) {
  if (model.description.adapter === 'kev06') return createKev06ArtifactLock(model);
  throw new Error('No artifact recipe is registered for adapter ' + model.description.adapter + '.');
}

async function createArtifactLock(models) {
  const records = await Promise.all(models.map(async (model) => [model.modelId, await createModelArtifactLock(model)]));
  return {schemaVersion: 1, byModelId: Object.fromEntries(records)};
}

function renderCatalogModule() {
  return (
    "import modelSpec from './model-spec.json' with {type: 'json'};\n" +
    "import artifactLock from './artifacts.lock.json' with {type: 'json'};\n\n" +
    'function deepFreeze(value) {\n' +
    "  if (!value || typeof value !== 'object' || Object.isFrozen(value)) return value;\n" +
    '  for (const child of Object.values(value)) deepFreeze(child);\n' +
    '  return Object.freeze(value);\n' +
    '}\n\n' +
    'export const MODEL_CATALOG = deepFreeze(\n' +
    '  modelSpec.models.map((model) => {\n' +
    '    const artifacts = artifactLock.byModelId[model.modelId];\n' +
    '    return {\n' +
    '      modelId: model.modelId,\n' +
    '      description: {\n' +
    '        ...model.description,\n' +
    '        assets: {...model.description.assets, manifestSha256: artifacts.manifest.canonicalSha256}\n' +
    '      },\n' +
    '      artifacts\n' +
    '    };\n' +
    '  })\n' +
    ');\n' +
    'export const MODEL_IDS = Object.freeze(MODEL_CATALOG.map((entry) => entry.modelId));\n' +
    'export const MODEL_CATALOG_BY_ID = Object.freeze(\n' +
    '  Object.fromEntries(MODEL_CATALOG.map((entry) => [entry.modelId, entry]))\n' +
    ');\n'
  );
}

function renderTypeFiles(models) {
  const ids = models.map((model) => "'" + model.modelId + "'");
  const idTypes = ids.join(' | ');
  const adapterTypes = [...new Set(models.map((model) => "'" + model.description.adapter + "'"))].join(' | ');
  const tupleTypes = ids.join(', ');
  const typeSource =
    [
      'export type ModelId = ' + idTypes + ';',
      'export type ModelAdapterId = ' + adapterTypes + ';',
      'export interface CatalogModelLimit {',
      "  readonly resource: 'tokens' | 'options' | 'questions';",
      "  readonly scope: 'state' | 'question' | 'request';",
      '  readonly min?: number;',
      '  readonly max?: number;',
      '  readonly description?: string;',
      '}',
      'export interface CatalogModelDescription {',
      '  readonly adapter: ModelAdapterId;',
      '  readonly precision: string;',
      '  readonly checkpoint: Readonly<Record<string, string>>;',
      '  readonly assets: Readonly<{',
      "    integrity: 'pinned-manifest';",
      '    manifestSha256: string;',
      '    location: string;',
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
      "    imageChoice?: Readonly<{status: 'not-implemented'}>;",
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
      '  readonly upstream: Readonly<{',
      '    repository: string;',
      '    revision: string;',
      '    sourcePath: string;',
      '    sourceBytes: number;',
      '    sourceSha256: string;',
      '    shardMapPath: string;',
      '    shardMapSha256: string;',
      '  }>;',
      '}',
      'export interface ModelCatalogEntry {',
      '  readonly modelId: ModelId;',
      '  readonly description: CatalogModelDescription;',
      '  readonly artifacts: ModelArtifactLock;',
      '}',
      'export declare const MODEL_IDS: readonly [' + tupleTypes + '];',
      'export declare const MODEL_CATALOG: readonly ModelCatalogEntry[];',
      'export declare const MODEL_CATALOG_BY_ID: Readonly<Record<ModelId, ModelCatalogEntry>>;'
    ].join('\n') + '\n';
  return {
    types: typeSource,
    entry: "export * from './types.generated.js';\n"
  };
}

function renderJson(value) {
  return JSON.stringify(value, null, 2) + '\n';
}

async function writeOutputs(outputs, checkOnly) {
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

async function main() {
  const {value: rawSpec} = await readJson(specPath);
  const models = validateSpec(rawSpec);
  const artifactLock = await createArtifactLock(models);
  const types = renderTypeFiles(rawSpec.models);
  const outputs = [
    [path.join(catalogRoot, 'artifacts.lock.json'), renderJson(artifactLock)],
    [path.join(catalogRoot, 'index.generated.js'), renderCatalogModule()],
    [path.join(catalogRoot, 'types.generated.d.ts'), types.types],
    [path.join(catalogRoot, 'index.generated.d.ts'), types.entry]
  ];
  await writeOutputs(outputs, process.argv.includes('--check'));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
