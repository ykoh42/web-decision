import type {AssetCache} from '../../src/assets.js';
import {CACHE_NAME as ASSET_CACHE_NAME} from '../../src/assets.js';
import {MODEL_CATALOG_BY_ID} from '../../src/catalog/index.generated.ts';
import {WebDecisionError} from '../../src/curated/errors.ts';
import {readLoadOptions} from '../../src/curated/load-options.ts';
import type {ModelDescription, ModelId} from '../../src/curated/types.js';
import type {StrandsQwen35LoadedCandidateModel} from '../../src/curated/strands-qwen35-candidate-types.js';

const QWEN35_MODEL_ID = 'strands-decider-2b-qwen3.5-v1-2610';
type Qwen35Adapter = typeof import('../../src/curated/adapters/strands-qwen35.ts');
type Qwen35AdapterDependencies = NonNullable<Parameters<Qwen35Adapter['loadModel']>[2]>;
const qwen35Entry = MODEL_CATALOG_BY_ID[QWEN35_MODEL_ID];
const {adapter: _adapter, ...loaderMetadata} = qwen35Entry.description;
const loaderDescription = Object.freeze({modelId: QWEN35_MODEL_ID, ...loaderMetadata});
const {directArtifactRoot: _directArtifactRoot, ...publicAssets} = loaderDescription.assets;
const qwen35Description = Object.freeze({...loaderDescription, assets: Object.freeze(publicAssets)});
const candidateDescriptions = Object.freeze([qwen35Description]);

function isQwen35ModelId(value: unknown): value is ModelId {
  return value === QWEN35_MODEL_ID;
}

export {CACHE_NAME, openAssetCache} from '../../src/assets.ts';
export {KananaFloat, kananaFloat} from '../../src/curated/index.ts';

export async function loadModel(
  options: object,
  description: ModelDescription,
  dependencies?: Qwen35AdapterDependencies
): Promise<StrandsQwen35LoadedCandidateModel> {
  const adapter = await import('../../src/curated/adapters/strands-qwen35.ts');
  return adapter.loadModel(options, description, dependencies);
}

/** Match the package cache clearing API in the unpublished candidate entry. */
export async function clearModelCache(): Promise<boolean> {
  if (!globalThis.caches) return false;
  return globalThis.caches.delete(ASSET_CACHE_NAME);
}

/** Test-only candidate API. Never used by the package's default curated entry. */
export function listModels(): readonly ModelDescription[] {
  return candidateDescriptions;
}

/** Describe the pending Qwen candidate inside an unpublished test archive. */
export function describeModel(modelId: ModelId): ModelDescription {
  if (!isQwen35ModelId(modelId))
    throw new WebDecisionError('INVALID_INPUT', 'Unsupported decision model ID.', undefined);
  return qwen35Description;
}

/** Load Qwen through its real pinned adapter; optional dependencies are test-only injection. */
export async function loadDecisionModel(
  input: unknown,
  dependencies?: Qwen35AdapterDependencies
): Promise<StrandsQwen35LoadedCandidateModel> {
  const {modelId, options} = readLoadOptions(input, isQwen35ModelId);
  if (modelId !== QWEN35_MODEL_ID)
    throw new WebDecisionError('INVALID_INPUT', 'Unsupported decision model ID.', undefined);
  const model = await loadModel(options, loaderDescription, dependencies);
  try {
    Object.defineProperty(model, 'description', {value: qwen35Description});
    return model;
  } catch (error) {
    await Promise.allSettled([Promise.resolve().then(() => model.dispose())]);
    throw error;
  }
}

export async function openQwen35PackageArtifacts(options: {cache?: AssetCache | null; offline?: boolean}) {
  const [r19Module, visionModule] = await Promise.all([
    import('../../src/strands/qwen35-r19-artifacts.mts'),
    import('../../src/strands/qwen35-vision-artifacts.mts')
  ]);
  const [r19, vision] = await Promise.all([
    r19Module.openPinnedQwen35R19Artifacts(options),
    visionModule.openPinnedQwen35VisionArtifacts(options)
  ]);
  return {r19, vision};
}
