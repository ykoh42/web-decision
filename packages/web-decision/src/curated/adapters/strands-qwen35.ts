import {openAssetCache} from '../../assets.js';
import {WebDecisionError} from '../errors.ts';
import type {
  StrandsQwen35CandidateLoadOptions,
  StrandsQwen35LoadedCandidateModel
} from '../strands-qwen35-candidate-types.js';
import type {ModelDescription} from '../types.js';
import {loadPinnedQwen35Model} from '../../strands/qwen35-model-loader.mts';

const MODEL_ID = 'strands-decider-2b-qwen3.5-v1-2610';

type Qwen35Session = Awaited<ReturnType<typeof loadPinnedQwen35Model>>;
type Qwen35AdapterDependencies = Readonly<{
  openCache: typeof openAssetCache;
  loadPinnedModel: (options: Parameters<typeof loadPinnedQwen35Model>[0]) => Promise<Qwen35Session>;
}>;

function loadOptions(value: object): StrandsQwen35CandidateLoadOptions {
  const options = value as Record<string, unknown>;
  if (Object.hasOwn(options, 'assetsUrl'))
    throw new WebDecisionError('INVALID_INPUT', 'assetsUrl is not supported for the pinned Qwen3.5 model.');
  if (options.cache !== undefined && typeof options.cache !== 'boolean')
    throw new WebDecisionError('INVALID_INPUT', 'cache must be a boolean.');
  if (options.offline !== undefined && typeof options.offline !== 'boolean')
    throw new WebDecisionError('INVALID_INPUT', 'offline must be a boolean.');
  if (options.offline && options.cache === false)
    throw new WebDecisionError('INVALID_INPUT', 'offline requires the model cache.');
  if (options.onProgress !== undefined && typeof options.onProgress !== 'function')
    throw new WebDecisionError('INVALID_INPUT', 'onProgress must be a function.');
  return options as StrandsQwen35CandidateLoadOptions;
}

/** Curated adapter implementation held behind the pending provenance approval gate. */
export async function loadModel(
  rawOptions: object,
  description: ModelDescription,
  dependencies: Qwen35AdapterDependencies = {openCache: openAssetCache, loadPinnedModel: loadPinnedQwen35Model}
): Promise<StrandsQwen35LoadedCandidateModel> {
  if (description.modelId !== MODEL_ID)
    throw new WebDecisionError('INVALID_ASSET', 'Qwen3.5 adapter received a different model descriptor.');
  const options = loadOptions(rawOptions);
  const cache = await dependencies.openCache(options.cache !== false, options.signal);
  const session = await dependencies.loadPinnedModel({
    cache,
    offline: options.offline ?? false,
    ...(options.signal ? {signal: options.signal} : {}),
    ...(options.onProgress ? {onProgress: options.onProgress} : {})
  });
  let disposal: Promise<void> | undefined;
  try {
    return {
      modelId: MODEL_ID,
      description,
      systemOne: session.systemOne.bind(session),
      releaseCachedMemory: session.releaseCachedMemory.bind(session),
      dispose() {
        if (!disposal) disposal = Promise.resolve().then(() => session.dispose());
        return disposal;
      }
    };
  } catch (error) {
    await Promise.allSettled([Promise.resolve().then(() => session.dispose())]);
    throw error;
  }
}
