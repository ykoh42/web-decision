import {openAssetCache} from '../../assets.js';
import {loadMapika2bCandidate} from '../../mapika2b/loader.mts';
import {openPackagedMapika2bText} from '../../mapika2b/packaged.mts';
import type {Mapika2bLimits, Mapika2bTemperatureOptions} from '../../mapika2b/contract.mts';
import {WebDecisionError} from '../errors.ts';
import type {ModelDescription, ModelProgress} from '../types.js';

const MODEL_ID = 'mapika-decider-2b-v12';

interface MapikaOptions extends Mapika2bTemperatureOptions {
  readonly cache?: boolean;
  readonly offline?: boolean;
  readonly signal?: AbortSignal;
  readonly onProgress?: (progress: ModelProgress) => void;
  readonly limits?: Partial<Mapika2bLimits>;
}

/** Load the pinned Mapika graph and checkpoint through the curated model interface. */
export async function loadModel(rawOptions: object, description: ModelDescription) {
  if (description.modelId !== MODEL_ID)
    throw new WebDecisionError('INVALID_ASSET', 'Mapika adapter received a different model descriptor.');
  const options = rawOptions as MapikaOptions;
  if (options.cache !== undefined && typeof options.cache !== 'boolean')
    throw new WebDecisionError('INVALID_INPUT', 'cache must be a boolean.');
  if (options.offline !== undefined && typeof options.offline !== 'boolean')
    throw new WebDecisionError('INVALID_INPUT', 'offline must be a boolean.');
  if (options.offline && options.cache === false)
    throw new WebDecisionError('INVALID_INPUT', 'offline requires the model cache.');
  if (options.onProgress !== undefined && typeof options.onProgress !== 'function')
    throw new WebDecisionError('INVALID_INPUT', 'onProgress must be a function.');
  const cache = await openAssetCache(options.cache !== false, options.signal);
  const assetOptions = {cache, offline: options.offline ?? false, signal: options.signal};
  const artifacts = await openPackagedMapika2bText(assetOptions);
  const session = await loadMapika2bCandidate({
    ...assetOptions,
    artifacts,
    ...(options.onProgress ? {onProgress: options.onProgress} : {}),
    ...(options.limits === undefined ? {} : {limits: options.limits}),
    ...(options.temperature === undefined ? {} : {temperature: options.temperature}),
    ...(options.temperatureByType === undefined ? {} : {temperatureByType: options.temperatureByType})
  });
  return {modelId: MODEL_ID, description, systemOne: session.systemOne, dispose: session.dispose};
}
