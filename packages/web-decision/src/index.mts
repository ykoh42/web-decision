import {Tokenizer} from '@huggingface/tokenizers';
import {instantiate, createPolyfillWASI} from './runtime/tvm.generated.mjs';
import {CACHE_NAME} from './assets.js';
import {WebDecisionError} from './kev.js';
import {loadExperimentalDeberta} from './deberta/loader.mjs';
import {loadExperimentalStrandsFromSource} from './strands/loader.mjs';
import {loadExperimentalStrandsGemma4FromSource} from './strands/gemma4-e2b-loader.mjs';
import {selectModelRegistration} from './model-family.js';
import {describeStableModel, normalizeStableModelOptions} from './stable-models.js';
import {loadKevDecisionModel} from './kev-loader.js';
import {loadExperimentalMapika} from './mapika/loader.js';
import {loadExperimentalMapikaQ4} from './mapika-q4/loader.js';
import {loadExperimentalKev08} from './kev08/loader.js';
import {loadH2oCandidate} from './h2o/loader.mjs';
import {loadNariCandidate} from './nari/loader.mjs';
import {loadImajevCandidate} from './imajev/candidate-loader.mjs';
import {loadFlashCandidate} from './pplx/loader.mjs';
import {loadMapika4Candidate} from './mapika4b/loader.mjs';
import {loadElephantCandidate} from './elephant/loader.mjs';
import {loadClefCandidate} from './clef/loader.mjs';
import {loadOpenDeciderNano} from './opendecider-nano/loader.mjs';
import {loadTinyDecide} from './tinydecide/loader.mjs';

export {WebDecisionError};
export {describeModel, isDecisionModel, listDecisionModels} from './model-registry.js';
export {describeStableModel, isStableModelId, listStableModelIds} from './stable-models.js';

function loadRegisteredModel(
  family: string,
  options: Record<string, unknown>,
  description: ReturnType<typeof selectModelRegistration>['description'],
  stableModelId: string | undefined
) {
  const runtimeBindings = {Tokenizer, instantiate, createPolyfillWASI};
  // Each legacy loader validates its own options and runtime bindings.
  const loaders: Record<string, () => Promise<unknown>> = {
    kev: () => loadKevDecisionModel(options, description, stableModelId),
    deberta: () => loadExperimentalDeberta(options),
    mapika: () => loadExperimentalMapika(options),
    'mapika-q4': () => loadExperimentalMapikaQ4(options as unknown as Parameters<typeof loadExperimentalMapikaQ4>[0]),
    kev08: () => loadExperimentalKev08(options),
    h2o: () => loadH2oCandidate(options, {expectedModel: description.model, runtimeBindings}),
    nari: () => loadNariCandidate(options, {...runtimeBindings, expectedModel: description.model}),
    imajev: () => loadImajevCandidate(options, runtimeBindings as unknown as Parameters<typeof loadImajevCandidate>[1]),
    pplx: () => loadFlashCandidate(options, runtimeBindings as unknown as Parameters<typeof loadFlashCandidate>[1]),
    mapika4b: () => loadMapika4Candidate(options, runtimeBindings),
    elephant: () => loadElephantCandidate(options, runtimeBindings),
    clef: () => loadClefCandidate(options, runtimeBindings as unknown as Parameters<typeof loadClefCandidate>[1]),
    'opendecider-nano': () => loadOpenDeciderNano(options, runtimeBindings),
    tinydecide: () => loadTinyDecide(options),
    strands: () =>
      loadExperimentalStrandsFromSource({
        assetsUrl: options.assetsUrl,
        cache: options.cache,
        offline: options.offline,
        signal: options.signal,
        onProgress: options.onProgress,
        maxPrefetchBytes: options.maxPrefetchBytes,
        createTokenizer: (data: unknown, config: unknown) => new Tokenizer(data as object, config as object),
        createRuntime: (wasm: unknown) => instantiate(wasm as ArrayBuffer, createPolyfillWASI()),
        gpuApi: globalThis.navigator?.gpu
      } as unknown as Parameters<typeof loadExperimentalStrandsFromSource>[0]),
    'strands-gemma4': () =>
      loadExperimentalStrandsGemma4FromSource({
        assetsUrl: options.assetsUrl,
        cache: options.cache,
        offline: options.offline,
        signal: options.signal,
        onProgress: options.onProgress,
        maxPrefetchBytes: options.maxPrefetchBytes,
        createTokenizer: (data: unknown, config: unknown) => new Tokenizer(data as object, config as object),
        createRuntime: (wasm: unknown) => instantiate(wasm as ArrayBuffer, createPolyfillWASI()),
        gpuApi: globalThis.navigator?.gpu
      } as unknown as Parameters<typeof loadExperimentalStrandsGemma4FromSource>[0])
  };
  const load = loaders[family];
  if (!load) throw new WebDecisionError('INVALID_ASSET', `No loader is registered for ${family}.`);
  return load();
}

/** Load a registered decision model, or a stable public ID with model-specific options. */
export async function loadDecisionModel(options: unknown) {
  if (typeof options === 'string')
    throw new WebDecisionError('INVALID_INPUT', 'Pass a stable ID as the modelId property.');
  const stableRequest =
    options && typeof options === 'object' && 'modelId' in options ? normalizeStableModelOptions(options) : undefined;
  if (stableRequest?.modelId === 'kev-0.6b-q4f16' && Object.hasOwn(stableRequest.options, 'assetsUrl'))
    throw new WebDecisionError('INVALID_INPUT', 'assetsUrl is not supported for the stable Kev model ID.');
  const loadOptions = (stableRequest?.options ?? options) as Record<string, unknown>;
  const {family, description} = selectModelRegistration(loadOptions);
  const debertaReadback = loadOptions && Object.getOwnPropertyDescriptor(loadOptions, 'debertaReadback');
  if (
    family !== 'deberta' &&
    debertaReadback &&
    (!Object.hasOwn(debertaReadback, 'value') || debertaReadback.value !== undefined)
  )
    throw new WebDecisionError('INVALID_INPUT', 'debertaReadback is only supported by the DeBERTa model.');
  const model = (await loadRegisteredModel(family, loadOptions, description, stableRequest?.modelId)) as object & {
    dispose?: () => unknown;
  };
  try {
    // Keep route IDs internal to dispatch and expose the stable ID to callers.
    Object.defineProperty(model, 'description', {
      value: stableRequest ? describeStableModel(stableRequest.modelId) : description
    });
    return model;
  } catch (error) {
    await Promise.allSettled([Promise.resolve().then(() => model.dispose?.())]);
    throw error;
  }
}

/** Delete only model assets cached by this library for the current origin. */
export async function clearModelCache() {
  if (!globalThis.caches) return false;
  return caches.delete(CACHE_NAME);
}
