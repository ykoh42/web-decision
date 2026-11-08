import {Tokenizer} from '@huggingface/tokenizers';
import {instantiate} from '../../../.toolchains/apache-tvm-src/web/src/runtime.ts';
import {createPolyfillWASI} from '../../../.toolchains/apache-tvm-src/web/src/compact.ts';
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
import {loadDecimaModel} from './decima/loader.mjs';
import {loadTinyDecide} from './tinydecide/loader.mjs';
import {loadTamev} from './tamev/loader.mjs';
import {loadSkejja} from './skejja/loader.mjs';
import {loadEttin} from './ettin/loader.mjs';
import {loadLayaModel} from './laya/loader.mjs';
import {loadLayaTypedDecisionsModel} from './laya-typed/loader.mjs';
import {loadXDecisionModel} from './xdecision/loader.mjs';
import {loadGlinerMobileModel} from './gliner-mobile/loader.mjs';
import {loadD1OmniModel} from './d1-omni/loader.mjs';
import {loadVerdictModel} from './verdict/loader.mjs';
import {loadSafeDecideModel} from './safedecide/loader.mjs';
import {loadCrivoModel} from './crivo/loader.mjs';

export {WebDecisionError};
export {describeModel, isDecisionModel, listDecisionModels} from './model-registry.js';
export {describeStableModel, isStableModelId, listStableModelIds} from './stable-models.js';

function loadRegisteredModel(family, options, description) {
  const runtimeBindings = {Tokenizer, instantiate, createPolyfillWASI};
  const loaders = {
    kev: () => loadKevDecisionModel(options, description),
    deberta: () => loadExperimentalDeberta(options),
    mapika: () => loadExperimentalMapika(options),
    'mapika-q4': () => loadExperimentalMapikaQ4(options),
    kev08: () => loadExperimentalKev08(options),
    h2o: () => loadH2oCandidate(options, {expectedModel: description.model, runtimeBindings}),
    nari: () => loadNariCandidate(options, {...runtimeBindings, expectedModel: description.model}),
    imajev: () => loadImajevCandidate(options, runtimeBindings),
    pplx: () => loadFlashCandidate(options, runtimeBindings),
    mapika4b: () => loadMapika4Candidate(options, runtimeBindings),
    elephant: () => loadElephantCandidate(options, runtimeBindings),
    clef: () => loadClefCandidate(options, runtimeBindings),
    'opendecider-nano': () => loadOpenDeciderNano(options, runtimeBindings),
    decima: () => loadDecimaModel(options),
    tinydecide: () => loadTinyDecide(options),
    tamev: () => loadTamev(options),
    skejja: () => loadSkejja(options),
    ettin: () => loadEttin(options),
    laya: () => loadLayaModel(options),
    'laya-typed': () => loadLayaTypedDecisionsModel(options),
    xdecision: () => loadXDecisionModel(options),
    'gliner-mobile': () => loadGlinerMobileModel(options),
    'd1-omni': () => loadD1OmniModel(options),
    verdict: () => loadVerdictModel(options),
    safedecide: () => loadSafeDecideModel(options),
    crivo: () => loadCrivoModel(options),
    strands: () =>
      loadExperimentalStrandsFromSource({
        assetsUrl: options.assetsUrl,
        cache: options.cache,
        offline: options.offline,
        signal: options.signal,
        onProgress: options.onProgress,
        maxPrefetchBytes: options.maxPrefetchBytes,
        createTokenizer: (data, config) => new Tokenizer(data, config),
        createRuntime: (wasm) => instantiate(wasm, createPolyfillWASI()),
        gpuApi: globalThis.navigator?.gpu
      }),
    'strands-gemma4': () =>
      loadExperimentalStrandsGemma4FromSource({
        assetsUrl: options.assetsUrl,
        cache: options.cache,
        offline: options.offline,
        signal: options.signal,
        onProgress: options.onProgress,
        maxPrefetchBytes: options.maxPrefetchBytes,
        createTokenizer: (data, config) => new Tokenizer(data, config),
        createRuntime: (wasm) => instantiate(wasm, createPolyfillWASI()),
        gpuApi: globalThis.navigator?.gpu
      })
  };
  const load = loaders[family];
  if (!load) throw new WebDecisionError('INVALID_ASSET', `No loader is registered for ${family}.`);
  return load();
}

/** Load a registered decision model, or a stable public ID with model-specific options. */
export async function loadDecisionModel(options) {
  if (typeof options === 'string')
    throw new WebDecisionError('INVALID_INPUT', 'Pass a stable ID as the modelId property.');
  const stableRequest =
    options && typeof options === 'object' && 'modelId' in options ? normalizeStableModelOptions(options) : undefined;
  const loadOptions = stableRequest?.options ?? options;
  const {family, description} = selectModelRegistration(loadOptions);
  const debertaReadback = loadOptions && Object.getOwnPropertyDescriptor(loadOptions, 'debertaReadback');
  if (
    family !== 'deberta' &&
    debertaReadback &&
    (!Object.hasOwn(debertaReadback, 'value') || debertaReadback.value !== undefined)
  )
    throw new WebDecisionError('INVALID_INPUT', 'debertaReadback is only supported by the DeBERTa model.');
  const model = await loadRegisteredModel(family, loadOptions, description);
  try {
    if (stableRequest?.modelId.startsWith('tamev-')) {
      const descriptor = Object.getOwnPropertyDescriptor(model, 'systemOne');
      if (!descriptor || !Object.hasOwn(descriptor, 'value') || typeof descriptor.value !== 'function')
        throw new WebDecisionError('MODEL_LOAD_FAILED', 'Stable TAMEV model has no System One method.');
      const systemOne = descriptor.value.bind(model);
      Object.defineProperty(model, 'systemOne', {
        ...descriptor,
        value: async (...args) => ({...(await systemOne(...args)), model: stableRequest.modelId})
      });
    }
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
