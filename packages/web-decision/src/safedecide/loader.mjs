import {Tokenizer} from '@huggingface/tokenizers';
import {parseJson, waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {SAFEDECIDE_MODEL, SAFEDECIDE_MODEL_CONFIG, SAFEDECIDE_SPEC} from './contract.mjs';
import {createSafeDecideRunner} from './runner.mjs';
import {createSafeDecideSession} from './session.mjs';
import {openSafeDecideSource} from './source.mjs';
import {createSafeDecideTokenizer} from './tokenizer.mjs';

const LOAD_KEYS = new Set(['model', 'assetsUrl', 'cache', 'offline', 'signal', 'onProgress']);
const fail = (code, message, cause) => new WebDecisionError(code, message, cause ? {cause} : undefined);

function validSignal(signal) {
  return (
    signal == null ||
    (typeof signal.throwIfAborted === 'function' &&
      typeof signal.addEventListener === 'function' &&
      typeof signal.removeEventListener === 'function')
  );
}

export function validateSafeDecideLoadOptions(options, model = SAFEDECIDE_MODEL) {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', 'SafeDecide load options are required.');
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (typeof key !== 'string' || !LOAD_KEYS.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw fail('INVALID_INPUT', 'Unsupported SafeDecide load option.');
  }
  if (
    options.model !== model ||
    typeof options.assetsUrl !== 'string' ||
    !options.assetsUrl ||
    (options.cache !== undefined && typeof options.cache !== 'boolean') ||
    (options.offline !== undefined && typeof options.offline !== 'boolean') ||
    (options.offline && options.cache !== true) ||
    (options.onProgress !== undefined && typeof options.onProgress !== 'function') ||
    !validSignal(options.signal)
  )
    throw fail('INVALID_INPUT', 'Invalid SafeDecide load option values.');

  let base;
  try {
    base = new URL(options.assetsUrl, globalThis.location?.href);
  } catch (cause) {
    throw fail('INVALID_INPUT', 'Invalid SafeDecide assetsUrl.', cause);
  }
  if (!['http:', 'https:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', 'SafeDecide assetsUrl must be an HTTP or HTTPS directory.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return {
    base,
    signal: options.signal,
    cacheEnabled: options.cache === true,
    offline: options.offline === true,
    onProgress: options.onProgress
  };
}

function positiveTemperatures(temperatures) {
  return (
    temperatures &&
    !Array.isArray(temperatures) &&
    ['choice', 'noul', 'score'].every((type) => Number.isFinite(temperatures[type]) && temperatures[type] > 0) &&
    Object.keys(temperatures).every((type) => ['choice', 'noul', 'score'].includes(type))
  );
}

function validateRules(config) {
  const expected = [
    {alpha: 0.05, threshold: null},
    {alpha: 0.1, threshold: 0.7},
    {alpha: 0.2, threshold: 0.5}
  ];
  if (
    !Array.isArray(config.selective_rules) ||
    config.selective_rules.length !== expected.length ||
    config.selective_rules.some(
      (rule, index) =>
        rule?.alpha !== expected[index].alpha ||
        rule?.threshold !== expected[index].threshold ||
        rule?.delta !== 0.1 ||
        typeof rule?.statement !== 'string' ||
        !rule.statement
    ) ||
    config.selective_rule?.alpha !== 0.1 ||
    config.selective_rule.threshold !== 0.7 ||
    config.selective_rule.delta !== 0.1 ||
    config.selective_rule.statement !== config.selective_rules[1].statement
  )
    throw fail('INVALID_ASSET', 'SafeDecide certified action rules do not match the pinned checkpoint.');
}

export function validateSafeDecideModelConfig(
  config,
  safedecideConfig,
  tokenizerConfig,
  modelConfig = SAFEDECIDE_MODEL_CONFIG
) {
  if (
    config?.model_type !== modelConfig.modelType ||
    config?.vocab_size !== modelConfig.vocabSize ||
    config?.hidden_size !== modelConfig.hiddenSize ||
    config?.num_hidden_layers !== modelConfig.layers ||
    config?.max_position_embeddings !== 8192 ||
    config?.bos_token_id !== modelConfig.specialTokenIds.bosId ||
    config?.eos_token_id !== modelConfig.specialTokenIds.eosId ||
    config?.pad_token_id !== modelConfig.specialTokenIds.padId ||
    config?.mask_token_id !== modelConfig.specialTokenIds.maskId ||
    (config?.num_labels !== undefined && config.num_labels !== 1) ||
    safedecideConfig?.base_model !== 'jhu-clsp/mmBERT-small' ||
    safedecideConfig?.profile !== 'clean' ||
    safedecideConfig?.max_length !== SAFEDECIDE_SPEC.maxSequenceTokens ||
    !positiveTemperatures(safedecideConfig?.temperatures) ||
    (safedecideConfig?.question_bias && Object.keys(safedecideConfig.question_bias).length > 0) ||
    tokenizerConfig?.max_length !== 384 ||
    tokenizerConfig?.truncation_side !== 'right' ||
    tokenizerConfig?.padding_side !== 'right'
  )
    throw fail('INVALID_ASSET', 'SafeDecide configuration does not match the pinned checkpoint.');
  validateRules(safedecideConfig);
  return safedecideConfig;
}

function validateSession(session) {
  const inputs = session?.inputNames ?? [];
  const outputs = session?.outputNames ?? [];
  if (
    inputs.length !== 2 ||
    !inputs.includes('input_ids') ||
    !inputs.includes('attention_mask') ||
    outputs.length !== 1 ||
    outputs[0] !== 'logits'
  )
    throw fail('INVALID_ASSET', 'SafeDecide ONNX graph has an unexpected interface.');
  return session;
}

async function getHardwareAdapter(gpu, signal) {
  if (typeof gpu?.requestAdapter !== 'function') throw fail('UNSUPPORTED_ENVIRONMENT', 'SafeDecide requires WebGPU.');
  let adapter;
  try {
    adapter = await waitForAbort(
      () => gpu.requestAdapter({powerPreference: 'high-performance', forceFallbackAdapter: false}),
      signal
    );
  } catch (cause) {
    signal?.throwIfAborted();
    throw fail('MODEL_LOAD_FAILED', 'Could not initialize a hardware WebGPU adapter for SafeDecide.', cause);
  }
  if (!adapter) throw fail('UNSUPPORTED_ENVIRONMENT', 'No hardware WebGPU adapter is available for SafeDecide.');
  if (adapter.info?.isFallbackAdapter === true || adapter.isFallbackAdapter === true)
    throw fail('UNSUPPORTED_ENVIRONMENT', 'SafeDecide does not run on a fallback WebGPU adapter.');
  return adapter;
}

async function createSession(ort, adapter, graphBytes, signal) {
  if (typeof ort?.InferenceSession?.create !== 'function' || typeof ort?.Tensor !== 'function')
    throw fail('UNSUPPORTED_ENVIRONMENT', 'ONNX Runtime WebGPU is unavailable.');
  const webgpu = ort.env?.webgpu;
  if (!webgpu) throw fail('UNSUPPORTED_ENVIRONMENT', 'ONNX Runtime WebGPU execution provider is unavailable.');
  webgpu.powerPreference = 'high-performance';
  webgpu.forceFallbackAdapter = false;
  webgpu.adapter = adapter;
  return waitForAbort(
    () =>
      ort.InferenceSession.create(new Uint8Array(graphBytes), {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all'
      }),
    signal,
    (lateSession) => lateSession?.release()
  );
}

async function load(options, runtime, spec, model) {
  const checked = validateSafeDecideLoadOptions(options, model);
  const {signal} = checked;
  let session;
  try {
    signal?.throwIfAborted();
    const source = await runtime.openSource({...checked, spec});
    const [configBytes, safeConfigBytes, tokenizerConfigBytes, tokenizerBytes] = await Promise.all([
      source.read('config.json'),
      source.read('safedecide_config.json'),
      source.read('tokenizer_config.json'),
      source.read('tokenizer.json')
    ]);
    const config = parseJson(configBytes, 'config.json');
    const safedecideConfig = validateSafeDecideModelConfig(
      config,
      parseJson(safeConfigBytes, 'safedecide_config.json'),
      parseJson(tokenizerConfigBytes, 'tokenizer_config.json'),
      spec.modelConfig
    );
    const tokenizer = runtime.createTokenizer(
      parseJson(tokenizerBytes, 'tokenizer.json'),
      parseJson(tokenizerConfigBytes, 'tokenizer_config.json'),
      spec.modelConfig,
      spec.maxSequenceTokens
    );
    signal?.throwIfAborted();
    const adapter = await getHardwareAdapter(runtime.gpuApi, signal);
    signal?.throwIfAborted();
    const graphBytes = await source.read('onnx/model.onnx');
    signal?.throwIfAborted();
    session = await createSession(runtime.ort, adapter, graphBytes, signal);
    validateSession(session);
    signal?.throwIfAborted();
    const runner = runtime.createRunner(runtime.ort, session, tokenizer.special.padId);
    const loaded = runtime.createLoadedSession({runner, tokenizer, config: safedecideConfig, model});
    session = null;
    return loaded;
  } catch (cause) {
    await Promise.allSettled([session?.release()].filter(Boolean));
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
    throw fail('MODEL_LOAD_FAILED', `Could not load ${model} on ONNX Runtime WebGPU.`, cause);
  }
}

function runtimeDefaults(runtime = {}) {
  return {
    ...runtime,
    Tokenizer: runtime.Tokenizer ?? Tokenizer,
    gpuApi: runtime.gpuApi ?? globalThis.navigator?.gpu,
    openSource: runtime.openSource ?? openSafeDecideSource,
    createTokenizer:
      runtime.createTokenizer ??
      ((data, tokenizerConfig, modelConfig, maxSequenceTokens) =>
        createSafeDecideTokenizer(
          runtime.Tokenizer ?? Tokenizer,
          data,
          tokenizerConfig,
          modelConfig,
          maxSequenceTokens
        )),
    createRunner: runtime.createRunner ?? ((ort, session, padId) => createSafeDecideRunner(ort, session, padId)),
    createLoadedSession: runtime.createLoadedSession ?? createSafeDecideSession
  };
}

export async function loadSafeDecideModel(options, runtime = {}) {
  validateSafeDecideLoadOptions(options);
  const defaults = runtimeDefaults(runtime);
  const ort = defaults.ort ?? (await import('onnxruntime-web/webgpu'));
  return load(options, {...defaults, ort}, SAFEDECIDE_SPEC, SAFEDECIDE_MODEL);
}

export async function loadFakeSafeDecide(options, deps) {
  if (deps?.fake !== true) throw new TypeError('Fake dependencies are required.');
  return load(options, runtimeDefaults(deps), deps.spec ?? SAFEDECIDE_SPEC, deps.model ?? SAFEDECIDE_MODEL);
}
