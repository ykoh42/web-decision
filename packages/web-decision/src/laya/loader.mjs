import {Tokenizer} from '@huggingface/tokenizers';
import {parseJson, waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {createLayaRunner} from './runner.mjs';
import {createLayaSession} from './session.mjs';
import {openLayaSource} from './source.mjs';
import {LAYA_MODEL, LAYA_MODEL_CONFIG, LAYA_SPEC} from './contract.mjs';
import {createLayaTokenizer} from './tokenizer.mjs';

const LOAD_KEYS = new Set(['model', 'assetsUrl', 'cache', 'offline', 'signal', 'onProgress', 'maxPrefetchBytes']);
const fail = (code, message, cause) => new WebDecisionError(code, message, cause ? {cause} : undefined);
const INPUT_NAMES = ['input_ids', 'attention_mask', 'marker_pos', 'marker_mask', 'qtype'];
const OUTPUT_NAMES = ['logits', 'act_logits'];

function validSignal(signal) {
  return (
    signal == null ||
    (typeof signal.throwIfAborted === 'function' &&
      typeof signal.addEventListener === 'function' &&
      typeof signal.removeEventListener === 'function')
  );
}

export function validateLayaLoadOptions(options, model = LAYA_MODEL) {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', 'Laya load options are required.');
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (typeof key !== 'string' || !LOAD_KEYS.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw fail('INVALID_INPUT', 'Unsupported Laya load option.');
  }
  if (
    options.model !== model ||
    typeof options.assetsUrl !== 'string' ||
    !options.assetsUrl ||
    (options.cache !== undefined && typeof options.cache !== 'boolean') ||
    (options.offline !== undefined && typeof options.offline !== 'boolean') ||
    (options.offline && options.cache === false) ||
    (options.maxPrefetchBytes !== undefined && options.maxPrefetchBytes !== 0) ||
    (options.onProgress !== undefined && typeof options.onProgress !== 'function') ||
    !validSignal(options.signal)
  )
    throw fail('INVALID_INPUT', 'Invalid Laya load option values.');

  let base;
  try {
    base = new URL(options.assetsUrl, globalThis.location?.href);
  } catch (cause) {
    throw fail('INVALID_INPUT', 'Invalid Laya assetsUrl.', cause);
  }
  if (!['http:', 'https:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', 'Laya assetsUrl must be an HTTP or HTTPS directory.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return {
    base,
    signal: options.signal,
    cacheEnabled: options.cache !== false,
    offline: options.offline === true,
    onProgress: options.onProgress
  };
}

function validateLayaModelConfig(config, modelConfig = LAYA_MODEL_CONFIG) {
  const laya = config?.laya;
  const specialTokenIds = modelConfig.specialTokenIds;
  const expectedQtypes = {choice: 0, score: 1, noul: 2};
  if (
    config?.model_type !== modelConfig.modelType ||
    config?.vocab_size !== modelConfig.vocabSize ||
    (config?.mask_token_id !== undefined && config.mask_token_id !== specialTokenIds.maskId) ||
    config?.sep_token_id !== specialTokenIds.sepId ||
    config?.pad_token_id !== specialTokenIds.padId ||
    laya?.max_len !== 1024 ||
    laya?.head_max_len !== 256 ||
    laya?.max_option_tokens !== 48 ||
    JSON.stringify(laya?.qtypes) !== JSON.stringify(expectedQtypes) ||
    !Array.isArray(laya?.temperature) ||
    laya.temperature.length !== 3 ||
    laya.temperature.some((value) => !Number.isFinite(value) || value <= 0) ||
    !laya.temperature_by_options ||
    typeof laya.temperature_by_options !== 'object' ||
    Array.isArray(laya.temperature_by_options) ||
    Object.entries(laya.temperature_by_options).some(
      ([key, value]) => !/^(choice|score|noul):(2|3-5|6-10|11\+)$/.test(key) || !Number.isFinite(value) || value <= 0
    ) ||
    !Array.isArray(laya.temperature_clamp) ||
    laya.temperature_clamp.length !== 2 ||
    laya.temperature_clamp[0] !== 0.5 ||
    laya.temperature_clamp[1] !== 5
  )
    throw fail('INVALID_ASSET', 'Laya config does not match the pinned checkpoint.');
  return {byType: [...laya.temperature], byOptions: {...laya.temperature_by_options}};
}

function validateSession(session) {
  const inputs = new Set(session?.inputNames ?? []);
  const outputs = new Set(session?.outputNames ?? []);
  if (!INPUT_NAMES.every((name) => inputs.has(name)) || !OUTPUT_NAMES.every((name) => outputs.has(name)))
    throw fail('INVALID_ASSET', 'Laya ONNX graph has an unexpected interface.');
  return session;
}

async function createSession(ort, adapter, graphBytes, dataBytes, signal) {
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
        graphOptimizationLevel: 'all',
        externalData: [{path: 'model_fp16.onnx_data', data: new Uint8Array(dataBytes)}]
      }),
    signal,
    (lateSession) => lateSession?.release()
  );
}

async function getHardwareAdapter(gpu, signal) {
  if (typeof gpu?.requestAdapter !== 'function') throw fail('UNSUPPORTED_ENVIRONMENT', 'Laya requires WebGPU.');
  let adapter;
  try {
    adapter = await waitForAbort(
      () => gpu.requestAdapter({powerPreference: 'high-performance', forceFallbackAdapter: false}),
      signal
    );
  } catch (cause) {
    signal?.throwIfAborted();
    throw fail('MODEL_LOAD_FAILED', 'Could not initialize a hardware WebGPU adapter for Laya.', cause);
  }
  if (!adapter) throw fail('UNSUPPORTED_ENVIRONMENT', 'No hardware WebGPU adapter is available for Laya.');
  if (adapter.info?.isFallbackAdapter === true || adapter.isFallbackAdapter === true)
    throw fail('UNSUPPORTED_ENVIRONMENT', 'Laya does not run on a fallback WebGPU adapter.');
  return adapter;
}

export async function loadLayaModel(
  options,
  {
    ort,
    Tokenizer: TokenizerClass = Tokenizer,
    openSource = openLayaSource,
    gpuApi = globalThis.navigator?.gpu,
    model = LAYA_MODEL,
    modelName = 'Laya',
    spec = LAYA_SPEC
  } = {}
) {
  const checked = validateLayaLoadOptions(options, model);
  const {signal} = checked;
  let session;
  try {
    const runtime = ort ?? (await import('onnxruntime-web/webgpu'));
    signal?.throwIfAborted();
    const source = await openSource({...checked, spec});
    const [configBytes, tokenizerConfigBytes, tokenizerBytes] = await Promise.all([
      source.read('config.json'),
      source.read('tokenizer_config.json'),
      source.read('tokenizer.json')
    ]);
    const config = parseJson(configBytes, 'config.json');
    const modelConfig = spec.modelConfig ?? LAYA_MODEL_CONFIG;
    const temperatures = validateLayaModelConfig(config, modelConfig);
    const tokenizer = createLayaTokenizer(
      TokenizerClass,
      parseJson(tokenizerBytes, 'tokenizer.json'),
      parseJson(tokenizerConfigBytes, 'tokenizer_config.json'),
      config,
      modelConfig.specialTokenIds
    );
    signal?.throwIfAborted();

    const [graphBytes, dataBytes] = await Promise.all([
      source.read('onnx/model_fp16.onnx'),
      source.read('onnx/model_fp16.onnx_data')
    ]);
    signal?.throwIfAborted();
    const adapter = await getHardwareAdapter(gpuApi, signal);
    session = validateSession(await createSession(ort ?? runtime, adapter, graphBytes, dataBytes, signal));
    signal?.throwIfAborted();
    const runner = createLayaRunner(ort ?? runtime, session, tokenizer);
    const loaded = createLayaSession({runner, tokenizer, temperatures, model});
    session = null;
    return loaded;
  } catch (cause) {
    await Promise.allSettled([session?.release()].filter(Boolean));
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
    throw fail('MODEL_LOAD_FAILED', `Could not load ${modelName} on ONNX Runtime WebGPU.`, cause);
  }
}
