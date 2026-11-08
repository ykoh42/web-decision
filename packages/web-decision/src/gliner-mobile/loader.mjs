import {Tokenizer} from '@huggingface/tokenizers';
import {parseJson, waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {createGlinerMobileRunner} from './runner.mjs';
import {createGlinerMobileSession} from './session.mjs';
import {GLINER_MOBILE_MODEL} from './contract.mjs';
import {openGlinerMobileSource} from './source.mjs';
import {createGlinerMobileTokenizer} from './tokenizer.mjs';

const LOAD_KEYS = new Set(['model', 'assetsUrl', 'cache', 'offline', 'signal', 'onProgress', 'maxPrefetchBytes']);
const fail = (code, message, cause) => new WebDecisionError(code, message, cause ? {cause} : undefined);
const INPUT_NAMES = ['input_ids', 'attention_mask', 'marker_positions'];
const OUTPUT_NAMES = ['logits'];

function validSignal(signal) {
  return (
    signal == null ||
    (typeof signal.throwIfAborted === 'function' &&
      typeof signal.addEventListener === 'function' &&
      typeof signal.removeEventListener === 'function')
  );
}

export function validateGlinerMobileLoadOptions(options) {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', 'GLiNER load options are required.');
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (typeof key !== 'string' || !LOAD_KEYS.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw fail('INVALID_INPUT', 'Unsupported GLiNER load option.');
  }
  if (
    options.model !== GLINER_MOBILE_MODEL ||
    typeof options.assetsUrl !== 'string' ||
    !options.assetsUrl ||
    (options.cache !== undefined && typeof options.cache !== 'boolean') ||
    (options.offline !== undefined && typeof options.offline !== 'boolean') ||
    (options.offline && options.cache === false) ||
    (options.maxPrefetchBytes !== undefined && options.maxPrefetchBytes !== 0) ||
    (options.onProgress !== undefined && typeof options.onProgress !== 'function') ||
    !validSignal(options.signal)
  )
    throw fail('INVALID_INPUT', 'Invalid GLiNER load option values.');

  let base;
  try {
    base = new URL(options.assetsUrl, globalThis.location?.href);
  } catch (cause) {
    throw fail('INVALID_INPUT', 'Invalid GLiNER assetsUrl.', cause);
  }
  if (!['http:', 'https:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', 'GLiNER assetsUrl must be an HTTP or HTTPS directory.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return {
    base,
    signal: options.signal,
    cacheEnabled: options.cache !== false,
    offline: options.offline === true,
    onProgress: options.onProgress
  };
}

function validateModelConfig(config) {
  const gliner = config?.gliner2;
  if (
    config?.model_type !== 'deberta-v2' ||
    config?.vocab_size !== 128011 ||
    config?.max_position_embeddings !== 512 ||
    gliner?.max_len !== 512 ||
    gliner?.temperature !== 1 ||
    gliner?.source !== 'fastino/GLiNER2.5-Decide' ||
    gliner?.inputs !== 'input_ids, attention_mask, marker_positions (index of each [L] token)' ||
    gliner?.output !== "logits (batch, markers); softmax within each question's markers" ||
    config?.['transformers.js_config']?.dtype !== 'q4f16' ||
    JSON.stringify(config['transformers.js_config'].use_external_data_format) !==
      JSON.stringify({'model_q4f16.onnx': 1})
  )
    throw fail('INVALID_ASSET', 'GLiNER config does not match the pinned mobile checkpoint.');
}

function validateSession(session) {
  const inputs = new Set(session?.inputNames ?? []);
  const outputs = new Set(session?.outputNames ?? []);
  if (!INPUT_NAMES.every((name) => inputs.has(name)) || !OUTPUT_NAMES.every((name) => outputs.has(name)))
    throw fail('INVALID_ASSET', 'GLiNER ONNX graph has an unexpected interface.');
  return session;
}

async function createSession(ort, adapter, graphBytes, dataBytes, signal) {
  if (typeof ort?.InferenceSession?.create !== 'function' || typeof ort?.Tensor !== 'function')
    throw fail('UNSUPPORTED_ENVIRONMENT', 'ONNX Runtime WebGPU is unavailable.');
  const webgpu = ort.env?.webgpu;
  if (!webgpu) throw fail('UNSUPPORTED_ENVIRONMENT', 'ONNX Runtime WebGPU provider is unavailable.');
  webgpu.powerPreference = 'high-performance';
  webgpu.forceFallbackAdapter = false;
  webgpu.adapter = adapter;
  return waitForAbort(
    () =>
      ort.InferenceSession.create(new Uint8Array(graphBytes), {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all',
        externalData: [{path: 'model_q4f16.onnx_data', data: new Uint8Array(dataBytes)}]
      }),
    signal,
    (lateSession) => lateSession?.release()
  );
}

async function getHardwareAdapter(gpu, signal) {
  if (typeof gpu?.requestAdapter !== 'function') throw fail('UNSUPPORTED_ENVIRONMENT', 'GLiNER requires WebGPU.');
  let adapter;
  try {
    adapter = await waitForAbort(
      () => gpu.requestAdapter({powerPreference: 'high-performance', forceFallbackAdapter: false}),
      signal
    );
  } catch (cause) {
    signal?.throwIfAborted();
    throw fail('MODEL_LOAD_FAILED', 'Could not initialize a hardware WebGPU adapter for GLiNER.', cause);
  }
  if (!adapter) throw fail('UNSUPPORTED_ENVIRONMENT', 'No hardware WebGPU adapter is available for GLiNER.');
  if (adapter.info?.isFallbackAdapter === true || adapter.isFallbackAdapter === true)
    throw fail('UNSUPPORTED_ENVIRONMENT', 'GLiNER does not run on a fallback WebGPU adapter.');
  return adapter;
}

export async function loadGlinerMobileModel(
  options,
  {
    ort,
    Tokenizer: TokenizerClass = Tokenizer,
    openSource = openGlinerMobileSource,
    gpuApi = globalThis.navigator?.gpu
  } = {}
) {
  const checked = validateGlinerMobileLoadOptions(options);
  const {signal} = checked;
  let session;
  try {
    const runtime = ort ?? (await import('onnxruntime-web/webgpu'));
    signal?.throwIfAborted();
    const source = await openSource(checked);
    const [configBytes, specialTokenMapBytes, tokenizerConfigBytes, tokenizerBytes] = await Promise.all([
      source.read('config.json'),
      source.read('special_tokens_map.json'),
      source.read('tokenizer_config.json'),
      source.read('tokenizer.json')
    ]);
    const config = parseJson(configBytes, 'config.json');
    validateModelConfig(config);
    const tokenizer = createGlinerMobileTokenizer(
      TokenizerClass,
      parseJson(tokenizerBytes, 'tokenizer.json'),
      parseJson(tokenizerConfigBytes, 'tokenizer_config.json'),
      parseJson(specialTokenMapBytes, 'special_tokens_map.json'),
      config
    );
    signal?.throwIfAborted();

    const [graphBytes, dataBytes] = await Promise.all([
      source.read('onnx/model_q4f16.onnx'),
      source.read('onnx/model_q4f16.onnx_data')
    ]);
    signal?.throwIfAborted();
    const adapter = await getHardwareAdapter(gpuApi, signal);
    session = validateSession(await createSession(ort ?? runtime, adapter, graphBytes, dataBytes, signal));
    signal?.throwIfAborted();
    const runner = createGlinerMobileRunner(ort ?? runtime, session);
    const model = createGlinerMobileSession({runner, tokenizer, model: GLINER_MOBILE_MODEL});
    session = null;
    return model;
  } catch (cause) {
    await Promise.allSettled([session?.release()].filter(Boolean));
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
    throw fail('MODEL_LOAD_FAILED', 'Could not load GLiNER on ONNX Runtime WebGPU.', cause);
  }
}
