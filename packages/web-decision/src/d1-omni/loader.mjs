import {Tokenizer} from '@huggingface/tokenizers';
import {parseJson, waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {createD1OmniSession} from './session.mjs';
import {D1_OMNI_ASSETS_URL, D1_OMNI_MODEL, D1_OMNI_SPEC, D1_OMNI_TEMPERATURES} from './contract.mjs';
import {openD1OmniSource} from './source.mjs';
import {createD1OmniTokenizer} from './tokenizer.mjs';
import {createD1OmniRunner, D1_OMNI_INPUT_NAMES, D1_OMNI_OUTPUT_NAMES} from './runner.mjs';

const LOAD_KEYS = new Set(['model', 'assetsUrl', 'cache', 'offline', 'signal', 'onProgress', 'maxPrefetchBytes']);
const fail = (code, message, cause) => new WebDecisionError(code, message, cause ? {cause} : undefined);

function validSignal(signal) {
  return (
    signal == null ||
    (typeof signal.throwIfAborted === 'function' &&
      typeof signal.addEventListener === 'function' &&
      typeof signal.removeEventListener === 'function')
  );
}

export function validateD1OmniLoadOptions(options) {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', 'd1-omni load options are required.');
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (typeof key !== 'string' || !LOAD_KEYS.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw fail('INVALID_INPUT', 'Unsupported d1-omni load option.');
  }
  const assetsUrl = options.assetsUrl === undefined ? D1_OMNI_ASSETS_URL : options.assetsUrl;
  if (
    options.model !== D1_OMNI_MODEL ||
    typeof assetsUrl !== 'string' ||
    !assetsUrl ||
    (options.cache !== undefined && typeof options.cache !== 'boolean') ||
    (options.offline !== undefined && typeof options.offline !== 'boolean') ||
    (options.offline && options.cache === false) ||
    (options.maxPrefetchBytes !== undefined && options.maxPrefetchBytes !== 0) ||
    (options.onProgress !== undefined && typeof options.onProgress !== 'function') ||
    !validSignal(options.signal)
  )
    throw fail('INVALID_INPUT', 'Invalid d1-omni load option values.');

  let base;
  try {
    base = new URL(assetsUrl, globalThis.location?.href);
  } catch (cause) {
    throw fail('INVALID_INPUT', 'Invalid d1-omni assetsUrl.', cause);
  }
  if (!['http:', 'https:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', 'd1-omni assetsUrl must be an HTTP or HTTPS directory.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return {
    base,
    signal: options.signal,
    cacheEnabled: options.cache !== false,
    offline: options.offline === true,
    onProgress: options.onProgress
  };
}

function validateConfig(config) {
  const temperatures = config?.temperatures;
  const expected = D1_OMNI_TEMPERATURES;
  if (
    config?.model_type !== 'd1_omni' ||
    config?.max_length !== D1_OMNI_SPEC.maxTokens ||
    config?.bos_token_id !== 1 ||
    config?.pad_token_id !== 0 ||
    config?.['transformers.js_config']?.dtype !== 'q8' ||
    config?.d1_omni?.max_length !== D1_OMNI_SPEC.maxTokens ||
    !temperatures ||
    Object.keys(expected).some((key) => temperatures[key] !== expected[key])
  )
    throw fail('INVALID_ASSET', 'd1-omni config does not match the pinned Q8 checkpoint.');
  return expected;
}

function validateSession(session) {
  const inputs = new Set(session?.inputNames ?? []);
  const outputs = new Set(session?.outputNames ?? []);
  if (
    !D1_OMNI_INPUT_NAMES.every((name) => inputs.has(name)) ||
    !D1_OMNI_OUTPUT_NAMES.every((name) => outputs.has(name))
  )
    throw fail('INVALID_ASSET', 'd1-omni ONNX graph has an unexpected interface.');
  return session;
}

async function getHardwareAdapter(gpu, signal) {
  if (typeof gpu?.requestAdapter !== 'function') throw fail('UNSUPPORTED_ENVIRONMENT', 'd1-omni requires WebGPU.');
  let adapter;
  try {
    adapter = await waitForAbort(
      () => gpu.requestAdapter({powerPreference: 'high-performance', forceFallbackAdapter: false}),
      signal
    );
  } catch (cause) {
    signal?.throwIfAborted();
    throw fail('MODEL_LOAD_FAILED', 'Could not initialize a hardware WebGPU adapter for d1-omni.', cause);
  }
  if (!adapter) throw fail('UNSUPPORTED_ENVIRONMENT', 'No hardware WebGPU adapter is available for d1-omni.');
  if (adapter.info?.isFallbackAdapter === true || adapter.isFallbackAdapter === true)
    throw fail('UNSUPPORTED_ENVIRONMENT', 'd1-omni does not run on a fallback WebGPU adapter.');
  return adapter;
}

async function createSession(ort, adapter, graphBytes, dataBytes, signal) {
  if (typeof ort?.InferenceSession?.create !== 'function' || typeof ort?.Tensor !== 'function')
    throw fail('UNSUPPORTED_ENVIRONMENT', 'ONNX Runtime WebGPU is unavailable.');
  if (!ort.env?.webgpu) throw fail('UNSUPPORTED_ENVIRONMENT', 'ONNX Runtime WebGPU provider is unavailable.');
  ort.env.webgpu.powerPreference = 'high-performance';
  ort.env.webgpu.forceFallbackAdapter = false;
  ort.env.webgpu.adapter = adapter;
  return waitForAbort(
    () =>
      ort.InferenceSession.create(new Uint8Array(graphBytes), {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all',
        externalData: [{path: 'model_quantized.onnx_data', data: new Uint8Array(dataBytes)}]
      }),
    signal,
    (lateSession) => lateSession?.release()
  );
}

export async function loadD1OmniModel(
  options,
  {ort, Tokenizer: TokenizerClass = Tokenizer, openSource = openD1OmniSource, gpuApi = globalThis.navigator?.gpu} = {}
) {
  const checked = validateD1OmniLoadOptions(options);
  const {signal} = checked;
  let session;
  try {
    const runtime = ort ?? (await import('onnxruntime-web/webgpu'));
    signal?.throwIfAborted();
    const source = await openSource({...checked, spec: D1_OMNI_SPEC});
    const [configBytes, tokenizerConfigBytes, tokenizerBytes] = await Promise.all([
      source.read('config.json'),
      source.read('tokenizer_config.json'),
      source.read('tokenizer.json')
    ]);
    const config = parseJson(configBytes, 'config.json');
    const temperatures = validateConfig(config);
    const tokenizer = createD1OmniTokenizer(
      TokenizerClass,
      parseJson(tokenizerBytes, 'tokenizer.json'),
      parseJson(tokenizerConfigBytes, 'tokenizer_config.json')
    );
    signal?.throwIfAborted();

    const [graphBytes, dataBytes] = await Promise.all([
      source.read('onnx/model_quantized.onnx'),
      source.read('onnx/model_quantized.onnx_data')
    ]);
    signal?.throwIfAborted();
    const adapter = await getHardwareAdapter(gpuApi, signal);
    session = validateSession(await createSession(ort ?? runtime, adapter, graphBytes, dataBytes, signal));
    signal?.throwIfAborted();
    const runner = createD1OmniRunner(ort ?? runtime, session, tokenizer.ids['<|pad|>']);
    const model = createD1OmniSession({runner, tokenizer, temperatures, model: D1_OMNI_MODEL});
    session = null;
    return model;
  } catch (cause) {
    await Promise.allSettled([session?.release()].filter(Boolean));
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
    throw fail('MODEL_LOAD_FAILED', 'Could not load d1-omni Q8 on ONNX Runtime WebGPU.', cause);
  }
}
