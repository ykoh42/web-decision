import {Tokenizer} from '@huggingface/tokenizers';
import {parseJson, waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {CRIVO_REPOSITORY, CRIVO_REVISION, getCrivoSpec} from './contract.mjs';
import {createCrivoRunner} from './runner.mjs';
import {createCrivoSession} from './session.mjs';
import {openCrivoSource} from './source.mjs';
import {createCrivoTokenizer, validateCrivoConfig} from './tokenizer.mjs';

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

export function validateCrivoLoadOptions(options) {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', 'Crivo load options are required.');
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (typeof key !== 'string' || !LOAD_KEYS.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw fail('INVALID_INPUT', 'Unsupported Crivo load option.');
  }
  const spec = getCrivoSpec(options.model);
  if (
    !spec ||
    (options.assetsUrl !== undefined && (typeof options.assetsUrl !== 'string' || !options.assetsUrl)) ||
    (options.cache !== undefined && typeof options.cache !== 'boolean') ||
    (options.offline !== undefined && typeof options.offline !== 'boolean') ||
    (options.offline && options.cache !== true) ||
    (options.onProgress !== undefined && typeof options.onProgress !== 'function') ||
    !validSignal(options.signal)
  )
    throw fail('INVALID_INPUT', 'Invalid Crivo load option values.');

  let base;
  try {
    const sourceUrl =
      options.assetsUrl ?? `https://huggingface.co/${CRIVO_REPOSITORY}/resolve/${CRIVO_REVISION}/models/32m/`;
    base = new URL(sourceUrl, globalThis.location?.href);
  } catch (cause) {
    throw fail('INVALID_INPUT', 'Invalid Crivo assetsUrl.', cause);
  }
  if (!['http:', 'https:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', 'Crivo assetsUrl must be an HTTP or HTTPS directory.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return {
    base,
    spec,
    signal: options.signal,
    cacheEnabled: options.cache !== false,
    offline: options.offline === true,
    onProgress: options.onProgress
  };
}

function validateSession(session) {
  const inputs = session?.inputNames ?? [];
  const outputs = session?.outputNames ?? [];
  const expectedInputs = ['input_ids', 'attention_mask', 'opt_positions', 'opt_mask', 'qtype'];
  if (
    inputs.length !== expectedInputs.length ||
    expectedInputs.some((name) => !inputs.includes(name)) ||
    outputs.length !== 1 ||
    outputs[0] !== 'logits'
  )
    throw fail('INVALID_ASSET', 'Crivo ONNX graph has an unexpected interface.');
  return session;
}

async function getHardwareAdapter(gpu, signal) {
  if (typeof gpu?.requestAdapter !== 'function') throw fail('UNSUPPORTED_ENVIRONMENT', 'Crivo FP32 requires WebGPU.');
  let adapter;
  try {
    adapter = await waitForAbort(
      () => gpu.requestAdapter({powerPreference: 'high-performance', forceFallbackAdapter: false}),
      signal
    );
  } catch (cause) {
    signal?.throwIfAborted();
    throw fail('MODEL_LOAD_FAILED', 'Could not initialize a hardware WebGPU adapter for Crivo.', cause);
  }
  if (!adapter) throw fail('UNSUPPORTED_ENVIRONMENT', 'No hardware WebGPU adapter is available for Crivo.');
  if (adapter.info?.isFallbackAdapter === true || adapter.isFallbackAdapter === true)
    throw fail('UNSUPPORTED_ENVIRONMENT', 'Crivo FP32 does not run on a fallback WebGPU adapter.');
  return adapter;
}

async function createInferenceSession(ort, spec, bytes, gpuApi, signal) {
  if (typeof ort?.InferenceSession?.create !== 'function' || typeof ort?.Tensor !== 'function')
    throw fail('UNSUPPORTED_ENVIRONMENT', 'ONNX Runtime Web is unavailable.');
  const options = {
    executionProviders: [spec.backend === 'onnxruntime-web-webgpu' ? 'webgpu' : 'wasm'],
    graphOptimizationLevel: 'all'
  };
  if (spec.backend === 'onnxruntime-web-webgpu') {
    const adapter = await getHardwareAdapter(gpuApi, signal);
    const webgpu = ort.env?.webgpu;
    if (!webgpu) throw fail('UNSUPPORTED_ENVIRONMENT', 'ONNX Runtime WebGPU is unavailable.');
    webgpu.powerPreference = 'high-performance';
    webgpu.forceFallbackAdapter = false;
    webgpu.adapter = adapter;
  } else {
    const wasm = ort.env?.wasm;
    if (!wasm) throw fail('UNSUPPORTED_ENVIRONMENT', 'ONNX Runtime WASM is unavailable.');
    wasm.numThreads = 1;
    options.intraOpNumThreads = 1;
    options.interOpNumThreads = 1;
  }
  return waitForAbort(
    () => ort.InferenceSession.create(new Uint8Array(bytes), options),
    signal,
    (lateSession) => lateSession?.release()
  );
}

async function load(options, runtime, spec) {
  const checked = validateCrivoLoadOptions(options);
  const {signal} = checked;
  let session;
  try {
    signal?.throwIfAborted();
    const source = await runtime.openSource({...checked, spec}, spec);
    const [configBytes, tokenizerConfigBytes, tokenizerBytes] = await Promise.all([
      source.read('config.json'),
      source.read('tokenizer_config.json'),
      source.read('tokenizer.json')
    ]);
    const config = parseJson(configBytes, 'config.json');
    const tokenizerConfig = parseJson(tokenizerConfigBytes, 'tokenizer_config.json');
    const tokenizerJson = parseJson(tokenizerBytes, 'tokenizer.json');
    validateCrivoConfig(config, tokenizerConfig);
    const tokenizer = runtime.createTokenizer(tokenizerJson, tokenizerConfig);
    signal?.throwIfAborted();
    const graphBytes = await source.read(spec.graphFile);
    session = await createInferenceSession(runtime.ort, spec, graphBytes, runtime.gpuApi, signal);
    validateSession(session);
    signal?.throwIfAborted();
    const runner = runtime.createRunner(runtime.ort, session);
    const loaded = runtime.createLoadedSession({runner, tokenizer, model: spec.model});
    session = null;
    return loaded;
  } catch (cause) {
    await Promise.allSettled([session?.release()].filter(Boolean));
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
    throw fail('MODEL_LOAD_FAILED', 'Could not load ' + spec.model + '.', cause);
  }
}

function runtimeDefaults(runtime = {}) {
  return {
    ...runtime,
    Tokenizer: runtime.Tokenizer ?? Tokenizer,
    gpuApi: runtime.gpuApi ?? globalThis.navigator?.gpu,
    openSource: runtime.openSource ?? openCrivoSource,
    createTokenizer:
      runtime.createTokenizer ?? ((data, config) => createCrivoTokenizer(runtime.Tokenizer ?? Tokenizer, data, config)),
    createRunner: runtime.createRunner ?? ((ort, session) => createCrivoRunner(ort, session)),
    createLoadedSession: runtime.createLoadedSession ?? createCrivoSession
  };
}

export async function loadCrivoModel(options, runtime = {}) {
  const checked = validateCrivoLoadOptions(options);
  const defaults = runtimeDefaults(runtime);
  const ort =
    defaults.ort ??
    (await import(
      checked.spec.backend === 'onnxruntime-web-webgpu' ? 'onnxruntime-web/webgpu' : 'onnxruntime-web/wasm'
    ));
  return load(options, {...defaults, ort}, checked.spec);
}
