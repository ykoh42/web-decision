import {Tokenizer} from '@huggingface/tokenizers';
import {parseJson, waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {
  VERDICT_ASSETS_URL,
  VERDICT_GLOBAL_TEMPERATURE,
  VERDICT_MODEL,
  VERDICT_SPEC,
  VERDICT_TEMPERATURES
} from './contract.mjs';
import {createVerdictRunner, VERDICT_INPUT_NAMES, VERDICT_OUTPUT_NAMES} from './runner.mjs';
import {createVerdictSession} from './session.mjs';
import {openVerdictSource} from './source.mjs';
import {createVerdictTokenizer} from './tokenizer.mjs';

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

export function validateVerdictLoadOptions(options) {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', 'Verdict load options are required.');
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (typeof key !== 'string' || !LOAD_KEYS.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw fail('INVALID_INPUT', 'Unsupported Verdict load option.');
  }
  const assetsUrl = options.assetsUrl === undefined ? VERDICT_ASSETS_URL : options.assetsUrl;
  if (
    options.model !== VERDICT_MODEL ||
    typeof assetsUrl !== 'string' ||
    !assetsUrl ||
    (options.cache !== undefined && typeof options.cache !== 'boolean') ||
    (options.offline !== undefined && typeof options.offline !== 'boolean') ||
    (options.offline && options.cache === false) ||
    (options.maxPrefetchBytes !== undefined && options.maxPrefetchBytes !== 0) ||
    (options.onProgress !== undefined && typeof options.onProgress !== 'function') ||
    !validSignal(options.signal)
  )
    throw fail('INVALID_INPUT', 'Invalid Verdict load option values.');

  let base;
  try {
    base = new URL(assetsUrl, globalThis.location?.href);
  } catch (cause) {
    throw fail('INVALID_INPUT', 'Invalid Verdict assetsUrl.', cause);
  }
  if (!['http:', 'https:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', 'Verdict assetsUrl must be an HTTP or HTTPS directory.');
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
  if (
    config?.model_type !== 'GLiClass' ||
    config?.max_num_classes !== VERDICT_SPEC.maxLabels ||
    config?.class_token_index !== 50368 ||
    config?.text_token_index !== 50369 ||
    config?.vocab_size !== 50370 ||
    config?.encoder_config?.model_type !== 'modernbert' ||
    config?.encoder_config?.max_position_embeddings !== 8192
  )
    throw fail('INVALID_ASSET', 'Verdict config does not match the pinned ModernBERT checkpoint.');
}

function validateBundleManifest(manifest) {
  if (
    manifest?.bundle_version !== '1.0.0' ||
    manifest?.model_architecture !== 'knowledgator/gliclass-modern-base-v2.0' ||
    manifest?.opset_version !== 17 ||
    manifest?.max_capacity_logits !== VERDICT_SPEC.maxLabels
  )
    throw fail('INVALID_ASSET', 'Verdict bundle manifest does not match the pinned model.');
}

function validateCalibrator(calibrator) {
  if (
    calibrator?.model_id !== 'openjev-modernbert-151.4m' ||
    calibrator?.format_version !== 'rlcd-calibrator-v1' ||
    calibrator?.temperature !== VERDICT_GLOBAL_TEMPERATURE ||
    !calibrator.per_k ||
    Object.entries(VERDICT_TEMPERATURES).some(([key, value]) => calibrator.per_k[key] !== value) ||
    Object.keys(calibrator.per_k).some((key) => !Object.hasOwn(VERDICT_TEMPERATURES, key))
  )
    throw fail('INVALID_ASSET', 'Verdict calibrator does not match the pinned model.');
  return calibrator;
}

function validateSession(session) {
  const inputs = new Set(session?.inputNames ?? []);
  const outputs = new Set(session?.outputNames ?? []);
  if (
    !VERDICT_INPUT_NAMES.every((name) => inputs.has(name)) ||
    !VERDICT_OUTPUT_NAMES.every((name) => outputs.has(name))
  )
    throw fail('INVALID_ASSET', 'Verdict ONNX graph has an unexpected interface.');
  return session;
}

async function getHardwareAdapter(gpu, signal) {
  if (typeof gpu?.requestAdapter !== 'function') throw fail('UNSUPPORTED_ENVIRONMENT', 'Verdict requires WebGPU.');
  let adapter;
  try {
    adapter = await waitForAbort(
      () => gpu.requestAdapter({powerPreference: 'high-performance', forceFallbackAdapter: false}),
      signal
    );
  } catch (cause) {
    signal?.throwIfAborted();
    throw fail('MODEL_LOAD_FAILED', 'Could not initialize a hardware WebGPU adapter for Verdict.', cause);
  }
  if (!adapter) throw fail('UNSUPPORTED_ENVIRONMENT', 'No hardware WebGPU adapter is available for Verdict.');
  if (adapter.info?.isFallbackAdapter === true || adapter.isFallbackAdapter === true)
    throw fail('UNSUPPORTED_ENVIRONMENT', 'Verdict does not run on a fallback WebGPU adapter.');
  return adapter;
}

async function createSession(ort, adapter, graphBytes, signal) {
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
        graphOptimizationLevel: 'all'
      }),
    signal,
    (lateSession) => lateSession?.release()
  );
}

export async function loadVerdictModel(
  options,
  {ort, Tokenizer: TokenizerClass = Tokenizer, openSource = openVerdictSource, gpuApi = globalThis.navigator?.gpu} = {}
) {
  const checked = validateVerdictLoadOptions(options);
  const {signal} = checked;
  let session;
  try {
    const runtime = ort ?? (await import('onnxruntime-web/webgpu'));
    signal?.throwIfAborted();
    const source = await openSource({...checked, spec: VERDICT_SPEC});
    const [configBytes, tokenizerConfigBytes, tokenizerBytes, calibratorBytes, bundleBytes] = await Promise.all([
      source.read('config.json'),
      source.read('tokenizer_config.json'),
      source.read('tokenizer.json'),
      source.read('calibrator.json'),
      source.read('bundle_manifest.json')
    ]);
    const config = parseJson(configBytes, 'config.json');
    const tokenizerConfig = parseJson(tokenizerConfigBytes, 'tokenizer_config.json');
    const tokenizerData = parseJson(tokenizerBytes, 'tokenizer.json');
    const calibrator = validateCalibrator(parseJson(calibratorBytes, 'calibrator.json'));
    validateModelConfig(config);
    validateBundleManifest(parseJson(bundleBytes, 'bundle_manifest.json'));
    const tokenizer = createVerdictTokenizer(TokenizerClass, tokenizerData, tokenizerConfig);
    signal?.throwIfAborted();

    const graphBytes = await source.read('model_fp16.onnx');
    signal?.throwIfAborted();
    const adapter = await getHardwareAdapter(gpuApi, signal);
    session = await createSession(ort ?? runtime, adapter, graphBytes, signal);
    validateSession(session);
    signal?.throwIfAborted();
    const runner = createVerdictRunner(ort ?? runtime, session, tokenizer.special.pad);
    const model = createVerdictSession({runner, tokenizer, calibrator, model: VERDICT_MODEL});
    session = null;
    return model;
  } catch (cause) {
    await Promise.allSettled([session?.release()].filter(Boolean));
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
    throw fail('MODEL_LOAD_FAILED', 'Could not load Verdict 151M FP16 on ONNX Runtime WebGPU.', cause);
  }
}
