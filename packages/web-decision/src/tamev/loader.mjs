import {Tokenizer} from '@huggingface/tokenizers';
import * as defaultOrt from 'onnxruntime-web/wasm';
import {parseJson, waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {
  TAMEV_MICRO_PROFILE,
  TAMEV_MICRO_MODEL,
  TAMEV_NANO_MODEL,
  TAMEV_NANO_FLASH_MODEL,
  TAMEV_PROFILES
} from './contract.mjs';
import {createTamevRunner} from './runner.mjs';
import {createTamevSession} from './session.mjs';
import {openTamevSource} from './source.mjs';
import {createTamevTokenizer} from './tokenizer.mjs';

const LOAD_KEYS = new Set([
  'model',
  'assetsUrl',
  'cache',
  'offline',
  'signal',
  'onProgress',
  'maxPrefetchBytes',
  'debertaReadback'
]);
const ID_ONLY_MODELS = new Set([TAMEV_MICRO_MODEL, TAMEV_NANO_MODEL, TAMEV_NANO_FLASH_MODEL]);
const fail = (code, message, cause) => new WebDecisionError(code, message, cause ? {cause} : undefined);

export function validateTamevLoadOptions(options, profile = TAMEV_MICRO_PROFILE) {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', `${profile.name} load options are required.`);
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (typeof key !== 'string' || !LOAD_KEYS.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw fail('INVALID_INPUT', `Unsupported ${profile.name} load option.`);
  }
  const values = Object.fromEntries(Reflect.ownKeys(options).map((key) => [key, options[key]]));
  if (
    (values.model !== undefined && values.model !== profile.model) ||
    (values.assetsUrl !== undefined && (typeof values.assetsUrl !== 'string' || !values.assetsUrl)) ||
    (values.assetsUrl === undefined && !ID_ONLY_MODELS.has(profile.model)) ||
    (values.cache !== undefined && typeof values.cache !== 'boolean') ||
    (values.offline !== undefined && typeof values.offline !== 'boolean') ||
    (values.offline && values.cache === false) ||
    (values.maxPrefetchBytes !== undefined && values.maxPrefetchBytes !== 0) ||
    values.debertaReadback !== undefined ||
    (values.onProgress !== undefined && typeof values.onProgress !== 'function') ||
    (values.signal != null &&
      (typeof values.signal.throwIfAborted !== 'function' ||
        typeof values.signal.addEventListener !== 'function' ||
        typeof values.signal.removeEventListener !== 'function'))
  )
    throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
  let base;
  try {
    const sourceUrl = values.assetsUrl ?? `https://huggingface.co/${profile.repository}/resolve/${profile.revision}/`;
    base = new URL(sourceUrl, globalThis.location?.href);
  } catch (cause) {
    throw fail('INVALID_INPUT', `Invalid ${profile.name} assetsUrl.`, cause);
  }
  if (!['https:', 'http:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', `${profile.name} assetsUrl must be an HTTP or HTTPS directory.`);
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return {
    base,
    signal: values.signal,
    cacheEnabled: values.cache !== false,
    offline: values.offline === true,
    onProgress: values.onProgress
  };
}

function validateConfig(config, profile) {
  const calibration = config?.calibration_by_k;
  if (
    Object.entries(profile.expectedConfig).some(([key, value]) => config?.[key] !== value) ||
    !calibration ||
    Object.keys(profile.calibrationByK).some((key) => calibration[key] !== profile.calibrationByK[key]) ||
    Object.keys(calibration).some((key) => !Object.hasOwn(profile.calibrationByK, key))
  )
    throw fail('INVALID_ASSET', `${profile.name} configuration does not match the pinned model.`);
  return config;
}

async function createSession(ort, bytes, signal) {
  if (typeof ort?.InferenceSession?.create !== 'function')
    throw fail('UNSUPPORTED_ENVIRONMENT', 'ONNX Runtime Web is unavailable.');
  return waitForAbort(
    () =>
      ort.InferenceSession.create(new Uint8Array(bytes), {
        executionProviders: ['wasm'],
        intraOpNumThreads: 1,
        interOpNumThreads: 1,
        graphOptimizationLevel: 'all'
      }),
    signal,
    (lateSession) => lateSession?.release()
  );
}

function validateSession(session, profile) {
  const inputs = new Set(session?.inputNames ?? []);
  const outputs = new Set(session?.outputNames ?? []);
  if (
    !['ctx_input_ids', 'ctx_attention_mask', 'opt_input_ids', 'opt_attention_mask'].every((name) => inputs.has(name)) ||
    !['logits', 'probs'].every((name) => outputs.has(name))
  )
    throw fail('INVALID_ASSET', `${profile.name} ONNX graph has an unexpected interface.`);
  return session;
}

async function loadTamevProfile(
  options,
  profile,
  {
    publicModelId = profile.model,
    ort = defaultOrt,
    Tokenizer: TokenizerClass = Tokenizer,
    openSource = openTamevSource
  } = {}
) {
  const checked = validateTamevLoadOptions(options, profile);
  const {signal} = checked;
  let session;
  try {
    const source = await openSource(checked, profile);
    const [tokenizerBytes, tokenizerConfigBytes, configBytes] = await Promise.all([
      source.read('tokenizer.json'),
      source.read('tokenizer_config.json'),
      source.read('config.json')
    ]);
    const tokenizer = createTamevTokenizer(
      TokenizerClass,
      parseJson(tokenizerBytes, 'tokenizer.json'),
      parseJson(tokenizerConfigBytes, 'tokenizer_config.json'),
      profile
    );
    const config = validateConfig(parseJson(configBytes, 'config.json'), profile);
    signal?.throwIfAborted();
    const modelBytes = await source.read('model_int8.onnx');
    session = await createSession(ort, modelBytes, signal);
    validateSession(session, profile);
    signal?.throwIfAborted();
    const runner = createTamevRunner(ort, session, tokenizer);
    const model = createTamevSession({runner, tokenizer, config, profile, model: profile.model});
    session = null;
    return model;
  } catch (cause) {
    await Promise.allSettled([session?.release()].filter(Boolean));
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
    throw fail('MODEL_LOAD_FAILED', `Could not load ${publicModelId}.`, cause);
  }
}

export function loadTamevMicro(options, dependencies) {
  return loadTamevProfile(options, TAMEV_MICRO_PROFILE, dependencies);
}

export function loadTamevNano(options, dependencies) {
  return loadTamevProfile(options, TAMEV_PROFILES['experimental-tamev-nano-tinybert-int8'], dependencies);
}

export function loadTamev(options, dependencies) {
  const profile = TAMEV_PROFILES[options?.model ?? TAMEV_MICRO_MODEL];
  if (!profile) throw fail('INVALID_INPUT', 'Unsupported TAMEV model.');
  return loadTamevProfile(options, profile, dependencies);
}
