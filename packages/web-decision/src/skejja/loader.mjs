import {Tokenizer} from '@huggingface/tokenizers';
import * as defaultOrt from 'onnxruntime-web/wasm';
import {parseJson, waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {SKEJJA_INT8_PROFILE, SKEJJA_PROFILES, SKEJJA_REPOSITORY, SKEJJA_REVISION} from './contract.mjs';
import {createSkejjaRunner} from './runner.mjs';
import {createSkejjaSession} from './session.mjs';
import {openSkejjaSource} from './source.mjs';
import {createSkejjaTokenizer} from './tokenizer.mjs';

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
const fail = (code, message, cause) => new WebDecisionError(code, message, cause ? {cause} : undefined);

function assertLoadOptionsObject(options, profile) {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', `${profile.name} load options are required.`);
}

function hasSupportedDataProperty(options, key) {
  const descriptor = Object.getOwnPropertyDescriptor(options, key);
  return typeof key === 'string' && LOAD_KEYS.has(key) && descriptor && Object.hasOwn(descriptor, 'value');
}

function readLoadOptionValues(options, profile) {
  for (const key of Reflect.ownKeys(options)) {
    if (!hasSupportedDataProperty(options, key))
      throw fail('INVALID_INPUT', `Unsupported ${profile.name} load option.`);
  }
  return Object.fromEntries(Reflect.ownKeys(options).map((key) => [key, options[key]]));
}

function assertModelAndAssetOptions(values, profile) {
  if (values.model !== profile.model) throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
  if (values.assetsUrl !== undefined && (typeof values.assetsUrl !== 'string' || !values.assetsUrl))
    throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
}

function assertValidCacheOptions(values, profile) {
  if (values.cache !== undefined && typeof values.cache !== 'boolean')
    throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
  if (values.offline !== undefined && typeof values.offline !== 'boolean')
    throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
  if (values.offline && values.cache === false)
    throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
}

function isValidSignal(signal) {
  return (
    signal == null ||
    (typeof signal.throwIfAborted === 'function' &&
      typeof signal.addEventListener === 'function' &&
      typeof signal.removeEventListener === 'function')
  );
}

function assertValidLoadControls(values, profile) {
  if (values.maxPrefetchBytes !== undefined && values.maxPrefetchBytes !== 0)
    throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
  if (values.debertaReadback !== undefined) throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
  if (values.onProgress !== undefined && typeof values.onProgress !== 'function')
    throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
  if (!isValidSignal(values.signal)) throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
}

function assertValidLoadOptionValues(values, profile) {
  assertModelAndAssetOptions(values, profile);
  assertValidCacheOptions(values, profile);
  assertValidLoadControls(values, profile);
}

function resolveAssetsBase(values, profile) {
  const useArchive = values.assetsUrl === undefined;
  if (useArchive) {
    return {
      base: new URL(`https://huggingface.co/${SKEJJA_REPOSITORY}/resolve/${SKEJJA_REVISION}/`),
      useArchive
    };
  }

  let base;
  try {
    base = new URL(values.assetsUrl, globalThis.location?.href);
  } catch (cause) {
    throw fail('INVALID_INPUT', `Invalid ${profile.name} assetsUrl.`, cause);
  }
  if (!['https:', 'http:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', `${profile.name} assetsUrl must be an HTTP or HTTPS directory.`);
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return {base, useArchive};
}

export function validateSkejjaLoadOptions(options, profile = SKEJJA_INT8_PROFILE) {
  assertLoadOptionsObject(options, profile);
  const values = readLoadOptionValues(options, profile);
  assertValidLoadOptionValues(values, profile);
  const {base, useArchive} = resolveAssetsBase(values, profile);
  return {
    base,
    useArchive,
    signal: values.signal,
    cacheEnabled: values.cache !== false,
    offline: values.offline === true,
    onProgress: values.onProgress
  };
}

function validateCalibration(calibration, profile) {
  const value = calibration?.[profile.onnxFile];
  if (
    value?.temperature !== profile.temperature ||
    value?.platt_a !== profile.plattA ||
    value?.platt_b !== profile.plattB ||
    !Number.isFinite(value.temperature) ||
    !Number.isFinite(value.platt_a) ||
    !Number.isFinite(value.platt_b)
  )
    throw fail('INVALID_ASSET', `${profile.name} calibration does not match the pinned model.`);
}

function validateSession(session, profile) {
  const inputs = new Set(session?.inputNames ?? []);
  const outputs = new Set(session?.outputNames ?? []);
  if (!['input_ids', 'attention_mask', 'label_slots'].every((name) => inputs.has(name)) || !outputs.has('logits'))
    throw fail('INVALID_ASSET', `${profile.name} ONNX graph has an unexpected interface.`);
  return session;
}

async function createSession(ort, graphBytes, dataBytes, profile, signal) {
  if (typeof ort?.InferenceSession?.create !== 'function')
    throw fail('UNSUPPORTED_ENVIRONMENT', 'ONNX Runtime Web is unavailable.');
  const graph = new Uint8Array(graphBytes);
  const data = new Uint8Array(dataBytes);
  return waitForAbort(
    () =>
      ort.InferenceSession.create(graph, {
        executionProviders: ['wasm'],
        intraOpNumThreads: 1,
        interOpNumThreads: 1,
        graphOptimizationLevel: 'all',
        externalData: [{path: profile.dataFile, data}]
      }),
    signal,
    (lateSession) => lateSession?.release()
  );
}

async function loadSkejjaProfile(
  options,
  profile,
  {ort = defaultOrt, Tokenizer: TokenizerClass = Tokenizer, openSource = openSkejjaSource} = {}
) {
  const checked = validateSkejjaLoadOptions(options, profile);
  const {signal} = checked;
  let session;
  try {
    const source = await openSource(checked, profile);
    let tokenizer;
    {
      const [tokenizerBytes, tokenizerConfigBytes, calibrationBytes] = await Promise.all([
        source.read('tokenizer.json'),
        source.read('tokenizer_config.json'),
        source.read('calibration.json')
      ]);
      tokenizer = createSkejjaTokenizer(
        TokenizerClass,
        parseJson(tokenizerBytes, 'tokenizer.json'),
        parseJson(tokenizerConfigBytes, 'tokenizer_config.json')
      );
      const calibration = parseJson(calibrationBytes, 'calibration.json');
      validateCalibration(calibration, profile);
    }
    await source.release?.(['tokenizer.json', 'tokenizer_config.json', 'calibration.json']);
    signal?.throwIfAborted();

    {
      const [graphBytes, dataBytes] = await Promise.all([source.read(profile.onnxFile), source.read(profile.dataFile)]);
      try {
        session = validateSession(await createSession(ort, graphBytes, dataBytes, profile, signal), profile);
      } finally {
        await source.release?.([profile.onnxFile, profile.dataFile]);
      }
    }
    signal?.throwIfAborted();
    const runner = createSkejjaRunner(ort, session);
    const model = createSkejjaSession({runner, tokenizer, profile});
    session = null;
    return model;
  } catch (cause) {
    await Promise.allSettled([session?.release()].filter(Boolean));
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
    throw fail('MODEL_LOAD_FAILED', `Could not load ${profile.model}.`, cause);
  }
}

export function loadSkejja(options, dependencies) {
  const profile = SKEJJA_PROFILES[options?.model];
  if (!profile) throw fail('INVALID_INPUT', 'Unsupported Skejja model.');
  return loadSkejjaProfile(options, profile, dependencies);
}
