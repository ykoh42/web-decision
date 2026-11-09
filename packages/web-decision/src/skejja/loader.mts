import {Tokenizer} from '@huggingface/tokenizers';
import type {Tokenizer as TokenizerImplementation} from '@huggingface/tokenizers';
import * as defaultOrt from 'onnxruntime-web/wasm';
import type {InferenceSession} from 'onnxruntime-web';
import {parseJson, waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {
  SKEJJA_INT8_PROFILE,
  SKEJJA_INT8DYN_PROFILE,
  SKEJJA_PROFILES,
  SKEJJA_REPOSITORY,
  SKEJJA_REVISION
} from './contract.mjs';
import {createSkejjaRunner} from './runner.mjs';
import {createSkejjaSession, type SkejjaSession} from './session.mjs';
import {openSkejjaSource} from './source.mjs';
import {createSkejjaTokenizer, type SkejjaTokenizer} from './tokenizer.mjs';

type SkejjaProfile = typeof SKEJJA_INT8_PROFILE | typeof SKEJJA_INT8DYN_PROFILE;
type SkejjaSignal = Pick<AbortSignal, 'throwIfAborted' | 'addEventListener' | 'removeEventListener'> &
  Partial<Pick<AbortSignal, 'aborted' | 'reason'>>;
type SkejjaOrt = typeof defaultOrt;
type SkejjaProgressCallback = (progress: {
  readonly loadedBytes: number;
  readonly totalBytes: number;
  readonly fraction: number;
}) => void;

interface SkejjaLoadOptionValues {
  readonly model?: unknown;
  readonly assetsUrl?: unknown;
  readonly cache?: unknown;
  readonly offline?: unknown;
  readonly signal?: unknown;
  readonly onProgress?: unknown;
  readonly maxPrefetchBytes?: unknown;
  readonly debertaReadback?: unknown;
}

interface SkejjaCheckedLoadOptions {
  readonly base: URL;
  readonly useArchive: boolean;
  readonly signal: SkejjaSignal | null | undefined;
  readonly cacheEnabled: boolean;
  readonly offline: boolean;
  readonly onProgress: SkejjaProgressCallback | undefined;
}

interface SkejjaSource {
  read(name: string): Promise<ArrayBuffer>;
  release?(names: readonly string[]): void | Promise<void>;
}

type SkejjaTokenizerConstructor = new (data: object, config: object) => TokenizerImplementation;
type SkejjaOpenSource = (checked: SkejjaCheckedLoadOptions, profile: SkejjaProfile) => Promise<SkejjaSource>;

interface SkejjaLoadDependencies {
  readonly ort?: SkejjaOrt;
  readonly Tokenizer?: SkejjaTokenizerConstructor;
  readonly openSource?: SkejjaOpenSource;
}

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
type WebDecisionErrorCode = ConstructorParameters<typeof WebDecisionError>[0];

const fail = (code: WebDecisionErrorCode, message: string, cause?: unknown): WebDecisionError =>
  new WebDecisionError(code, message, cause ? {cause} : undefined);

function assertLoadOptionsObject(options: unknown, profile: SkejjaProfile): asserts options is object {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', `${profile.name} load options are required.`);
}

function hasSupportedDataProperty(options: object, key: string | symbol): boolean | PropertyDescriptor | undefined {
  const descriptor = Object.getOwnPropertyDescriptor(options, key);
  return typeof key === 'string' && LOAD_KEYS.has(key) && descriptor && Object.hasOwn(descriptor, 'value');
}

function readLoadOptionValues(options: object, profile: SkejjaProfile): SkejjaLoadOptionValues {
  for (const key of Reflect.ownKeys(options)) {
    if (!hasSupportedDataProperty(options, key))
      throw fail('INVALID_INPUT', `Unsupported ${profile.name} load option.`);
  }
  const optionRecord = options as Record<PropertyKey, unknown>;
  return Object.fromEntries(Reflect.ownKeys(options).map((key) => [key, optionRecord[key]])) as SkejjaLoadOptionValues;
}

function assertModelAndAssetOptions(values: SkejjaLoadOptionValues, profile: SkejjaProfile): void {
  if (values.model !== profile.model) throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
  if (values.assetsUrl !== undefined && (typeof values.assetsUrl !== 'string' || !values.assetsUrl))
    throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
}

function assertValidCacheOptions(values: SkejjaLoadOptionValues, profile: SkejjaProfile): void {
  if (values.cache !== undefined && typeof values.cache !== 'boolean')
    throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
  if (values.offline !== undefined && typeof values.offline !== 'boolean')
    throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
  if (values.offline && values.cache === false)
    throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
}

function isValidSignal(signal: unknown): signal is SkejjaSignal | null | undefined {
  const signalLike = signal as SkejjaSignal | null | undefined;
  return (
    signalLike == null ||
    (typeof signalLike.throwIfAborted === 'function' &&
      typeof signalLike.addEventListener === 'function' &&
      typeof signalLike.removeEventListener === 'function')
  );
}

function assertValidLoadControls(values: SkejjaLoadOptionValues, profile: SkejjaProfile): void {
  if (values.maxPrefetchBytes !== undefined && values.maxPrefetchBytes !== 0)
    throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
  if (values.debertaReadback !== undefined) throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
  if (values.onProgress !== undefined && typeof values.onProgress !== 'function')
    throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
  if (!isValidSignal(values.signal)) throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
}

function assertValidLoadOptionValues(values: SkejjaLoadOptionValues, profile: SkejjaProfile): void {
  assertModelAndAssetOptions(values, profile);
  assertValidCacheOptions(values, profile);
  assertValidLoadControls(values, profile);
}

function resolveAssetsBase(values: SkejjaLoadOptionValues, profile: SkejjaProfile): {base: URL; useArchive: boolean} {
  const useArchive = values.assetsUrl === undefined;
  if (useArchive) {
    return {
      base: new URL(`https://huggingface.co/${SKEJJA_REPOSITORY}/resolve/${SKEJJA_REVISION}/`),
      useArchive
    };
  }

  let base;
  try {
    base = new URL(values.assetsUrl as string, globalThis.location?.href);
  } catch (cause) {
    throw fail('INVALID_INPUT', `Invalid ${profile.name} assetsUrl.`, cause);
  }
  if (!['https:', 'http:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', `${profile.name} assetsUrl must be an HTTP or HTTPS directory.`);
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return {base, useArchive};
}

export function validateSkejjaLoadOptions(
  options: unknown,
  profile: SkejjaProfile = SKEJJA_INT8_PROFILE
): SkejjaCheckedLoadOptions {
  assertLoadOptionsObject(options, profile);
  const values = readLoadOptionValues(options, profile);
  assertValidLoadOptionValues(values, profile);
  const {base, useArchive} = resolveAssetsBase(values, profile);
  return {
    base,
    useArchive,
    signal: values.signal as SkejjaSignal | null | undefined,
    cacheEnabled: values.cache !== false,
    offline: values.offline === true,
    onProgress: values.onProgress as SkejjaProgressCallback | undefined
  };
}

function validateCalibration(calibration: unknown, profile: SkejjaProfile): void {
  const value = (calibration as Record<string, unknown> | null | undefined)?.[profile.onnxFile] as
    | {readonly temperature?: unknown; readonly platt_a?: unknown; readonly platt_b?: unknown}
    | null
    | undefined;
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

function validateSession(session: InferenceSession, profile: SkejjaProfile): InferenceSession {
  const inputs = new Set(session?.inputNames ?? []);
  const outputs = new Set(session?.outputNames ?? []);
  if (!['input_ids', 'attention_mask', 'label_slots'].every((name) => inputs.has(name)) || !outputs.has('logits'))
    throw fail('INVALID_ASSET', `${profile.name} ONNX graph has an unexpected interface.`);
  return session;
}

async function createSession(
  ort: SkejjaOrt | null | undefined,
  graphBytes: ArrayBuffer,
  dataBytes: ArrayBuffer,
  profile: SkejjaProfile,
  signal: SkejjaSignal | null | undefined
): Promise<InferenceSession> {
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
    signal as AbortSignal | null | undefined,
    (lateSession) => lateSession?.release()
  );
}

async function loadSkejjaProfile(
  options: unknown,
  profile: SkejjaProfile,
  {
    ort = defaultOrt,
    Tokenizer: TokenizerClass = Tokenizer,
    openSource = openSkejjaSource as SkejjaOpenSource
  }: SkejjaLoadDependencies = {}
): Promise<SkejjaSession> {
  const checked = validateSkejjaLoadOptions(options, profile);
  const {signal} = checked;
  let session: InferenceSession | null | undefined;
  try {
    const source = await openSource(checked, profile);
    let tokenizer: SkejjaTokenizer;
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
    if (
      cause instanceof WebDecisionError ||
      (cause as {readonly name?: unknown} | null | undefined)?.name === 'AbortError'
    )
      throw cause;
    throw fail('MODEL_LOAD_FAILED', `Could not load ${profile.model}.`, cause);
  }
}

export function loadSkejja(options: unknown, dependencies?: SkejjaLoadDependencies): Promise<SkejjaSession> {
  const profile =
    SKEJJA_PROFILES[(options as {readonly model?: unknown} | null | undefined)?.model as keyof typeof SKEJJA_PROFILES];
  if (!profile) throw fail('INVALID_INPUT', 'Unsupported Skejja model.');
  return loadSkejjaProfile(options, profile, dependencies);
}
