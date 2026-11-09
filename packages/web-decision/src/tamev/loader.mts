import {Tokenizer} from '@huggingface/tokenizers';
import type {ModelProgress} from '../curated/types.js';
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

type TamevProfile = (typeof TAMEV_PROFILES)[keyof typeof TAMEV_PROFILES];
type TamevSignal = Pick<AbortSignal, 'throwIfAborted' | 'addEventListener' | 'removeEventListener'>;
type TamevErrorCode = 'INVALID_INPUT' | 'INVALID_ASSET' | 'UNSUPPORTED_ENVIRONMENT' | 'MODEL_LOAD_FAILED';
type TamevOptionValues = Record<string, unknown>;
type TamevSourceOpener = (checked: TamevCheckedLoadOptions, profile: TamevProfile) => Promise<TamevAssetSource>;
type TamevRuntimeSessionFactory = (components: {
  readonly runner: ReturnType<typeof createTamevRunner>;
  readonly tokenizer: ReturnType<typeof createTamevTokenizer>;
  readonly config: TamevValidatedConfig;
  readonly profile: TamevProfile;
  readonly model: string;
}) => unknown;
type TamevWaitForAbort = <T>(
  operation: () => Promise<T>,
  signal: TamevSignal | null | undefined,
  cleanup?: ((value: T) => unknown) | undefined
) => Promise<T>;

interface TamevSignalCandidate {
  readonly throwIfAborted?: unknown;
  readonly addEventListener?: unknown;
  readonly removeEventListener?: unknown;
}

interface TamevCheckedLoadOptions {
  readonly base: URL;
  readonly signal: TamevSignal | null | undefined;
  readonly cacheEnabled: boolean;
  readonly offline: boolean;
  readonly onProgress: ((progress: ModelProgress) => void) | undefined;
}

interface TamevAssetSource {
  read(name: string): Promise<ArrayBuffer>;
}

interface TamevOnnxSession {
  readonly inputNames?: readonly string[] | null;
  readonly outputNames?: readonly string[] | null;
  release(): void | Promise<unknown>;
}

interface TamevOrtCandidate {
  readonly InferenceSession?: {
    readonly create?: (
      model: Uint8Array,
      options: {
        readonly executionProviders: ['wasm'];
        readonly intraOpNumThreads: 1;
        readonly interOpNumThreads: 1;
        readonly graphOptimizationLevel: 'all';
      }
    ) => Promise<TamevOnnxSession>;
  } | null;
}

interface TamevConfigCandidate {
  readonly calibration_by_k?: unknown;
  readonly [key: string]: unknown;
}

interface TamevValidatedConfig extends Record<string, unknown> {
  readonly calibration_by_k: Record<string, number>;
}

interface TamevLoaderDependencies {
  readonly publicModelId?: unknown;
  readonly ort?: unknown;
  readonly Tokenizer?: unknown;
  readonly openSource?: unknown;
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
const ID_ONLY_MODELS = new Set([TAMEV_MICRO_MODEL, TAMEV_NANO_MODEL, TAMEV_NANO_FLASH_MODEL]);
const fail = (code: TamevErrorCode, message: string, cause?: unknown): WebDecisionError =>
  new WebDecisionError(code, message, cause ? {cause} : undefined);

function assertLoadOptionDescriptors(options: object, profile: TamevProfile): void {
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (typeof key !== 'string' || !LOAD_KEYS.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw fail('INVALID_INPUT', `Unsupported ${profile.name} load option.`);
  }
}

function snapshotLoadOptionValues(options: object): TamevOptionValues {
  return Object.fromEntries(
    Reflect.ownKeys(options).map((key) => [key, (options as Record<PropertyKey, unknown>)[key]])
  );
}

function hasModelMismatch(values: TamevOptionValues, profile: TamevProfile): boolean {
  return values.model !== undefined && values.model !== profile.model;
}

function hasInvalidAssetsUrl(values: TamevOptionValues): boolean {
  return values.assetsUrl !== undefined && (typeof values.assetsUrl !== 'string' || !values.assetsUrl);
}

function requiresExplicitAssetsUrl(values: TamevOptionValues, profile: TamevProfile): boolean {
  return values.assetsUrl === undefined && !ID_ONLY_MODELS.has(profile.model);
}

function hasInvalidCacheType(values: TamevOptionValues): boolean {
  return values.cache !== undefined && typeof values.cache !== 'boolean';
}

function hasInvalidOfflineType(values: TamevOptionValues): boolean {
  return values.offline !== undefined && typeof values.offline !== 'boolean';
}

function disablesCacheWhileOffline(values: TamevOptionValues): boolean {
  return values.offline === true && values.cache === false;
}

function hasInvalidPrefetchLimit(values: TamevOptionValues): boolean {
  return values.maxPrefetchBytes !== undefined && values.maxPrefetchBytes !== 0;
}

function requestsDebertaReadback(values: TamevOptionValues): boolean {
  return values.debertaReadback !== undefined;
}

function hasInvalidProgressCallback(values: TamevOptionValues): boolean {
  return values.onProgress !== undefined && typeof values.onProgress !== 'function';
}

function hasInvalidSignal(values: TamevOptionValues): boolean {
  return (
    values.signal != null &&
    (typeof (values.signal as TamevSignalCandidate).throwIfAborted !== 'function' ||
      typeof (values.signal as TamevSignalCandidate).addEventListener !== 'function' ||
      typeof (values.signal as TamevSignalCandidate).removeEventListener !== 'function')
  );
}

function rejectInvalidLoadOptionValues(values: TamevOptionValues, profile: TamevProfile): void {
  if (hasModelMismatch(values, profile)) throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
  if (hasInvalidAssetsUrl(values)) throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
  if (requiresExplicitAssetsUrl(values, profile))
    throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
  if (hasInvalidCacheType(values)) throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
  if (hasInvalidOfflineType(values)) throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
  if (disablesCacheWhileOffline(values)) throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
  if (hasInvalidPrefetchLimit(values)) throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
  if (requestsDebertaReadback(values)) throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
  if (hasInvalidProgressCallback(values)) throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
  if (hasInvalidSignal(values)) throw fail('INVALID_INPUT', `Invalid ${profile.name} load option values.`);
}

function resolveTamevBase(values: TamevOptionValues, profile: TamevProfile): URL {
  let base: URL;
  try {
    const sourceUrl =
      (values.assetsUrl as string | undefined) ??
      `https://huggingface.co/${profile.repository}/resolve/${profile.revision}/`;
    base = new URL(sourceUrl, globalThis.location?.href);
  } catch (cause) {
    throw fail('INVALID_INPUT', `Invalid ${profile.name} assetsUrl.`, cause);
  }
  if (!['https:', 'http:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', `${profile.name} assetsUrl must be an HTTP or HTTPS directory.`);
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return base;
}

export function validateTamevLoadOptions(
  options: unknown,
  profile: TamevProfile = TAMEV_MICRO_PROFILE
): TamevCheckedLoadOptions {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', `${profile.name} load options are required.`);
  assertLoadOptionDescriptors(options, profile);
  const values = snapshotLoadOptionValues(options);
  rejectInvalidLoadOptionValues(values, profile);
  const base = resolveTamevBase(values, profile);
  return {
    base,
    signal: values.signal as TamevSignal | null | undefined,
    cacheEnabled: values.cache !== false,
    offline: values.offline === true,
    onProgress: values.onProgress as ((progress: ModelProgress) => void) | undefined
  };
}

function validateConfig(config: unknown, profile: TamevProfile): TamevValidatedConfig {
  const calibration = (config as TamevConfigCandidate | null | undefined)?.calibration_by_k;
  if (
    Object.entries(profile.expectedConfig).some(
      ([key, value]) => (config as TamevConfigCandidate | null | undefined)?.[key] !== value
    ) ||
    !calibration ||
    Object.keys(profile.calibrationByK).some(
      (key) => (calibration as Record<string, unknown>)[key] !== (profile.calibrationByK as Record<string, number>)[key]
    ) ||
    Object.keys(calibration as object).some((key) => !Object.hasOwn(profile.calibrationByK, key))
  )
    throw fail('INVALID_ASSET', `${profile.name} configuration does not match the pinned model.`);
  return config as TamevValidatedConfig;
}

async function createSession(
  ort: unknown,
  bytes: ArrayBuffer,
  signal: TamevSignal | null | undefined
): Promise<TamevOnnxSession> {
  if (typeof (ort as TamevOrtCandidate | null | undefined)?.InferenceSession?.create !== 'function')
    throw fail('UNSUPPORTED_ENVIRONMENT', 'ONNX Runtime Web is unavailable.');
  return (waitForAbort as TamevWaitForAbort)(
    () =>
      (ort as TamevOrtCandidate).InferenceSession!.create!(new Uint8Array(bytes), {
        executionProviders: ['wasm'],
        intraOpNumThreads: 1,
        interOpNumThreads: 1,
        graphOptimizationLevel: 'all'
      }),
    signal,
    (lateSession) => lateSession?.release()
  );
}

function validateSession(session: TamevOnnxSession, profile: TamevProfile): TamevOnnxSession {
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
  options: unknown,
  profile: TamevProfile,
  {
    publicModelId = profile.model,
    ort = defaultOrt,
    Tokenizer: TokenizerClass = Tokenizer,
    openSource = openTamevSource
  }: TamevLoaderDependencies = {}
): Promise<unknown> {
  const checked = validateTamevLoadOptions(options, profile);
  const {signal} = checked;
  let session: TamevOnnxSession | null | undefined;
  try {
    const source = await (openSource as TamevSourceOpener)(checked, profile);
    const [tokenizerBytes, tokenizerConfigBytes, configBytes] = await Promise.all([
      source.read('tokenizer.json'),
      source.read('tokenizer_config.json'),
      source.read('config.json')
    ]);
    const tokenizer = (
      createTamevTokenizer as (
        TokenizerClass: typeof Tokenizer,
        data: object,
        tokenizerConfig: object,
        profile: TamevProfile
      ) => ReturnType<typeof createTamevTokenizer>
    )(
      TokenizerClass as typeof Tokenizer,
      parseJson(tokenizerBytes, 'tokenizer.json') as object,
      parseJson(tokenizerConfigBytes, 'tokenizer_config.json') as object,
      profile
    );
    const config = validateConfig(parseJson(configBytes, 'config.json'), profile);
    signal?.throwIfAborted();
    const modelBytes = await source.read('model_int8.onnx');
    session = await createSession(ort, modelBytes, signal);
    validateSession(session, profile);
    signal?.throwIfAborted();
    const runner = createTamevRunner(ort, session, tokenizer);
    const model = (createTamevSession as unknown as TamevRuntimeSessionFactory)({
      runner,
      tokenizer,
      config,
      profile,
      model: profile.model
    });
    session = null;
    return model;
  } catch (cause) {
    await Promise.allSettled([session?.release()].filter(Boolean));
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || (cause as {name?: unknown} | null | undefined)?.name === 'AbortError')
      throw cause;
    throw fail('MODEL_LOAD_FAILED', `Could not load ${publicModelId}.`, cause);
  }
}

export function loadTamevMicro(options: unknown, dependencies?: TamevLoaderDependencies): Promise<unknown> {
  return loadTamevProfile(options, TAMEV_MICRO_PROFILE, dependencies);
}

export function loadTamevNano(options: unknown, dependencies?: TamevLoaderDependencies): Promise<unknown> {
  return loadTamevProfile(options, TAMEV_PROFILES['experimental-tamev-nano-tinybert-int8'], dependencies);
}

export function loadTamev(options: unknown, dependencies?: TamevLoaderDependencies): Promise<unknown> {
  const profile =
    TAMEV_PROFILES[
      ((options as {readonly model?: string} | null | undefined)?.model ??
        TAMEV_MICRO_MODEL) as keyof typeof TAMEV_PROFILES
    ];
  if (!profile) throw fail('INVALID_INPUT', 'Unsupported TAMEV model.');
  return loadTamevProfile(options, profile, dependencies);
}
