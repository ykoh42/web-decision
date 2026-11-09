import {parseJson} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {TinyDecide, type TinyDecideMetadata} from './engine.js';
import {TINYDECIDE_MODEL, TINYDECIDE_REPOSITORY, TINYDECIDE_REVISION} from './contract.mjs';
import {createTinyDecideSession} from './session.mjs';
import {openTinyDecideSource} from './source.mjs';
import type {TinyDecideSession} from './session.mjs';
import type {TinyDecideAssetSource, TinyDecideProgress, TinyDecideSourceOptions} from './source-types.mjs';

type WebDecisionErrorCode = ConstructorParameters<typeof WebDecisionError>[0];

export interface TinyDecideLoadProgress extends TinyDecideProgress {}

export interface TinyDecideLoadOptions {
  readonly model: string;
  readonly assetsUrl?: string | undefined;
  readonly cache?: boolean | undefined;
  readonly offline?: boolean | undefined;
  readonly signal?: AbortSignal | undefined;
  readonly onProgress?: ((progress: TinyDecideLoadProgress) => void) | undefined;
}

export interface TinyDecideLoaderEngine {
  readonly meta: TinyDecideMetadata;
  readonly cfg: TinyDecideMetadata['cfg'];
  readonly tok: TinyDecide['tok'];
  answer: TinyDecide['answer'];
}

export interface TinyDecideLoadDependencies {
  readonly TinyDecideClass?:
    | (new (metadata: TinyDecideMetadata, bytes: ArrayBuffer) => TinyDecideLoaderEngine)
    | undefined;
  readonly openSource?: ((options: TinyDecideSourceOptions) => Promise<TinyDecideAssetSource>) | undefined;
}

interface TinyDecideCheckedLoadOptions {
  readonly base: URL;
  readonly signal: AbortSignal | undefined;
  readonly cacheEnabled: boolean;
  readonly offline: boolean;
  readonly onProgress: ((progress: TinyDecideLoadProgress) => void) | undefined;
}

interface TinyDecidePinnedMetadata extends TinyDecideMetadata {
  readonly name?: unknown;
}

const LOAD_KEYS = new Set(['model', 'assetsUrl', 'cache', 'offline', 'signal', 'onProgress']);
const fail = (code: WebDecisionErrorCode, message: string, cause?: unknown): WebDecisionError =>
  new WebDecisionError(code, message, cause ? {cause} : undefined);

export function validateTinyDecideLoadOptions(options: unknown): TinyDecideCheckedLoadOptions {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', 'TinyDecide load options are required.');
  for (const key of Reflect.ownKeys(options as object)) {
    const descriptor = Object.getOwnPropertyDescriptor(options as object, key);
    if (typeof key !== 'string' || !LOAD_KEYS.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw fail('INVALID_INPUT', 'Unsupported TinyDecide load option.');
  }
  if (
    (options as Record<PropertyKey, unknown>).model !== TINYDECIDE_MODEL ||
    ((options as Record<PropertyKey, unknown>).assetsUrl !== undefined &&
      (typeof (options as Record<PropertyKey, unknown>).assetsUrl !== 'string' ||
        !(options as Record<PropertyKey, unknown>).assetsUrl)) ||
    ((options as Record<PropertyKey, unknown>).cache !== undefined &&
      typeof (options as Record<PropertyKey, unknown>).cache !== 'boolean') ||
    ((options as Record<PropertyKey, unknown>).offline !== undefined &&
      typeof (options as Record<PropertyKey, unknown>).offline !== 'boolean') ||
    ((options as Record<PropertyKey, unknown>).offline && (options as Record<PropertyKey, unknown>).cache === false) ||
    ((options as Record<PropertyKey, unknown>).onProgress !== undefined &&
      typeof (options as Record<PropertyKey, unknown>).onProgress !== 'function') ||
    ((options as Record<PropertyKey, unknown>).signal != null &&
      typeof ((options as Record<PropertyKey, unknown>).signal as AbortSignal).throwIfAborted !== 'function')
  )
    throw fail('INVALID_INPUT', 'Invalid TinyDecide load option values.');
  let base: URL;
  try {
    const sourceUrl =
      (options as Record<PropertyKey, unknown>).assetsUrl ??
      `https://huggingface.co/${TINYDECIDE_REPOSITORY}/resolve/${TINYDECIDE_REVISION}/`;
    base = new URL(sourceUrl as string, globalThis.location?.href);
  } catch (cause: unknown) {
    throw fail('INVALID_INPUT', 'Invalid TinyDecide assetsUrl.', cause);
  }
  if (!['https:', 'http:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', 'TinyDecide assetsUrl must be an HTTP or HTTPS directory.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return {
    base,
    signal: (options as Record<PropertyKey, unknown>).signal as AbortSignal | undefined,
    cacheEnabled: (options as Record<PropertyKey, unknown>).cache !== false,
    offline: (options as Record<PropertyKey, unknown>).offline === true,
    onProgress: (options as Record<PropertyKey, unknown>).onProgress as
      | ((progress: TinyDecideLoadProgress) => void)
      | undefined
  };
}

function validateMetadata(meta: unknown): TinyDecideMetadata {
  if (
    (meta as TinyDecidePinnedMetadata | null | undefined)?.name !== 'TinyDecide' ||
    (meta as TinyDecidePinnedMetadata).cfg?.arch !== 'electra' ||
    (meta as TinyDecidePinnedMetadata).cfg?.d !== 256 ||
    (meta as TinyDecidePinnedMetadata).cfg?.layers !== 12 ||
    (meta as TinyDecidePinnedMetadata).cfg?.dh_head !== 128 ||
    (meta as TinyDecidePinnedMetadata).format?.ts_max !== 128 ||
    (meta as TinyDecidePinnedMetadata).format?.q_max !== 192 ||
    (meta as TinyDecidePinnedMetadata).format?.k_max !== 32 ||
    (meta as TinyDecidePinnedMetadata).format?.span_max !== 16 ||
    (meta as TinyDecidePinnedMetadata).tokenizer?.kind !== 'wordpiece' ||
    (meta as TinyDecidePinnedMetadata).tokenizer?.vocab?.length !== 16000 ||
    !Array.isArray((meta as TinyDecidePinnedMetadata).tensors) ||
    (meta as TinyDecidePinnedMetadata).tensors.length !== 219 ||
    !Array.isArray((meta as TinyDecidePinnedMetadata).temp) ||
    (meta as TinyDecidePinnedMetadata).temp.length !== 4 ||
    (meta as TinyDecidePinnedMetadata).temp.some((value) => !Number.isFinite(value) || value <= 0) ||
    !Array.isArray((meta as TinyDecidePinnedMetadata).beta) ||
    (meta as TinyDecidePinnedMetadata).beta.length !== 5 ||
    (meta as TinyDecidePinnedMetadata).beta.some((value) => !Number.isFinite(value) || value < 0)
  )
    throw fail('INVALID_ASSET', 'TinyDecide metadata does not match the pinned checkpoint.');
  return meta as TinyDecideMetadata;
}

export async function loadTinyDecide(
  options: unknown,
  {TinyDecideClass = TinyDecide, openSource = openTinyDecideSource}: TinyDecideLoadDependencies = {}
): Promise<TinyDecideSession> {
  const checked = validateTinyDecideLoadOptions(options);
  const {signal} = checked;
  try {
    const source = await openSource(checked);
    const [metadataBytes, modelBytes] = await Promise.all([source.read('meta.json'), source.read('model.bin')]);
    const metadata = validateMetadata(parseJson(metadataBytes, 'meta.json'));
    signal?.throwIfAborted();
    const engine = new TinyDecideClass(metadata, modelBytes);
    signal?.throwIfAborted();
    return createTinyDecideSession({engine, model: TINYDECIDE_MODEL});
  } catch (cause: unknown) {
    signal?.throwIfAborted();
    if (
      cause instanceof WebDecisionError ||
      (cause as {readonly name?: unknown} | null | undefined)?.name === 'AbortError'
    )
      throw cause;
    throw fail('MODEL_LOAD_FAILED', 'Could not load TinyDecide.', cause);
  }
}
