import {parseJson} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {TinyDecide} from './engine.js';
import {TINYDECIDE_MODEL, TINYDECIDE_REPOSITORY, TINYDECIDE_REVISION} from './contract.mjs';
import {createTinyDecideSession} from './session.mjs';
import {openTinyDecideSource} from './source.mjs';

const LOAD_KEYS = new Set(['model', 'assetsUrl', 'cache', 'offline', 'signal', 'onProgress']);
const fail = (code, message, cause) => new WebDecisionError(code, message, cause ? {cause} : undefined);

export function validateTinyDecideLoadOptions(options) {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', 'TinyDecide load options are required.');
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (typeof key !== 'string' || !LOAD_KEYS.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw fail('INVALID_INPUT', 'Unsupported TinyDecide load option.');
  }
  if (
    options.model !== TINYDECIDE_MODEL ||
    (options.assetsUrl !== undefined && (typeof options.assetsUrl !== 'string' || !options.assetsUrl)) ||
    (options.cache !== undefined && typeof options.cache !== 'boolean') ||
    (options.offline !== undefined && typeof options.offline !== 'boolean') ||
    (options.offline && options.cache === false) ||
    (options.onProgress !== undefined && typeof options.onProgress !== 'function') ||
    (options.signal != null && typeof options.signal.throwIfAborted !== 'function')
  )
    throw fail('INVALID_INPUT', 'Invalid TinyDecide load option values.');
  let base;
  try {
    const sourceUrl =
      options.assetsUrl ?? `https://huggingface.co/${TINYDECIDE_REPOSITORY}/resolve/${TINYDECIDE_REVISION}/`;
    base = new URL(sourceUrl, globalThis.location?.href);
  } catch (cause) {
    throw fail('INVALID_INPUT', 'Invalid TinyDecide assetsUrl.', cause);
  }
  if (!['https:', 'http:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', 'TinyDecide assetsUrl must be an HTTP or HTTPS directory.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return {
    base,
    signal: options.signal,
    cacheEnabled: options.cache !== false,
    offline: options.offline === true,
    onProgress: options.onProgress
  };
}

function validateMetadata(meta) {
  if (
    meta?.name !== 'TinyDecide' ||
    meta.cfg?.arch !== 'electra' ||
    meta.cfg?.d !== 256 ||
    meta.cfg?.layers !== 12 ||
    meta.cfg?.dh_head !== 128 ||
    meta.format?.ts_max !== 128 ||
    meta.format?.q_max !== 192 ||
    meta.format?.k_max !== 32 ||
    meta.format?.span_max !== 16 ||
    meta.tokenizer?.kind !== 'wordpiece' ||
    meta.tokenizer?.vocab?.length !== 16000 ||
    !Array.isArray(meta.tensors) ||
    meta.tensors.length !== 219 ||
    !Array.isArray(meta.temp) ||
    meta.temp.length !== 4 ||
    meta.temp.some((value) => !Number.isFinite(value) || value <= 0) ||
    !Array.isArray(meta.beta) ||
    meta.beta.length !== 5 ||
    meta.beta.some((value) => !Number.isFinite(value) || value < 0)
  )
    throw fail('INVALID_ASSET', 'TinyDecide metadata does not match the pinned checkpoint.');
  return meta;
}

export async function loadTinyDecide(options, {TinyDecideClass = TinyDecide, openSource = openTinyDecideSource} = {}) {
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
  } catch (cause) {
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
    throw fail('MODEL_LOAD_FAILED', 'Could not load TinyDecide.', cause);
  }
}
