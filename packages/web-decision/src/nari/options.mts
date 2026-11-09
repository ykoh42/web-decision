import {WebDecisionError} from '../kev.js';
import {validateH2oSignal} from '../h2o/signal.mjs';
import {nariVariant} from './variants.mjs';
import type {NARI_FULL, NARI_MIXED} from './variants.mjs';

export {NARI_MODEL_ID, NARI_MIXED_MODEL_ID} from './variants.mjs';
type NariVariant = typeof NARI_FULL | typeof NARI_MIXED;

interface ValidatedNariOptions {
  readonly base: URL;
  readonly signal: AbortSignal | undefined;
  readonly variant: NariVariant;
}

const invalid = (message: string): WebDecisionError => new WebDecisionError('INVALID_INPUT', message);
const allowed = new Set([
  'model',
  'assetsUrl',
  'signal',
  'cache',
  'offline',
  'maxPrefetchBytes',
  'onProgress',
  'debertaReadback'
]);
const prototypeKeys = new Set<string | symbol>(Object.getOwnPropertyNames(Object.prototype));

export function validateNariOptions(options: unknown): ValidatedNariOptions {
  if (!options || typeof options !== 'object' || Array.isArray(options)) throw invalid('Invalid Nari options.');
  const prototype = Object.getPrototypeOf(options);
  if (
    prototype !== null &&
    (Object.getPrototypeOf(prototype) !== null || Reflect.ownKeys(prototype).some((key) => !prototypeKeys.has(key)))
  )
    throw invalid('Unsupported Nari option prototype.');
  const snapshot: Record<string, unknown> = Object.create(null);
  for (const key of Reflect.ownKeys(options)) {
    const field = Object.getOwnPropertyDescriptor(options, key);
    if (typeof key !== 'string' || !allowed.has(key) || !field || !Object.hasOwn(field, 'value'))
      throw invalid('Unsupported Nari load option.');
    snapshot[key] = field.value;
  }
  if (
    !nariVariant(snapshot.model) ||
    typeof snapshot.assetsUrl !== 'string' ||
    !snapshot.assetsUrl ||
    (snapshot.cache !== undefined && snapshot.cache !== false) ||
    (snapshot.offline !== undefined && snapshot.offline !== false) ||
    (snapshot.maxPrefetchBytes !== undefined && snapshot.maxPrefetchBytes !== 0) ||
    snapshot.onProgress !== undefined ||
    snapshot.debertaReadback !== undefined
  )
    throw invalid('Unsupported Nari load option.');
  let base: URL;
  try {
    base = new URL(snapshot.assetsUrl, globalThis.location?.href);
  } catch {
    throw invalid('Invalid Nari asset URL.');
  }
  if (!['https:', 'http:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw invalid('Invalid Nari asset URL.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  validateH2oSignal(snapshot.signal);
  return {
    base,
    signal: snapshot.signal as AbortSignal | undefined,
    variant: nariVariant(snapshot.model) as NariVariant
  };
}
