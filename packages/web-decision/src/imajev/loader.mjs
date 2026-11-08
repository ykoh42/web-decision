import {WebDecisionError} from '../kev.js';
import {validatePrefetchBytes} from '../shard-prefetch.js';
import {rejectUnverifiedImajevAssets} from './contract.mjs';

/** Fail before network, cache, WebGPU or graph initialization; no assets are trusted yet. */
export async function loadExperimentalImajev(options) {
  if (!options || typeof options.assetsUrl !== 'string' || !options.assetsUrl)
    throw new WebDecisionError('INVALID_INPUT', 'assetsUrl is required.');
  if (options.offline !== undefined && typeof options.offline !== 'boolean')
    throw new WebDecisionError('INVALID_INPUT', 'offline must be a boolean.');
  if (options.offline && options.cache === false)
    throw new WebDecisionError('INVALID_INPUT', 'Offline loading requires the model cache.');
  validatePrefetchBytes(options.maxPrefetchBytes);
  let base;
  try {
    base = new URL(options.assetsUrl, globalThis.location?.href);
  } catch (cause) {
    throw new WebDecisionError('INVALID_INPUT', 'Invalid assetsUrl.', {cause});
  }
  if (!['http:', 'https:'].includes(base.protocol) || base.search || base.hash || base.username || base.password)
    throw new WebDecisionError(
      'INVALID_INPUT',
      'assetsUrl must be an HTTP or HTTPS directory URL without credentials, query, or fragment.'
    );
  options.signal?.throwIfAborted();
  rejectUnverifiedImajevAssets();
}
