import {WebDecisionError} from '../kev.js';
import {openAssetCache, readVerifiedAsset} from '../assets.js';
import {KANANA_ASSETS_URL, KANANA_SOURCE_SPEC} from './contract.mjs';

function checkedBase(value) {
  let base;
  try {
    base = value instanceof URL ? new URL(value.href) : new URL(value);
  } catch (cause) {
    throw new WebDecisionError('INVALID_INPUT', 'Invalid Kanana assets URL.', {cause});
  }
  if (!['http:', 'https:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw new WebDecisionError('INVALID_INPUT', 'Kanana assets URL must be an HTTP or HTTPS directory.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return base;
}

/** Open only the pinned Kanana tokenizer and model metadata assets. */
export async function openKananaSource({
  base = KANANA_ASSETS_URL,
  signal,
  cacheEnabled = true,
  offline = false,
  onProgress,
  fetchImpl
} = {}) {
  const resolvedBase = checkedBase(base);
  const cache = await openAssetCache(cacheEnabled, signal);
  let loadedBytes = 0;

  return Object.freeze({
    async read(name) {
      return readVerifiedAsset(resolvedBase, KANANA_SOURCE_SPEC.manifest, name, {
        cache,
        signal,
        offline,
        maxAssetBytes: KANANA_SOURCE_SPEC.maxAssetBytes,
        fetchImpl,
        onBytes(bytes) {
          loadedBytes += bytes;
          onProgress?.({
            loadedBytes,
            totalBytes: KANANA_SOURCE_SPEC.totalBytes,
            fraction: loadedBytes / KANANA_SOURCE_SPEC.totalBytes
          });
        }
      });
    }
  });
}
