import {parseJson, sha256, waitForAbort} from './assets.js';
import {MAX_MANIFEST_BYTES, readBoundedResponse} from './bounded-response.js';
import {WebDecisionError} from './kev.js';

function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === 'object')
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, canonical(value[key])])
    );
  return value;
}

/** Verify canonical JSON so a reserialized cached manifest has the same identity. */
export async function readPinnedManifest(response, signal, expected, name) {
  try {
    const bytes = await readBoundedResponse(response, MAX_MANIFEST_BYTES, {signal});
    const manifest = parseJson(bytes, 'manifest.json');
    const canonicalBytes = new TextEncoder().encode(JSON.stringify(canonical(manifest)) + '\n');
    const digest = await waitForAbort(() => sha256(canonicalBytes), signal);
    signal?.throwIfAborted();
    if (digest !== expected)
      throw new WebDecisionError(
        'INVALID_ASSET',
        `The ${name} manifest does not match the verified experimental build.`
      );
    return manifest;
  } catch (cause) {
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
    throw new WebDecisionError('INVALID_ASSET', `Could not read the ${name} manifest.`, {cause});
  }
}
