import {parseJson, sha256, waitForAbort} from './assets.js';
import {MAX_MANIFEST_BYTES, readBoundedResponse} from './bounded-response.js';
import {WebDecisionError} from './kev.js';

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>;
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, canonical(record[key])])
    );
  }
  return value;
}

/** Verify canonical JSON so a reserialized cached manifest has the same identity. */
export async function readPinnedManifest(
  response: Pick<Response, 'body'>,
  signal: AbortSignal | null | undefined,
  expected: string,
  name: string
): Promise<unknown> {
  try {
    const bytes = await readBoundedResponse(response, MAX_MANIFEST_BYTES, {signal} as NonNullable<
      Parameters<typeof readBoundedResponse>[2]
    >);
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
    if (cause instanceof WebDecisionError || isAbortError(cause)) throw cause;
    throw new WebDecisionError('INVALID_ASSET', `Could not read the ${name} manifest.`, {cause});
  }
}

function isAbortError(cause: unknown): boolean {
  return (cause as {readonly name?: unknown} | null | undefined)?.name === 'AbortError';
}
