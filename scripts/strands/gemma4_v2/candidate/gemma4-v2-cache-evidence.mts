/** Rehash every digest-addressed entry in the library's CacheStorage. */
export async function verifyGemmaV2Cache() {
  const cache = await caches.open('web-decision-assets-v1');
  let verifiedBytes = 0;
  let verifiedEntries = 0;
  for (const request of await cache.keys()) {
    const url = new URL(request.url);
    if (url.pathname !== '/.web-decision-cache/sha256/') continue;
    const expectedDigest = url.searchParams.get('digest');
    if (!expectedDigest || !/^[0-9a-f]{64}$/u.test(expectedDigest)) throw new Error('Invalid cached digest key.');
    const response = await cache.match(request);
    if (!response) throw new Error('Cached entry disappeared during verification.');
    const bytes = await response.arrayBuffer();
    const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), (value) =>
      value.toString(16).padStart(2, '0')
    ).join('');
    if (digest !== expectedDigest) throw new Error(`Corrupted cached entry ${expectedDigest}`);
    verifiedBytes += bytes.byteLength;
    verifiedEntries++;
  }
  return {verifiedBytes, verifiedEntries};
}
