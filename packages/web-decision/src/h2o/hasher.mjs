// The library is the unmodified locally cached fast-sha256 1.3.0 UMD script.
// In browsers load package/sha256.js as a pinned classic script before this
// module, then pass globalThis.sha256.Hash. Node tests pass its CJS export.
export const MAX_CHUNK_BYTES = 8 * 1024 * 1024;
export const MAX_FILE_BYTES = 128 * 1024 * 1024;

export function createIncrementalSha256(HashClass = globalThis.sha256?.Hash) {
  if (typeof HashClass !== 'function') throw new TypeError('Pinned fast-sha256 Hash is required');
  const state = new HashClass();
  if (typeof state.update !== 'function' || typeof state.digest !== 'function' || typeof state.clean !== 'function')
    throw new TypeError('Invalid incremental SHA-256 implementation');
  let total = 0,
    done = false;
  return {
    update(bytes) {
      if (done) throw new Error('Hash already finalized');
      if (
        !(bytes instanceof Uint8Array) ||
        bytes.byteLength > MAX_CHUNK_BYTES ||
        bytes.byteLength > MAX_FILE_BYTES - total
      )
        throw new RangeError('Invalid or oversized SHA-256 update');
      state.update(bytes);
      total += bytes.byteLength;
    },
    digest() {
      if (done) throw new Error('Hash already finalized');
      done = true;
      try {
        const result = state.digest();
        if (!(result instanceof Uint8Array) || result.length !== 32) throw new Error('Invalid SHA-256 result');
        return [...result].map((x) => x.toString(16).padStart(2, '0')).join('');
      } finally {
        state.clean();
      }
    }
  };
}
