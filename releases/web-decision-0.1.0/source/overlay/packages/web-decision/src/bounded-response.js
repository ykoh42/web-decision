import {waitForAbort} from './abort.js';

export const MAX_MANIFEST_BYTES = 1024 * 1024;
export const MAX_ASSET_BYTES = 128 * 1024 * 1024;
export const MAX_VERIFIED_ASSET_BYTES = 1024 * 1024 * 1024;

export class ResponseSizeError extends Error {}

/** Request cancellation without waiting for a possibly noncooperative body. */
export function cancelBody(body) {
  try {
    Promise.resolve(body?.cancel()).catch(() => {});
  } catch {}
}

/** Dispose responses returned by a fetch or cache wrapper after cancellation. */
export function cancelResponse(response) {
  cancelBody(response?.body);
}

function appendChunk(buffer, length, chunk) {
  if (!ArrayBuffer.isView(chunk) || chunk.BYTES_PER_ELEMENT !== 1)
    throw new TypeError('Response stream did not return bytes.');
  if (chunk.byteLength > buffer.length - length) throw new ResponseSizeError('Response body is too large.');
  buffer.set(chunk, length);
  return length + chunk.byteLength;
}

/** Read decoded response bytes with an effective size limit and abort support. */
export async function readBoundedResponse(response, maxBytes, {exact = false, signal} = {}) {
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 0) throw new RangeError('Invalid response size limit.');
  if (signal?.aborted) {
    cancelResponse(response);
    signal.throwIfAborted();
  }
  const body = response?.body;
  if (!body) {
    if (exact && maxBytes !== 0) throw new ResponseSizeError('Response body is too short.');
    return new ArrayBuffer(0);
  }
  const reader = body.getReader();
  let length = 0,
    done = false;
  try {
    const buffer = new Uint8Array(maxBytes);
    for (;;) {
      const item = await waitForAbort(() => reader.read(), signal);
      if (item.done) {
        done = true;
        break;
      }
      length = appendChunk(buffer, length, item.value);
    }
    signal?.throwIfAborted();
    if (exact && length !== maxBytes) throw new ResponseSizeError('Response body is too short.');
    return exact || length === maxBytes ? buffer.buffer : buffer.buffer.slice(0, length);
  } finally {
    if (!done) cancelBody(reader);
    try {
      reader.releaseLock();
    } catch {}
  }
}
