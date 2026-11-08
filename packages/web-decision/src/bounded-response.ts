import {waitForAbort} from './abort.js';

export const MAX_MANIFEST_BYTES = 1024 * 1024;
export const MAX_ASSET_BYTES = 128 * 1024 * 1024;
export const MAX_VERIFIED_ASSET_BYTES = 1024 * 1024 * 1024;

export class ResponseSizeError extends Error {}

type CancelableBody = {cancel?: () => unknown} | null | undefined;

/** Request cancellation without waiting for a possibly noncooperative body. */
export function cancelBody(body: CancelableBody): void {
  try {
    Promise.resolve(body?.cancel?.()).catch(() => {});
  } catch {}
}

/** Dispose responses returned by a fetch or cache wrapper after cancellation. */
export function cancelResponse(response: Pick<Response, 'body'> | null | undefined): void {
  cancelBody(response?.body);
}

function appendChunk(buffer: Uint8Array, length: number, chunk: unknown): number {
  if (!ArrayBuffer.isView(chunk) || !('BYTES_PER_ELEMENT' in chunk) || chunk.BYTES_PER_ELEMENT !== 1)
    throw new TypeError('Response stream did not return bytes.');
  if (chunk.byteLength > buffer.length - length) throw new ResponseSizeError('Response body is too large.');
  const bytes = new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength);
  buffer.set(bytes, length);
  return length + chunk.byteLength;
}

interface BoundedResponseOptions {
  exact?: boolean;
  signal?: AbortSignal;
}

/** Read decoded response bytes with an effective size limit and abort support. */
export async function readBoundedResponse(
  response: Pick<Response, 'body'> | null | undefined,
  maxBytes: number,
  {exact = false, signal}: BoundedResponseOptions = {}
): Promise<ArrayBuffer> {
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
  let length = 0;
  let done = false;
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
