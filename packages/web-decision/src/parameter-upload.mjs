// Bounded parameter upload; GPU ownership belongs to the caller.
import {waitForAbort} from './abort.js';
const LIMIT = 128 * 1024 * 1024;
const CHUNK = 4 * 1024 * 1024;
const widths = {float16: 2, float32: 4, uint32: 4};
const hash = async (data) =>
  Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', data)), (v) => v.toString(16).padStart(2, '0')).join(
    ''
  );

function validateEntry(entry, index) {
  if (
    entry.position !== index + 5 ||
    !Array.isArray(entry.shape) ||
    !entry.shape.length ||
    entry.shape.some((n) => !Number.isSafeInteger(n) || n < 1) ||
    !Object.hasOwn(widths, entry.dtype)
  )
    throw new Error('Invalid parameter shape');
  const bytes = entry.shape.reduce((product, n) => product * n, widths[entry.dtype]);
  if (!Number.isSafeInteger(bytes) || bytes > LIMIT || bytes % 4) throw new Error('Invalid parameter size');
  if (entry.bindingKind === 'derived-rotary') {
    if (
      entry.dtype !== 'float32' ||
      !Array.isArray(entry.values) ||
      entry.values.length * 4 !== bytes ||
      entry.values.some((x) => !Number.isFinite(x))
    )
      throw new Error('Invalid rotary parameter');
    return;
  }
  if (
    entry.bindingKind !== 'q4-candidate' ||
    entry.outputBytes !== bytes ||
    !Array.isArray(entry.ranges) ||
    !entry.ranges.length
  )
    throw new Error('Invalid asset parameter');
  let offset = 0;
  for (const part of entry.ranges) {
    if (
      part.offset !== offset ||
      !Number.isSafeInteger(part.bytes) ||
      part.bytes < 4 ||
      part.bytes > CHUNK ||
      part.bytes % 4 ||
      !/^[a-f0-9]{64}$/.test(part.sha256)
    )
      throw new Error('Invalid parameter range');
    offset += part.bytes;
  }
  if (offset !== bytes) throw new Error('Incomplete parameter ranges');
}

function validate(entries, expectedCount) {
  if (![682, 684].includes(expectedCount) || !Array.isArray(entries) || entries.length !== expectedCount)
    throw new Error('Invalid asset manifest');
  const snapshot = structuredClone(entries);
  snapshot.forEach(validateEntry);
  return snapshot;
}

/** Read, verify and write each range in graph argument order, awaiting each write. */
export async function uploadParameterEntries(
  input,
  {createTensor, readRange, writeRange, signal, expectedCount = 682} = {}
) {
  if (typeof createTensor !== 'function' || typeof readRange !== 'function' || typeof writeRange !== 'function')
    throw new TypeError('Upload callbacks are required');
  const entries = validate(input, expectedCount);
  const tensors = [];
  for (const entry of entries) {
    signal?.throwIfAborted();
    const tensor = await createTensor(entry);
    tensors.push(tensor);
    if (entry.bindingKind === 'derived-rotary') {
      const bytes = new Uint8Array(Float32Array.from(entry.values).buffer);
      await writeRange(tensor, 0, bytes);
      signal?.throwIfAborted();
      continue;
    }
    for (const [index, part] of entry.ranges.entries()) {
      signal?.throwIfAborted();
      const read = await readRange(entry, index, part);
      signal?.throwIfAborted();
      if (!(read instanceof ArrayBuffer) || read.byteLength !== part.bytes) throw new Error('Wrong range size');
      const bytes = Uint8Array.from(new Uint8Array(read));
      const digest = await waitForAbort(() => hash(bytes), signal);
      signal?.throwIfAborted();
      if (digest !== part.sha256) throw new Error('Range hash mismatch');
      await writeRange(tensor, part.offset, bytes);
      signal?.throwIfAborted();
    }
  }
  return tensors;
}
