import {sha256, waitForAbort} from '../assets.js';
import type {ClefRangePart, ClefTensorEntry} from './types.mjs';

interface RangeUploadOptions {
  fake?: boolean;
  signal?: AbortSignal | undefined;
  readRange(
    entry: ClefTensorEntry,
    part: ClefRangePart,
    signal?: AbortSignal | undefined
  ): Promise<ArrayBuffer | Uint8Array>;
  writeRange(tensor: unknown, offset: number, bytes: Uint8Array): unknown | PromiseLike<unknown>;
}

interface UploadOptions extends RangeUploadOptions {
  createTensor(entry: ClefTensorEntry): unknown | PromiseLike<unknown>;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function finite(bytes: Uint8Array, dtype: ClefTensorEntry['dtype']): void {
  if (dtype === 'uint32') return;
  const width = dtype === 'float16' ? 2 : 4;
  if (bytes.length % width) throw new Error('Invalid Clef numeric alignment.');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let i = 0; i < bytes.length; i += width) {
    const bits = width === 2 ? view.getUint16(i, true) : view.getUint32(i, true);
    if ((bits & (width === 2 ? 0x7c00 : 0x7f800000)) === (width === 2 ? 0x7c00 : 0x7f800000))
      throw new Error('Nonfinite Clef parameter.');
  }
}

function validateRanges(entry: ClefTensorEntry, bytes: number): void {
  let offset = 0;
  for (const [partIndex, part] of entry.chunks.entries()) {
    if (
      part.offset !== offset ||
      !Number.isSafeInteger(part.length) ||
      part.length < 1 ||
      part.length > 4 * 1024 * 1024 ||
      typeof part.sha256 !== 'string' ||
      !/^[a-f0-9]{64}$/.test(part.sha256) ||
      offset % 4 ||
      (part.length % 4 &&
        !(partIndex === entry.chunks.length - 1 && entry.dtype === 'float16' && part.length % 4 === 2))
    )
      throw new Error('Invalid Clef parameter range.');
    offset += part.length;
  }
  if (offset !== bytes) throw new Error('Incomplete Clef parameter ranges.');
}

function validateEntries(snapshot: unknown): asserts snapshot is ClefTensorEntry[] {
  if (!Array.isArray(snapshot)) throw new Error('Invalid Clef parameter metadata.');
  for (const [index, entryValue] of snapshot.entries()) {
    const entry = isRecord(entryValue) ? entryValue : null;
    if (
      !entry ||
      entry.position !== index + 17 ||
      !Array.isArray(entry.shape) ||
      entry.shape.some((n) => !Number.isSafeInteger(n) || n < 1) ||
      typeof entry.dtype !== 'string' ||
      !['uint32', 'float16', 'float32'].includes(entry.dtype) ||
      !Array.isArray(entry.chunks) ||
      !entry.chunks.length
    )
      throw new Error('Invalid Clef parameter metadata.');
    const shape = entry.shape as number[];
    const dtype = entry.dtype as ClefTensorEntry['dtype'];
    const bytes = shape.reduce((n, dim) => n * dim, dtype === 'float16' ? 2 : 4);
    if (!Number.isSafeInteger(bytes) || bytes !== entry.bytes || bytes < 1 || bytes > 128 * 1024 * 1024)
      throw new Error('Invalid Clef parameter size.');
    validateRanges(entry as unknown as ClefTensorEntry, bytes);
  }
}

async function uploadRanges(
  entry: ClefTensorEntry,
  tensor: unknown,
  {readRange, writeRange, signal}: RangeUploadOptions
): Promise<void> {
  for (const [index, part] of entry.chunks.entries()) {
    signal?.throwIfAborted();
    const read = await readRange(entry, part, signal);
    signal?.throwIfAborted();
    const raw = read instanceof ArrayBuffer ? new Uint8Array(read) : read;
    if (!(raw instanceof Uint8Array) || raw.byteLength !== part.length) throw new Error('Wrong Clef range length.');
    const bytes = Uint8Array.from(raw);
    const digest = await waitForAbort(() => sha256(bytes), signal);
    signal?.throwIfAborted();
    if (digest !== part.sha256) throw new Error('Clef parameter hash mismatch.');
    const last = index === entry.chunks.length - 1;
    if (part.offset % 4 || (bytes.length % 4 && !(last && entry.dtype === 'float16' && bytes.length % 4 === 2)))
      throw new Error('Invalid Clef upload alignment.');
    finite(bytes, entry.dtype);
    const padded = bytes.length % 4 ? new Uint8Array(bytes.length + 2) : bytes;
    if (padded !== bytes) padded.set(bytes);
    await writeRange(tensor, part.offset, padded);
    signal?.throwIfAborted();
  }
}

async function upload(
  entries: unknown,
  {createTensor, readRange, writeRange, signal}: UploadOptions,
  count: number
): Promise<unknown[]> {
  if (
    !Array.isArray(entries) ||
    entries.length !== count ||
    typeof createTensor !== 'function' ||
    typeof readRange !== 'function' ||
    typeof writeRange !== 'function'
  )
    throw new TypeError('Invalid Clef upload.');
  const snapshot: unknown = structuredClone(entries);
  const tensors: unknown[] = [];
  validateEntries(snapshot);
  for (const entry of snapshot) {
    signal?.throwIfAborted();
    const tensor = await createTensor(entry);
    tensors.push(tensor);
    await uploadRanges(entry, tensor, {readRange, writeRange, signal});
  }
  return tensors;
}

export function uploadClefParameters(entries: unknown, options: UploadOptions): Promise<unknown[]> {
  return upload(entries, options, 797);
}
export function uploadFakeClefParameters(entries: unknown, options: UploadOptions): Promise<unknown[]> {
  if (options?.fake !== true || !Array.isArray(entries) || entries.length < 1 || entries.length > 4)
    throw new TypeError('Expected a bounded synthetic upload.');
  return upload(entries, options, entries.length);
}
