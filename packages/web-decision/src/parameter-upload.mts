// Bounded parameter upload; GPU ownership belongs to the caller.
import {waitForAbort} from './abort.js';

const LIMIT = 128 * 1024 * 1024;
const CHUNK = 4 * 1024 * 1024;
const widths = {float16: 2, float32: 4, uint32: 4};
type ParameterDtype = keyof typeof widths;

export interface ParameterUploadRangePart {
  readonly offset: number;
  readonly bytes: number;
  readonly sha256: string;
}

export interface DerivedParameterUploadEntry {
  readonly position: number;
  readonly shape: readonly number[];
  readonly dtype: unknown;
  readonly bindingKind: 'derived-rotary';
  readonly values: unknown;
}

export interface Q4ParameterUploadEntry {
  readonly position: number;
  readonly target?: unknown | undefined;
  readonly shape: readonly number[];
  readonly dtype: unknown;
  readonly bindingKind: 'q4-candidate';
  readonly outputBytes: unknown;
  readonly ranges: unknown;
}

export type ParameterUploadEntry = DerivedParameterUploadEntry | Q4ParameterUploadEntry;

export type ValidatedParameterUploadEntry<Entry extends ParameterUploadEntry> =
  Entry extends DerivedParameterUploadEntry
    ? Entry & {readonly dtype: 'float32'; readonly values: readonly number[]}
    : Entry extends Q4ParameterUploadEntry
      ? Entry & {
          readonly dtype: ParameterDtype;
          readonly outputBytes: number;
          readonly ranges: readonly ParameterUploadRangePart[];
        }
      : never;

type ValidatedQ4Entry<Entry extends ParameterUploadEntry> = Extract<
  ValidatedParameterUploadEntry<Entry>,
  {readonly bindingKind: 'q4-candidate'}
>;

interface RangeCandidate {
  readonly offset: unknown;
  readonly bytes: unknown;
  readonly sha256: unknown;
}

export interface ParameterUploadOptions<Entry extends ParameterUploadEntry, Tensor> {
  readonly createTensor?: ((entry: ValidatedParameterUploadEntry<Entry>) => Tensor | PromiseLike<Tensor>) | undefined;
  readonly readRange?:
    | ((
        entry: ValidatedQ4Entry<Entry>,
        index: number,
        part: ParameterUploadRangePart
      ) => ArrayBuffer | PromiseLike<ArrayBuffer>)
    | undefined;
  readonly writeRange?:
    | ((tensor: Tensor, offset: number, data: Uint8Array) => unknown | PromiseLike<unknown>)
    | undefined;
  readonly signal?: AbortSignal | null | undefined;
  readonly expectedCount?: number | undefined;
}

const hash = async (data: Uint8Array): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', data as BufferSource)), (v) =>
    v.toString(16).padStart(2, '0')
  ).join('');

function validateEntry(
  entry: ParameterUploadEntry,
  index: number
): asserts entry is ValidatedParameterUploadEntry<ParameterUploadEntry> {
  if (
    entry.position !== index + 5 ||
    !Array.isArray(entry.shape) ||
    !entry.shape.length ||
    entry.shape.some((n: unknown) => !Number.isSafeInteger(n) || (n as number) < 1) ||
    !Object.hasOwn(widths, entry.dtype as PropertyKey)
  )
    throw new Error('Invalid parameter shape');
  const bytes = (entry.shape as readonly unknown[]).reduce<number>(
    (product, n) => product * (n as number),
    widths[entry.dtype as ParameterDtype]
  );
  if (!Number.isSafeInteger(bytes) || bytes > LIMIT || bytes % 4) throw new Error('Invalid parameter size');
  if (entry.bindingKind === 'derived-rotary') {
    if (
      entry.dtype !== 'float32' ||
      !Array.isArray(entry.values) ||
      entry.values.length * 4 !== bytes ||
      entry.values.some((x: unknown) => !Number.isFinite(x))
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
  for (const part of entry.ranges as readonly RangeCandidate[]) {
    if (
      part.offset !== offset ||
      !Number.isSafeInteger(part.bytes) ||
      (part.bytes as number) < 4 ||
      (part.bytes as number) > CHUNK ||
      (part.bytes as number) % 4 ||
      !/^[a-f0-9]{64}$/.test(part.sha256 as string)
    )
      throw new Error('Invalid parameter range');
    offset += part.bytes as number;
  }
  if (offset !== bytes) throw new Error('Incomplete parameter ranges');
}

function validate<Entry extends ParameterUploadEntry>(
  entries: readonly Entry[],
  expectedCount: number
): Array<ValidatedParameterUploadEntry<Entry>> {
  if (![682, 684].includes(expectedCount) || !Array.isArray(entries) || entries.length !== expectedCount)
    throw new Error('Invalid asset manifest');
  const snapshot = structuredClone(entries);
  snapshot.forEach(validateEntry);
  return snapshot as Array<ValidatedParameterUploadEntry<Entry>>;
}

/** Read, verify and write each range in graph argument order, awaiting each write. */
export async function uploadParameterEntries<Entry extends ParameterUploadEntry, Tensor>(
  input: readonly Entry[],
  {createTensor, readRange, writeRange, signal, expectedCount = 682}: ParameterUploadOptions<Entry, Tensor> = {}
): Promise<Tensor[]> {
  if (typeof createTensor !== 'function' || typeof readRange !== 'function' || typeof writeRange !== 'function')
    throw new TypeError('Upload callbacks are required');
  const entries = validate(input, expectedCount);
  const tensors: Tensor[] = [];
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
      const read = await readRange(entry as ValidatedQ4Entry<Entry>, index, part);
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
