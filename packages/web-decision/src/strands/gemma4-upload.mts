const MAX_CHUNK_BYTES = 8 * 1024 * 1024;
const invalid = () => new Error('Invalid Gemma 4 shard upload metadata.');

type Gemma4DType = 'float16' | 'bfloat16' | 'float32';
type Gemma4Record = Record<string, unknown>;

interface Gemma4ExpectedParameter {
  readonly index: number;
  readonly shape: readonly number[];
  readonly dtype: Gemma4DType;
}

interface Gemma4TensorRecord {
  readonly name: string;
  readonly shape: readonly number[];
  readonly dtype: Gemma4DType;
  readonly nbytes: number;
  readonly byteOffset: number;
}

interface Gemma4UploadEntry {
  readonly index: number;
  readonly shape: number[];
  readonly dtype: Gemma4DType;
  readonly nbytes: number;
  readonly offset: number;
}

interface Gemma4TensorDestination {
  readonly byteOffset: number;
  readonly shape: readonly number[];
  getDataPtr(): unknown;
}

interface Gemma4Allocation {
  readonly tensor: Gemma4TensorDestination;
  readonly capacityBytes: number;
}

interface Gemma4UploadContext {
  copyRawBytesToBuffer(data: Uint8Array, pointer: unknown, offset: number, length: number): void;
}

interface Gemma4UploadOptions {
  readonly bytes: Uint8Array;
  readonly records: readonly Gemma4TensorRecord[];
  readonly expectedParameters: readonly Gemma4ExpectedParameter[];
  readonly parameterCount: number;
  readonly allocate: (entry: Gemma4UploadEntry) => Gemma4Allocation | PromiseLike<Gemma4Allocation>;
  readonly context: Gemma4UploadContext;
  readonly sync: () => PromiseLike<unknown>;
  readonly check?: (() => void) | undefined;
  readonly signal?: AbortSignal | null | undefined;
  readonly maxChunkBytes?: number | undefined;
}

function parameterIndex(name: unknown, count: number): number {
  if (typeof name !== 'string' || !/^param_(0|[1-9][0-9]*)$/.test(name)) throw invalid();
  const index = Number(name.slice(6));
  if (!Number.isSafeInteger(index) || index >= count) throw invalid();
  return index;
}

function tensorBytes(shape: unknown, dtype: unknown): number {
  const width = dtype === 'float16' || dtype === 'bfloat16' ? 2 : dtype === 'float32' ? 4 : 0;
  if (!width || !Array.isArray(shape)) throw invalid();
  let bytes = width;
  for (let i = 0; i < shape.length; i++) {
    if (!Object.hasOwn(shape, i) || !Number.isSafeInteger(shape[i]) || (shape[i] as number) < 1) throw invalid();
    bytes *= shape[i] as number;
    if (!Number.isSafeInteger(bytes)) throw invalid();
  }
  return bytes;
}

function validateRecords(
  bytes: unknown,
  records: unknown,
  expectedParameters: unknown,
  parameterCount: number
): Gemma4UploadEntry[] {
  if (
    !(bytes instanceof Uint8Array) ||
    !Array.isArray(records) ||
    !Array.isArray(expectedParameters) ||
    !Number.isSafeInteger(parameterCount) ||
    parameterCount < 1 ||
    !records.length ||
    records.length !== expectedParameters.length
  )
    throw invalid();
  const expected = new Map<number, {readonly shape: number[]; readonly dtype: Gemma4DType}>();
  for (let i = 0; i < expectedParameters.length; i++) {
    const item = (expectedParameters as unknown[])[i] as Gemma4Record | null | undefined;
    const index = item?.index;
    if (
      !Object.hasOwn(expectedParameters, i) ||
      !Number.isSafeInteger(index) ||
      (index as number) < 0 ||
      (index as number) >= parameterCount ||
      expected.has(index as number)
    )
      throw invalid();
    tensorBytes(item!.shape, item!.dtype);
    expected.set(index as number, {
      shape: [...(item!.shape as number[])],
      dtype: item!.dtype as Gemma4DType
    });
  }
  const entries: Gemma4UploadEntry[] = [];
  for (let i = 0; i < records.length; i++) {
    if (!Object.hasOwn(records, i)) throw invalid();
    const record = (records as unknown[])[i] as Gemma4Record | null | undefined;
    const index = parameterIndex(record?.name, parameterCount);
    const specification = expected.get(index);
    if (!specification) throw invalid();
    expected.delete(index);
    const nbytes = tensorBytes(record!.shape, record!.dtype);
    const shape = [...(record!.shape as number[])];
    if (
      record!.dtype !== specification.dtype ||
      shape.length !== specification.shape.length ||
      shape.some((value, position) => value !== specification.shape[position]!) ||
      record!.nbytes !== nbytes ||
      !Number.isSafeInteger(record!.byteOffset) ||
      (record!.byteOffset as number) < 0 ||
      nbytes > (bytes as Uint8Array).byteLength - (record!.byteOffset as number)
    )
      throw invalid();
    entries.push({
      index,
      shape,
      dtype: record!.dtype as Gemma4DType,
      nbytes,
      offset: record!.byteOffset as number
    });
  }
  if (expected.size) throw invalid();
  const ordered = entries.slice().sort((a, b) => a.offset - b.offset);
  for (let i = 1; i < ordered.length; i++) {
    if (ordered[i]!.offset < ordered[i - 1]!.offset + ordered[i - 1]!.nbytes) throw invalid();
  }
  return entries;
}

function destination(
  allocation: unknown,
  entry: Gemma4UploadEntry,
  pointers: Set<number>
): {readonly tensor: Gemma4TensorDestination; readonly pointer: number} {
  const tensor = (allocation as Gemma4Allocation | null | undefined)?.tensor;
  const capacity = (allocation as Gemma4Allocation | null | undefined)?.capacityBytes;
  const padded = Math.ceil(entry.nbytes / 4) * 4;
  if (
    !tensor ||
    !Number.isSafeInteger(capacity) ||
    !Number.isSafeInteger(padded) ||
    (capacity as number) < padded ||
    tensor.byteOffset !== 0 ||
    !Array.isArray(tensor.shape) ||
    tensor.shape.length !== entry.shape.length ||
    entry.shape.some((value, index) => !Object.hasOwn(tensor.shape, index) || tensor.shape[index]! !== value) ||
    typeof tensor.getDataPtr !== 'function'
  )
    throw invalid();
  const pointer: number = tensor.getDataPtr() as number;
  if (!Number.isSafeInteger(pointer) || pointer < 0 || pointers.has(pointer)) throw invalid();
  pointers.add(pointer);
  return {tensor, pointer};
}

/**
 * Upload one already verified shard. The caller owns bytes exclusively until this
 * settles, and owns all tensors and GPU work on failure. The allocator must
 * provide the actual independent allocation capacity, including physical padding.
 * expectedParameters must come from an independently verified graph/shard plan.
 */
export async function uploadGemma4Shard({
  bytes,
  records,
  expectedParameters,
  parameterCount,
  allocate,
  context,
  sync,
  check = () => {},
  signal,
  maxChunkBytes = MAX_CHUNK_BYTES
}: Gemma4UploadOptions): Promise<Map<number, Gemma4TensorDestination>> {
  if (
    typeof allocate !== 'function' ||
    typeof context?.copyRawBytesToBuffer !== 'function' ||
    typeof sync !== 'function' ||
    typeof check !== 'function' ||
    !Number.isSafeInteger(maxChunkBytes) ||
    maxChunkBytes < 4 ||
    maxChunkBytes > MAX_CHUNK_BYTES ||
    maxChunkBytes % 4
  )
    throw invalid();
  signal?.throwIfAborted();
  const entries = validateRecords(bytes, records, expectedParameters, parameterCount);
  const result = new Map<number, Gemma4TensorDestination>(),
    pointers = new Set<number>();
  let pending = false;
  try {
    for (const entry of entries) {
      signal?.throwIfAborted();
      check();
      const {tensor, pointer} = destination(await allocate({...entry, shape: [...entry.shape]}), entry, pointers);
      const aligned = entry.nbytes - (entry.nbytes % 4);
      for (let offset = 0; offset < aligned;) {
        signal?.throwIfAborted();
        check();
        const length = Math.min(maxChunkBytes, aligned - offset);
        const data = bytes.subarray(entry.offset + offset, entry.offset + offset + length);
        // A write may throw after submitting work; the owner must still drain it.
        pending = true;
        context.copyRawBytesToBuffer(data, pointer, offset, length);
        offset += length;
      }
      if (aligned !== entry.nbytes) {
        signal?.throwIfAborted();
        check();
        const tail = new Uint8Array(4);
        tail.set(bytes.subarray(entry.offset + aligned, entry.offset + entry.nbytes));
        pending = true;
        context.copyRawBytesToBuffer(tail, pointer, aligned, tail.length);
      }
      result.set(entry.index, tensor);
    }
    signal?.throwIfAborted();
    check();
    if (pending) {
      await sync();
      pending = false;
    }
    signal?.throwIfAborted();
    check();
    return result;
  } catch (error) {
    if (pending) {
      try {
        await sync();
      } catch (drainError) {
        throw new AggregateError([error, drainError], 'Gemma 4 shard upload failed and could not be drained.', {
          cause: drainError
        });
      }
    }
    throw error;
  }
}
