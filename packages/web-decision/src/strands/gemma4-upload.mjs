const MAX_CHUNK_BYTES = 8 * 1024 * 1024;
const invalid = () => new Error('Invalid Gemma 4 shard upload metadata.');

function parameterIndex(name, count) {
  if (typeof name !== 'string' || !/^param_(0|[1-9][0-9]*)$/.test(name)) throw invalid();
  const index = Number(name.slice(6));
  if (!Number.isSafeInteger(index) || index >= count) throw invalid();
  return index;
}

function tensorBytes(shape, dtype) {
  const width = dtype === 'float16' || dtype === 'bfloat16' ? 2 : dtype === 'float32' ? 4 : 0;
  if (!width || !Array.isArray(shape)) throw invalid();
  let bytes = width;
  for (let i = 0; i < shape.length; i++) {
    if (!Object.hasOwn(shape, i) || !Number.isSafeInteger(shape[i]) || shape[i] < 1) throw invalid();
    bytes *= shape[i];
    if (!Number.isSafeInteger(bytes)) throw invalid();
  }
  return bytes;
}

function validateRecords(bytes, records, expectedParameters, parameterCount) {
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
  const expected = new Map();
  for (let i = 0; i < expectedParameters.length; i++) {
    const item = expectedParameters[i];
    const index = item?.index;
    if (
      !Object.hasOwn(expectedParameters, i) ||
      !Number.isSafeInteger(index) ||
      index < 0 ||
      index >= parameterCount ||
      expected.has(index)
    )
      throw invalid();
    tensorBytes(item.shape, item.dtype);
    expected.set(index, {shape: [...item.shape], dtype: item.dtype});
  }
  const entries = [];
  for (let i = 0; i < records.length; i++) {
    if (!Object.hasOwn(records, i)) throw invalid();
    const record = records[i];
    const index = parameterIndex(record?.name, parameterCount);
    const specification = expected.get(index);
    if (!specification) throw invalid();
    expected.delete(index);
    const nbytes = tensorBytes(record.shape, record.dtype);
    const shape = [...record.shape];
    if (
      record.dtype !== specification.dtype ||
      shape.length !== specification.shape.length ||
      shape.some((value, position) => value !== specification.shape[position]) ||
      record.nbytes !== nbytes ||
      !Number.isSafeInteger(record.byteOffset) ||
      record.byteOffset < 0 ||
      nbytes > bytes.byteLength - record.byteOffset
    )
      throw invalid();
    entries.push({index, shape, dtype: record.dtype, nbytes, offset: record.byteOffset});
  }
  if (expected.size) throw invalid();
  const ordered = entries.slice().sort((a, b) => a.offset - b.offset);
  for (let i = 1; i < ordered.length; i++) {
    if (ordered[i].offset < ordered[i - 1].offset + ordered[i - 1].nbytes) throw invalid();
  }
  return entries;
}

function destination(allocation, entry, pointers) {
  const tensor = allocation?.tensor;
  const capacity = allocation?.capacityBytes;
  const padded = Math.ceil(entry.nbytes / 4) * 4;
  if (
    !tensor ||
    !Number.isSafeInteger(capacity) ||
    !Number.isSafeInteger(padded) ||
    capacity < padded ||
    tensor.byteOffset !== 0 ||
    !Array.isArray(tensor.shape) ||
    tensor.shape.length !== entry.shape.length ||
    entry.shape.some((value, index) => !Object.hasOwn(tensor.shape, index) || tensor.shape[index] !== value) ||
    typeof tensor.getDataPtr !== 'function'
  )
    throw invalid();
  const pointer = tensor.getDataPtr();
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
}) {
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
  const result = new Map(),
    pointers = new Set();
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
