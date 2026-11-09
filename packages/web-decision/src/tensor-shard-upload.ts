import {WebDecisionError} from './kev.js';
import {assertLoadActive} from './runtime-lifecycle.js';
import {processPrefetchedShards} from './shard-prefetch.js';

interface Tensor {
  readonly byteOffset?: number;
  getDataPtr(): unknown;
}

interface UploadOptions {
  readonly index: {
    readonly records: readonly {
      readonly offset: number;
      readonly nbytes: number;
      readonly shape: number[];
      readonly dtype: string;
    }[];
    readonly shards: readonly {readonly path: string; readonly nbytes: number}[];
    readonly totalBytes: number;
  };
  readonly read: (path: string, signal?: AbortSignal) => Promise<ArrayBuffer>;
  readonly runtime: {
    readonly lib?: {
      readonly webGPUContext?: {
        copyRawBytesToBuffer(bytes: Uint8Array, pointer: unknown, offset: number, count: number): unknown;
      };
    };
    empty(shape: number[], dtype: string, device: unknown): Tensor;
  };
  readonly device: {sync(): Promise<unknown>};
  readonly weights: unknown[];
  readonly signal?: AbortSignal | undefined;
  readonly maxPrefetchBytes?: number | undefined;
  readonly lifecycle: {readonly disposed: boolean; readonly loss: unknown};
  readonly uncaptured: string[];
  readonly inspectFragment?: (index: number, bytes: Uint8Array, offset: number) => void;
}

/** Upload verified, contiguous tensor shards with bounded staging memory. */
export async function uploadTensorShards({
  index,
  read,
  runtime,
  device,
  weights,
  signal,
  maxPrefetchBytes,
  lifecycle,
  uncaptured,
  inspectFragment
}: UploadOptions): Promise<void> {
  // The pinned TVM internal supports offsets, so large tensors do not require
  // an equally large JavaScript staging buffer.
  const context = runtime.lib?.webGPUContext;
  if (!context || typeof context.copyRawBytesToBuffer !== 'function')
    throw new WebDecisionError('MODEL_LOAD_FAILED', 'The pinned TVM streaming upload API is unavailable.');
  let recordIndex = 0,
    offset = 0;
  const consume = async (shard: {readonly nbytes: number}, buffer: ArrayBuffer) => {
    const data = new Uint8Array(buffer);
    if (data.length !== shard.nbytes) throw new WebDecisionError('INVALID_ASSET', 'Invalid tensor shard length.');
    let start = 0;
    while (start < data.length) {
      signal?.throwIfAborted();
      assertLoadActive(lifecycle);
      const record = index.records[recordIndex];
      if (!record || offset < record.offset || offset >= record.offset + record.nbytes)
        throw new WebDecisionError('INVALID_ASSET', 'Tensor parameter spans are inconsistent.');
      const relative = offset - record.offset,
        count = Math.min(data.length - start, record.nbytes - relative),
        fragment = data.subarray(start, start + count);
      inspectFragment?.(recordIndex, fragment, relative);
      const tensor = (weights[recordIndex] ??= runtime.empty(record.shape, record.dtype, device)) as Tensor;
      if (typeof tensor.byteOffset !== 'number' || tensor.byteOffset % 4 || count % 4)
        throw new WebDecisionError('MODEL_LOAD_FAILED', 'Tensor does not support aligned streaming upload.');
      context.copyRawBytesToBuffer(fragment, tensor.getDataPtr(), tensor.byteOffset + relative, count);
      start += count;
      offset += count;
      if (relative + count === record.nbytes) recordIndex++;
    }
    await device.sync();
    signal?.throwIfAborted();
    assertLoadActive(lifecycle);
    if (uncaptured.length) throw new WebDecisionError('MODEL_LOAD_FAILED', uncaptured[0]!);
  };
  await processPrefetchedShards(
    index.shards,
    (shard, readSignal) => read(shard.path, readSignal),
    consume,
    signal,
    maxPrefetchBytes
  );
  if (recordIndex !== index.records.length || offset !== index.totalBytes)
    throw new WebDecisionError('INVALID_ASSET', 'Incomplete tensor parameter data.');
}
