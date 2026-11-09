import {WebDecisionError} from './kev.js';

interface TensorRecord {
  readonly index: number;
  readonly offset: number;
  readonly dtype: 'float16' | 'float32' | 'int32' | 'uint32';
  readonly shape: number[];
  readonly nbytes: number;
}

interface TensorShard {
  readonly path: string;
  readonly nbytes: number;
}

interface TensorIndex {
  readonly totalBytes: number;
  readonly records: TensorRecord[];
  readonly shards: TensorShard[];
}

interface TensorIndexLimits {
  readonly parameterCount: number;
  readonly totalBytes: number;
  readonly shardCount: number;
  readonly maxShardBytes: number;
  readonly allowedDtypes?: string[] | undefined;
}

function invalid(message: string): never {
  throw new WebDecisionError('INVALID_ASSET', message);
}

function validateRecord(record: TensorRecord, index: number, offset: number, allowedDtypes: string[]): void {
  const width = {float16: 2, float32: 4, int32: 4, uint32: 4}[record?.dtype];
  if (
    record?.index !== index ||
    record.offset !== offset ||
    !width ||
    !allowedDtypes.includes(record.dtype) ||
    !Array.isArray(record.shape) ||
    !record.shape.length ||
    record.shape.some((n) => !Number.isSafeInteger(n) || n < 1) ||
    !Number.isSafeInteger(record.nbytes) ||
    record.nbytes < 4 ||
    record.nbytes % 4
  )
    invalid('Invalid tensor parameter metadata.');
  const size = record.shape.reduce((product, n) => product * n, width);
  if (!Number.isSafeInteger(size) || size !== record.nbytes) invalid('Invalid tensor parameter shape.');
}

/** Validate contiguous parameter and shard spans before allocating GPU memory. */
export function validateTensorIndex(
  manifest: unknown,
  index: unknown,
  {
    parameterCount,
    totalBytes,
    shardCount,
    maxShardBytes,
    allowedDtypes = ['float16', 'float32', 'int32']
  }: TensorIndexLimits
): unknown {
  if (
    !index ||
    (index as TensorIndex).totalBytes !== totalBytes ||
    !Array.isArray((index as TensorIndex).records) ||
    (index as TensorIndex).records.length !== parameterCount ||
    !Array.isArray((index as TensorIndex).shards) ||
    (index as TensorIndex).shards.length !== shardCount
  )
    invalid('Invalid tensor weight index.');
  let total = 0;
  for (const [i, record] of (index as TensorIndex).records.entries()) {
    validateRecord(record, i, total, allowedDtypes);
    total += record.nbytes;
    if (!Number.isSafeInteger(total) || total > totalBytes) invalid('Invalid tensor parameter range.');
  }
  if (total !== totalBytes) invalid('Missing tensor parameter data.');
  total = 0;
  for (const [i, shard] of (index as TensorIndex).shards.entries()) {
    const path = `weights/shard-${String(i).padStart(4, '0')}.bin`;
    if (
      shard?.path !== path ||
      !Number.isSafeInteger(shard.nbytes) ||
      shard.nbytes < 4 ||
      shard.nbytes > maxShardBytes ||
      shard.nbytes % 4 ||
      shard.nbytes !== (manifest as {files?: Record<string, {bytes: number}>}).files?.[path]?.bytes
    )
      invalid('Invalid tensor shard metadata.');
    total += shard.nbytes;
  }
  if (total !== totalBytes) invalid('Missing tensor shard data.');
  return index;
}
