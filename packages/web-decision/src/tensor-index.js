import {WebDecisionError} from './kev.js';

function invalid(message) {
  throw new WebDecisionError('INVALID_ASSET', message);
}

function validateRecord(record, index, offset, allowedDtypes) {
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
  manifest,
  index,
  {parameterCount, totalBytes, shardCount, maxShardBytes, allowedDtypes = ['float16', 'float32', 'int32']}
) {
  if (
    !index ||
    index.totalBytes !== totalBytes ||
    !Array.isArray(index.records) ||
    index.records.length !== parameterCount ||
    !Array.isArray(index.shards) ||
    index.shards.length !== shardCount
  )
    invalid('Invalid tensor weight index.');
  let total = 0;
  for (const [i, record] of index.records.entries()) {
    validateRecord(record, i, total, allowedDtypes);
    total += record.nbytes;
    if (!Number.isSafeInteger(total) || total > totalBytes) invalid('Invalid tensor parameter range.');
  }
  if (total !== totalBytes) invalid('Missing tensor parameter data.');
  total = 0;
  for (const [i, shard] of index.shards.entries()) {
    const path = `weights/shard-${String(i).padStart(4, '0')}.bin`;
    if (
      shard?.path !== path ||
      !Number.isSafeInteger(shard.nbytes) ||
      shard.nbytes < 4 ||
      shard.nbytes > maxShardBytes ||
      shard.nbytes % 4 ||
      shard.nbytes !== manifest.files?.[path]?.bytes
    )
      invalid('Invalid tensor shard metadata.');
    total += shard.nbytes;
  }
  if (total !== totalBytes) invalid('Missing tensor shard data.');
  return index;
}
