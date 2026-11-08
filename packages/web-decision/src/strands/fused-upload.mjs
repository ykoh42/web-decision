import {readVerifiedRange} from './range-reader.mjs';

/** Upload one source-file region with a verified, bounded range per chunk. */
export async function uploadFusedRecord(
  base,
  record,
  context,
  pointer,
  byteOffset,
  sync,
  check,
  withWrite = async (work) => work(),
  {signal, readRange = readVerifiedRange} = {}
) {
  if (
    ![record?.sourceOffset, record?.sourceFileBytes, record?.bytes, byteOffset].every(Number.isSafeInteger) ||
    record.sourceOffset < 0 ||
    record.sourceOffset % 4 ||
    record.bytes <= 0 ||
    record.bytes % 4 ||
    record.sourceFileBytes < record.sourceOffset + record.bytes ||
    byteOffset < 0 ||
    byteOffset % 4 ||
    !Array.isArray(record.chunks) ||
    !record.chunks.length
  )
    throw new Error('Invalid fused tensor range metadata');
  let offset = 0;
  for (const chunk of record.chunks) {
    signal?.throwIfAborted();
    check();
    if (
      !Number.isSafeInteger(chunk.offset) ||
      chunk.offset !== offset ||
      !Number.isSafeInteger(chunk.bytes) ||
      chunk.bytes <= 0 ||
      chunk.bytes > 8 * 1024 * 1024 ||
      chunk.bytes % 4 ||
      offset + chunk.bytes > record.bytes
    )
      throw new Error('Invalid fused tensor chunk metadata');
    const response = await readRange(
      base,
      {
        path: record.sourcePath,
        offset: record.sourceOffset + offset,
        totalBytes: record.sourceFileBytes,
        length: chunk.bytes,
        sha256: chunk.sha256
      },
      {signal}
    );
    const data = new Uint8Array(response.bytes);
    if (data.length !== chunk.bytes) throw new Error('Verified range has the wrong length');
    signal?.throwIfAborted();
    check();
    await withWrite(async () => {
      context.copyRawBytesToBuffer(data, pointer, byteOffset + offset, data.length);
      await sync();
    });
    offset += data.length;
    signal?.throwIfAborted();
    check();
  }
  if (offset !== record.bytes) throw new Error('Fused ranges did not cover the tensor');
}
