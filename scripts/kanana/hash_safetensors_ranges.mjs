#!/usr/bin/env node
import {createHash} from 'node:crypto';
import {createReadStream} from 'node:fs';
import {open, writeFile} from 'node:fs/promises';
import {basename, resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {KANANA_SAFETENSORS_SPEC} from '../../packages/web-decision/src/kanana/weight-source.mjs';
import {
  SAFETENSORS_MAX_HEADER_BYTES,
  inspectBf16ToFp16,
  parseSafetensorsHeader
} from '../../packages/web-decision/src/kanana/safetensors.mjs';

export const SAFETENSORS_HASH_CHUNK_BYTES = 8 * 1024 * 1024;

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

async function readExactly(file, buffer, offset, length, position) {
  let read = 0;
  while (read < length) {
    const result = await file.read(buffer, offset + read, length - read, position + read);
    if (!result.bytesRead) throw new Error('Unexpected EOF while reading Safetensors header.');
    read += result.bytesRead;
  }
}

function validateHashInputs(filePath, path, chunkBytes, verifyBf16ToFp16) {
  if (
    typeof filePath !== 'string' ||
    !filePath ||
    typeof path !== 'string' ||
    !path ||
    !Number.isSafeInteger(chunkBytes) ||
    chunkBytes < 8 ||
    chunkBytes > SAFETENSORS_HASH_CHUNK_BYTES ||
    chunkBytes % 8 !== 0 ||
    typeof verifyBf16ToFp16 !== 'boolean'
  )
    throw new TypeError('Safetensors hash inputs or chunk size are invalid.');
}

async function readSafetensorsHeader(filePath) {
  const file = await open(filePath, 'r');
  try {
    const totalBytes = (await file.stat()).size;
    if (!Number.isSafeInteger(totalBytes) || totalBytes < 8) throw new Error('Safetensors file is too small.');
    const prefix = new Uint8Array(8);
    await readExactly(file, prefix, 0, prefix.byteLength, 0);
    const headerLengthBig = new DataView(prefix.buffer).getBigUint64(0, true);
    if (headerLengthBig > BigInt(SAFETENSORS_MAX_HEADER_BYTES)) throw new Error('Safetensors header is too large.');

    const headerLength = Number(headerLengthBig);
    const headerRange = new Uint8Array(8 + headerLength);
    headerRange.set(prefix);
    await readExactly(file, headerRange, 8, headerLength, 8);
    return {headerRange, parsed: parseSafetensorsHeader(headerRange, totalBytes), totalBytes};
  } finally {
    await file.close();
  }
}

function conversionStatistics() {
  return {invalidValueCount: 0, overflowValueCount: 0, nonFiniteValueCount: 0, firstFailure: undefined};
}

function inspectBf16Chunk(record, bytes, fileOffset, pendingByte, statistics) {
  let candidate = bytes;
  let candidateFileOffset = fileOffset;
  if (pendingByte !== undefined) {
    candidate = Buffer.concat([Buffer.of(pendingByte), bytes]);
    candidateFileOffset -= 1;
  }

  const completeLength = candidate.byteLength - (candidate.byteLength % 2);
  if (completeLength > 0) {
    const inspected = inspectBf16ToFp16(candidate.subarray(0, completeLength));
    statistics.invalidValueCount += inspected.invalidValueCount;
    statistics.overflowValueCount += inspected.overflowValueCount;
    statistics.nonFiniteValueCount += inspected.nonFiniteValueCount;
    if (inspected.firstFailure && !statistics.firstFailure) {
      const failureOffset = candidateFileOffset + inspected.firstFailure.byteOffset;
      statistics.firstFailure = {
        tensor: record.name,
        sourceDtype: 'BF16',
        targetDtype: 'float16',
        fileOffset: failureOffset,
        tensorOffset: failureOffset - record.offset,
        reason: inspected.firstFailure.reason,
        message: inspected.firstFailure.message
      };
    }
  }
  return completeLength < candidate.byteLength ? candidate[completeLength] : undefined;
}

function tensorConversion(statistics) {
  const failure = statistics.firstFailure;
  if (failure) {
    failure.invalidValueCount = statistics.invalidValueCount;
    failure.overflowValueCount = statistics.overflowValueCount;
    failure.nonFiniteValueCount = statistics.nonFiniteValueCount;
  }
  const conversion = {
    sourceDtype: 'BF16',
    targetDtype: 'float16',
    verified: true,
    rounding: 'nearest-even',
    valid: statistics.invalidValueCount === 0,
    invalidValueCount: statistics.invalidValueCount,
    overflowValueCount: statistics.overflowValueCount,
    nonFiniteValueCount: statistics.nonFiniteValueCount
  };
  if (failure) conversion.failure = failure;
  return conversion;
}

async function hashTensor(filePath, record, chunkBytes, verifyBf16ToFp16) {
  const chunks = [];
  const tensorDigest = createHash('sha256');
  const inspectConversion = verifyBf16ToFp16 && record.dtype === 'BF16';
  const statistics = conversionStatistics();
  let offset = record.offset;
  const end = record.offset + record.length;

  while (offset < end) {
    const length = Math.min(chunkBytes, end - offset);
    const chunkDigest = createHash('sha256');
    let streamOffset = 0;
    let pendingByte;
    for await (const chunk of createReadStream(filePath, {
      start: offset,
      end: offset + length - 1,
      highWaterMark: chunkBytes
    })) {
      chunkDigest.update(chunk);
      tensorDigest.update(chunk);
      if (inspectConversion)
        pendingByte = inspectBf16Chunk(record, chunk, offset + streamOffset, pendingByte, statistics);
      streamOffset += chunk.byteLength;
    }
    if (inspectConversion && pendingByte !== undefined)
      throw new Error(`BF16 tensor ${record.name} ended with an incomplete 16-bit value.`);

    chunks.push({offset, length, sha256: chunkDigest.digest('hex')});
    offset += length;
  }

  const tensor = {...record, sha256: tensorDigest.digest('hex'), chunks};
  if (inspectConversion) tensor.conversion = tensorConversion(statistics);
  return {tensor, statistics: inspectConversion ? statistics : undefined};
}

function addConversionStatistics(total, current) {
  total.invalidValueCount += current.invalidValueCount;
  total.overflowValueCount += current.overflowValueCount;
  total.nonFiniteValueCount += current.nonFiniteValueCount;
  if (current.firstFailure) total.failures.push(current.firstFailure);
}

function makeHashManifest(path, headerRange, parsed, totalBytes, tensors, verifyBf16ToFp16, statistics) {
  const manifest = {
    format: 'kanana-safetensors-range-manifest-v1',
    path,
    totalBytes,
    headerLength: parsed.headerLength,
    headerSha256: sha256(headerRange),
    tensorCount: parsed.tensorCount,
    tensors
  };
  if (verifyBf16ToFp16) {
    manifest.conversionVerification = {
      enabled: true,
      sourceDtype: 'BF16',
      targetDtype: 'float16',
      rounding: 'nearest-even',
      valid: statistics.invalidValueCount === 0,
      invalidValueCount: statistics.invalidValueCount,
      overflowValueCount: statistics.overflowValueCount,
      nonFiniteValueCount: statistics.nonFiniteValueCount,
      failures: statistics.failures
    };
  }
  return manifest;
}

/** Hash a Safetensors index with bounded per-range reads; no tensor is retained in memory. */
export async function hashSafetensorsFile(
  filePath,
  {path = basename(filePath), chunkBytes = SAFETENSORS_HASH_CHUNK_BYTES, verifyBf16ToFp16 = false} = {}
) {
  validateHashInputs(filePath, path, chunkBytes, verifyBf16ToFp16);
  const {headerRange, parsed, totalBytes} = await readSafetensorsHeader(filePath);
  const tensors = [];
  const statistics = {...conversionStatistics(), failures: []};
  for (const record of parsed.tensors) {
    const result = await hashTensor(filePath, record, chunkBytes, verifyBf16ToFp16);
    tensors.push(result.tensor);
    if (result.statistics) addConversionStatistics(statistics, result.statistics);
  }
  return makeHashManifest(path, headerRange, parsed, totalBytes, tensors, verifyBf16ToFp16, statistics);
}

function assertPinnedSource(manifest) {
  const spec = KANANA_SAFETENSORS_SPEC;
  if (
    manifest.totalBytes !== spec.totalBytes ||
    manifest.headerLength !== spec.headerLength ||
    manifest.headerSha256 !== spec.headerSha256 ||
    manifest.tensorCount !== spec.tensorCount
  )
    throw new Error('Input file does not match the pinned Kanana Safetensors checkpoint.');
  return {
    ...manifest,
    model: spec.model,
    repository: spec.repository,
    revision: spec.revision,
    path: spec.path
  };
}

async function main() {
  const args = process.argv.slice(2);
  const verifyBf16ToFp16 = args.includes('--verify-fp16-rn');
  const positional = args.filter((arg) => arg !== '--verify-fp16-rn');
  const [inputPath, outputPath] = positional;
  if (!inputPath || positional.length > 2)
    throw new Error('Usage: node hash_safetensors_ranges.mjs model.safetensors [manifest.json] [--verify-fp16-rn]');
  if (!verifyBf16ToFp16) throw new Error('Pass --verify-fp16-rn to generate the pinned Kanana range manifest.');
  const manifest = assertPinnedSource(
    await hashSafetensorsFile(resolve(inputPath), {
      path: KANANA_SAFETENSORS_SPEC.path,
      verifyBf16ToFp16
    })
  );
  const serialized = `${JSON.stringify(manifest, null, 2)}\n`;
  if (outputPath) await writeFile(resolve(outputPath), serialized);
  else process.stdout.write(serialized);
  if (verifyBf16ToFp16 && !manifest.conversionVerification.valid) {
    process.stderr.write(
      `BF16 to finite FP16 conversion failed for ${manifest.conversionVerification.failures.length} tensor(s).\n`
    );
    process.exitCode = 2;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href)
  main().catch((error) => {
    process.stderr.write(`${error?.message || String(error)}\n`);
    process.exitCode = 1;
  });
