import {sha256, waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {
  MANIFEST_SHA256,
  MANIFEST_BYTES,
  parsePinnedManifest,
  TOKENIZER_CONFIG_SHA256,
  TOKENIZER_SHA256,
  WASM_BYTES,
  WASM_SHA256
} from './manifest.mjs';
import {RangeCacheTransaction} from './range-cache.mjs';

const MAX_SMALL = 20 * 1024 * 1024;
const invalid = (message) => new WebDecisionError('INVALID_ASSET', message);
const range = (offset, length, sha256) => Object.freeze({offset, length, sha256});

const PINNED_SMALL_FILES = Object.freeze([
  Object.freeze({
    id: 'manifest',
    path: 'fused-ordered-manifest-4096-candidate-v1.json',
    bytes: MANIFEST_BYTES,
    sha256: MANIFEST_SHA256,
    chunks: Object.freeze([range(0, MANIFEST_BYTES, MANIFEST_SHA256)])
  }),
  Object.freeze({
    id: 'wasm',
    path: 'full-fused-tvm-4096-candidate-v1/model.wasm',
    bytes: WASM_BYTES,
    sha256: WASM_SHA256,
    chunks: Object.freeze([range(0, WASM_BYTES, WASM_SHA256)])
  }),
  Object.freeze({
    id: 'tokenizer',
    path: 'checkpoint/tokenizer.json',
    bytes: 19989592,
    sha256: TOKENIZER_SHA256,
    chunks: Object.freeze([
      range(0, 8388608, '4ca2408791853d31f4049546249a60dd5b4f034201f3dfd391916d0c999161af'),
      range(8388608, 8388608, '2d6f0488857832d7ff1d0cebc284f152bacafab14692fb5cf764c796c87898a2'),
      range(16777216, 3212376, 'c5f763e067ed597a2ee4c6feb566bc6484935a672f6bcc9a7eb6dbca83d5d88a')
    ])
  }),
  Object.freeze({
    id: 'tokenizerConfig',
    path: 'checkpoint/tokenizer_config.json',
    bytes: 1128,
    sha256: TOKENIZER_CONFIG_SHA256,
    chunks: Object.freeze([range(0, 1128, TOKENIZER_CONFIG_SHA256)])
  })
]);

function smallRanges(files) {
  if (
    !Array.isArray(files) ||
    files.length !== 4 ||
    new Set(files.map((file) => file.id)).size !== 4 ||
    !['manifest', 'wasm', 'tokenizer', 'tokenizerConfig'].every((id) => files.some((file) => file.id === id))
  )
    throw invalid('Unexpected small-asset set.');
  const specs = [];
  for (const file of files) {
    if (
      !Number.isSafeInteger(file.bytes) ||
      file.bytes < 1 ||
      file.bytes > MAX_SMALL ||
      !/^[0-9a-f]{64}$/.test(file.sha256) ||
      !Array.isArray(file.chunks) ||
      !file.chunks.length
    )
      throw invalid('Invalid small-asset metadata.');
    let offset = 0;
    for (const chunk of file.chunks) {
      if (chunk.offset !== offset) throw invalid('Small-asset ranges must be contiguous.');
      specs.push({path: file.path, totalBytes: file.bytes, offset, length: chunk.length, sha256: chunk.sha256});
      offset += chunk.length;
    }
    if (offset !== file.bytes) throw invalid('Small-asset ranges do not cover the file.');
  }
  return specs;
}

function weightRanges(manifest) {
  if (!Array.isArray(manifest?.records)) throw invalid('Model manifest has no tensor records.');
  return manifest.records.flatMap((record) =>
    record.chunks.map((chunk) => ({
      path: record.sourcePath,
      totalBytes: record.sourceFileBytes,
      offset: record.sourceOffset + chunk.offset,
      length: chunk.bytes,
      sha256: chunk.sha256
    }))
  );
}

/** Generic composition is exported only for tiny fixtures; production uses the pinned wrapper below. */
async function openRangedSource({base, cache, offline = false, signal, markerId, files, parseManifest}) {
  const small = smallRanges(files),
    transaction = new RangeCacheTransaction({base, cache, offline, signal, markerId});
  await transaction.open();
  transaction.require(small);
  const bytes = {};
  for (const file of files) {
    const data = new Uint8Array(file.bytes);
    for (const chunk of file.chunks) {
      const result = await transaction.read(base, {
        path: file.path,
        totalBytes: file.bytes,
        offset: chunk.offset,
        length: chunk.length,
        sha256: chunk.sha256
      });
      data.set(new Uint8Array(result.bytes), chunk.offset);
    }
    const digest = await waitForAbort(() => sha256(data), signal);
    signal?.throwIfAborted();
    if (digest !== file.sha256) throw invalid(`Small-asset hash mismatch: ${file.id}.`);
    bytes[file.id] = data;
  }
  const manifest = await parseManifest(bytes.manifest, signal);
  transaction.require(weightRanges(manifest));
  transaction.seal();
  return {bytes, manifest, transaction};
}

export function openPinnedRangeSource({base, cache, offline, signal}) {
  return openRangedSource({
    base,
    cache,
    offline,
    signal,
    files: PINNED_SMALL_FILES,
    markerId: `strands-fused-fp16:${MANIFEST_SHA256}`,
    parseManifest: parsePinnedManifest
  });
}
