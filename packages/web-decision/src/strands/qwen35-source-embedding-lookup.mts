import {waitForAbort} from '../abort.js';
import {WebDecisionError} from '../kev.js';
import {validateQwen35WordAssetManifest} from './qwen35-assets.mjs';
import type {Qwen35WordTensor} from './qwen35-assets.mjs';
import {
  QWEN35_SOURCE_MAX_RANGE_BYTES,
  type Qwen35PinnedSource,
  type Qwen35SourceTensorReadOptions
} from './qwen35-source.mts';

const EMBEDDING_TENSOR_NAME = 'model.language_model.embed_tokens.weight';
const EMBEDDING_SHAPE = [248_320, 2_048] as const;
const invalid = (message: string) => new WebDecisionError('INVALID_ASSET', message);

export interface Qwen35SourceEmbeddingLookupOptions {
  readonly maxCachedRanges?: number | undefined;
  readonly signal?: AbortSignal | null | undefined;
}

export interface Qwen35SourceEmbeddingLookup {
  readonly lookupTokenEmbeddings: (
    tokenIds: Int32Array,
    options?: Readonly<{signal?: AbortSignal | null | undefined}>
  ) => Promise<Float32Array>;
  releaseCachedMemory(): void;
  dispose(): void;
}

export interface Qwen35SourceEmbeddingTable {
  readonly tensor: Qwen35WordTensor;
  readonly sourceFileBytes: number;
  readonly tensorFileOffset: number;
  readonly sourceRangeBytes: number;
}

interface RowFragment {
  readonly sourceOffset: number;
  readonly targetOffset: number;
  readonly bytes: number;
}

function validateEmbeddingTable(table: Qwen35SourceEmbeddingTable): number {
  const tensor = table?.tensor;
  if (
    !tensor ||
    !Array.isArray(tensor.shape) ||
    tensor.shape.length !== 2 ||
    tensor.shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension < 1) ||
    tensor.dtype !== 'BF16' ||
    !Number.isSafeInteger(tensor.elementCount) ||
    tensor.elementCount !== tensor.shape[0]! * tensor.shape[1]! ||
    !Number.isSafeInteger(tensor.sourceByteCount) ||
    tensor.sourceByteCount !== tensor.elementCount * 2 ||
    !Number.isSafeInteger(table.sourceFileBytes) ||
    table.sourceFileBytes < 1 ||
    !Number.isSafeInteger(table.tensorFileOffset) ||
    table.tensorFileOffset < 0 ||
    !Number.isSafeInteger(table.sourceRangeBytes) ||
    table.sourceRangeBytes < 1 ||
    table.sourceRangeBytes > QWEN35_SOURCE_MAX_RANGE_BYTES ||
    table.tensorFileOffset > table.sourceFileBytes - tensor.sourceByteCount
  )
    throw invalid('Invalid pinned Qwen3.5 source embedding table.');
  return tensor.shape[1]!;
}

function groupEmbeddingFragments(
  tokenIds: Int32Array,
  table: Qwen35SourceEmbeddingTable,
  hiddenSize: number,
  vocabularySize: number
): Map<number, RowFragment[]> {
  const rowBytes = hiddenSize * 2;
  const fragmentsByWindow = new Map<number, RowFragment[]>();
  for (let row = 0; row < tokenIds.length; row += 1) {
    const tokenId = tokenIds[row]!;
    if (tokenId < 0 || tokenId >= vocabularySize)
      throw invalid(`Qwen3.5 token ID ${tokenId} is outside the pinned embedding table.`);
    let fileOffset = table.tensorFileOffset + tokenId * rowBytes;
    let targetOffset = row * hiddenSize;
    const rowEnd = fileOffset + rowBytes;
    while (fileOffset < rowEnd) {
      const windowIndex = Math.floor(fileOffset / table.sourceRangeBytes);
      const windowStart = windowIndex * table.sourceRangeBytes;
      const bytes = Math.min(rowEnd, windowStart + table.sourceRangeBytes) - fileOffset;
      if (bytes < 1 || bytes % 2 !== 0) throw invalid('Qwen3.5 BF16 source embedding row was split between values.');
      const fragments = fragmentsByWindow.get(windowIndex) ?? [];
      fragments.push({
        sourceOffset: fileOffset - Math.max(windowStart, table.tensorFileOffset),
        targetOffset,
        bytes
      });
      fragmentsByWindow.set(windowIndex, fragments);
      fileOffset += bytes;
      targetOffset += bytes / 2;
    }
  }
  return fragmentsByWindow;
}

/** Build an exact BF16 row gather over original Safetensors source windows. */
export function createQwen35SourceEmbeddingRowsLookup(
  source: Qwen35PinnedSource,
  table: Qwen35SourceEmbeddingTable,
  {maxCachedRanges = 2, signal: defaultSignal}: Qwen35SourceEmbeddingLookupOptions = {}
): Qwen35SourceEmbeddingLookup {
  const hiddenSize = validateEmbeddingTable(table);
  if (!source || typeof source.readTensorRange !== 'function')
    throw invalid('Qwen3.5 source embedding lookup requires a pinned tensor range reader.');
  if (!Number.isSafeInteger(maxCachedRanges) || maxCachedRanges < 0)
    throw new RangeError('Qwen3.5 source embedding cache size must be a non-negative integer.');

  const tensor = table.tensor;
  const tensorEnd = table.tensorFileOffset + tensor.sourceByteCount;
  const cache = new Map<number, Uint8Array>();
  let closed = false;
  let generation = 0;

  function remember(index: number, bytes: Uint8Array, expectedGeneration: number): void {
    if (maxCachedRanges === 0 || closed || generation !== expectedGeneration) return;
    cache.delete(index);
    cache.set(index, bytes);
    while (cache.size > maxCachedRanges) cache.delete(cache.keys().next().value!);
  }

  async function readWindow(
    index: number,
    signal?: AbortSignal | null | undefined
  ): Promise<Readonly<{tensorOffset: number; bytes: Uint8Array}>> {
    signal?.throwIfAborted();
    if (closed) throw new WebDecisionError('SESSION_CLOSED', 'The Qwen3.5 source embedding lookup has been disposed.');
    const rangeStart = index * table.sourceRangeBytes;
    const rangeEnd = Math.min(rangeStart + table.sourceRangeBytes, table.sourceFileBytes);
    const fileStart = Math.max(rangeStart, table.tensorFileOffset);
    const fileEnd = Math.min(rangeEnd, tensorEnd);
    if (fileStart >= fileEnd) throw invalid('Qwen3.5 embedding range window does not overlap its tensor.');
    const tensorOffset = fileStart - table.tensorFileOffset;
    const expectedLength = fileEnd - fileStart;
    const cached = cache.get(index);
    if (cached) {
      cache.delete(index);
      cache.set(index, cached);
      if (cached.byteLength !== expectedLength)
        throw invalid('Cached Qwen3.5 source embedding range has an unexpected size.');
      return Object.freeze({tensorOffset, bytes: cached});
    }

    const expectedGeneration = generation;
    const readOptions: Qwen35SourceTensorReadOptions = {signal};
    const bytes = await waitForAbort(
      () => source.readTensorRange('base', EMBEDDING_TENSOR_NAME, tensorOffset, expectedLength, readOptions),
      signal ?? undefined
    );
    signal?.throwIfAborted();
    if (bytes.byteLength !== expectedLength) throw invalid('Qwen3.5 source embedding range has an unexpected size.');
    remember(index, bytes, expectedGeneration);
    return Object.freeze({tensorOffset, bytes});
  }

  async function lookupTokenEmbeddings(
    tokenIds: Int32Array,
    options: Readonly<{signal?: AbortSignal | null | undefined}> = {}
  ): Promise<Float32Array> {
    if (closed) throw new WebDecisionError('SESSION_CLOSED', 'The Qwen3.5 source embedding lookup has been disposed.');
    if (!(tokenIds instanceof Int32Array))
      throw new TypeError('Qwen3.5 source embedding token IDs must be an Int32Array.');
    const signal = options.signal === undefined ? defaultSignal : options.signal;
    signal?.throwIfAborted();

    const vocabularySize = tensor.shape[0]!;
    const elementCount = tokenIds.length * hiddenSize;
    if (!Number.isSafeInteger(elementCount))
      throw invalid('Qwen3.5 source embedding output size exceeds the safe integer range.');
    const output = new Float32Array(elementCount);
    const outputBits = new Uint32Array(output.buffer);
    const fragmentsByWindow = groupEmbeddingFragments(tokenIds, table, hiddenSize, vocabularySize);

    for (const windowIndex of [...fragmentsByWindow.keys()].sort((left, right) => left - right)) {
      signal?.throwIfAborted();
      const window = await readWindow(windowIndex, signal);
      const view = new DataView(window.bytes.buffer, window.bytes.byteOffset, window.bytes.byteLength);
      for (const fragment of fragmentsByWindow.get(windowIndex)!) {
        const targetEnd = fragment.targetOffset + fragment.bytes / 2;
        let target = fragment.targetOffset;
        for (let sourceOffset = fragment.sourceOffset; target < targetEnd; sourceOffset += 2, target += 1) {
          outputBits[target] = view.getUint16(sourceOffset, true) << 16;
          if (!Number.isFinite(output[target]!))
            throw invalid('Qwen3.5 source embedding row contains a non-finite BF16 value.');
        }
      }
    }
    signal?.throwIfAborted();
    return output;
  }

  return Object.freeze({
    lookupTokenEmbeddings,
    releaseCachedMemory() {
      generation += 1;
      cache.clear();
    },
    dispose() {
      generation += 1;
      closed = true;
      cache.clear();
    }
  });
}

/** Bind to the pinned original Hugging Face Safetensors base checkpoint. */
export async function createQwen35SourceTokenEmbeddingLookup(
  source: Qwen35PinnedSource,
  options: Qwen35SourceEmbeddingLookupOptions = {}
): Promise<Qwen35SourceEmbeddingLookup> {
  options.signal?.throwIfAborted();
  if (!source || typeof source.readTensorRange !== 'function' || !source.manifest)
    throw invalid('Qwen3.5 source token lookup requires the pinned Safetensors source reader.');
  const manifest = await waitForAbort(
    () => validateQwen35WordAssetManifest(source.manifest),
    options.signal ?? undefined
  );
  options.signal?.throwIfAborted();
  const asset = manifest.assets.base;
  const tensor = asset.tensors.find(({name}) => name === EMBEDDING_TENSOR_NAME);
  if (
    !tensor ||
    tensor.dtype !== 'BF16' ||
    tensor.shape.length !== 2 ||
    tensor.shape[0] !== EMBEDDING_SHAPE[0] ||
    tensor.shape[1] !== EMBEDDING_SHAPE[1]
  )
    throw invalid('Pinned Qwen3.5 source lacks the expected BF16 token embedding table.');
  return createQwen35SourceEmbeddingRowsLookup(
    source,
    {
      tensor,
      sourceFileBytes: asset.source.bytes,
      tensorFileOffset: 8 + asset.source.headerBytes + tensor.sourceDataByteOffset,
      sourceRangeBytes: QWEN35_SOURCE_MAX_RANGE_BYTES
    },
    options
  );
}
