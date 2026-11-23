import pinnedInventory from '../../assets/strands-gemma4-v2/source-inventory.json' with {type: 'json'};
import {sha256, type AssetCache} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {parseSafetensorsHeader} from '../kanana/safetensors.mjs';
import {readVerifiedRange} from './range-reader.mjs';
import {RangeCacheTransaction, type RangeCacheSpec} from './range-cache.mjs';

export const GEMMA4_V2_REPOSITORY = 'StrandsAgents/strands-decider-E2B-gemma4-v2-2611';
export const GEMMA4_V2_REVISION = 'a5f32ea1bffe2e91c25c0c641d444e8115e28212';
const INVENTORY_SHA256 = 'c4c1f9ca91d272d88f2559dd186fa0d37e878fa70d378c44a20de978179366aa';
const WORD = 'model.language_model.embed_tokens.weight';
const PER_LAYER = 'model.language_model.embed_tokens_per_layer.weight';
const LAYERS = 35;
const PER_LAYER_WIDTH = 256;
const PLE_ROW_BYTES = LAYERS * PER_LAYER_WIDTH * Uint16Array.BYTES_PER_ELEMENT;
const MAX_RANGE = 8 * 1024 * 1024;

type Range = readonly [offset: number, length: number, sha256: string];
type Tensor = Readonly<{offset: number; length: number; shape: readonly number[]; dtype: string}>;
type Inventory = Readonly<{
  format: string;
  repository: string;
  revision: string;
  file: string;
  bytes: number;
  sha256: string;
  headerBytes: number;
  chunks: readonly Range[];
  tensors: Readonly<Record<string, Tensor>>;
}>;
export type Gemma4V2SourceOptions = Readonly<{
  cache?: AssetCache | null;
  offline?: boolean;
  signal?: AbortSignal | null;
}>;
const invalid = (message: string) => new WebDecisionError('INVALID_ASSET', message);

function validateInventory(inventory: Inventory): void {
  if (
    inventory.format !== 'web-decision-strands-gemma4-v2-source-v1' ||
    inventory.repository !== GEMMA4_V2_REPOSITORY ||
    inventory.revision !== GEMMA4_V2_REVISION ||
    inventory.file !== 'model.safetensors' ||
    !Number.isSafeInteger(inventory.bytes) ||
    inventory.bytes < 8 ||
    !Number.isSafeInteger(inventory.headerBytes) ||
    inventory.headerBytes < 8 ||
    inventory.headerBytes > inventory.bytes ||
    !/^[0-9a-f]{64}$/u.test(inventory.sha256) ||
    !Array.isArray(inventory.chunks)
  )
    throw invalid('Invalid Gemma 4 v2 source inventory.');
  let end = 0;
  for (const chunk of inventory.chunks) {
    if (
      !Array.isArray(chunk) ||
      chunk.length !== 3 ||
      chunk[0] !== end ||
      !Number.isSafeInteger(chunk[1]) ||
      chunk[1] < 1 ||
      chunk[1] > MAX_RANGE ||
      !/^[0-9a-f]{64}$/u.test(chunk[2])
    )
      throw invalid('Invalid Gemma 4 v2 source range.');
    end += chunk[1];
    if (!Number.isSafeInteger(end) || end > inventory.bytes) throw invalid('Source ranges exceed the file.');
  }
  if (end !== inventory.bytes || !inventory.tensors || typeof inventory.tensors !== 'object')
    throw invalid('Incomplete Gemma 4 v2 source inventory.');
}

function findChunk(chunks: readonly Range[], offset: number): number {
  let left = 0;
  let right = chunks.length;
  while (left < right) {
    const mid = Math.floor((left + right) / 2);
    if (chunks[mid]![0] + chunks[mid]![1] <= offset) left = mid + 1;
    else right = mid;
  }
  if (left >= chunks.length || chunks[left]![0] > offset)
    throw invalid('Source range does not cover the requested byte.');
  return left;
}

function verifyHeader(bytes: Uint8Array, inventory: Inventory): void {
  const header = parseSafetensorsHeader(bytes, inventory.bytes);
  const records = Object.entries(inventory.tensors);
  if (header.dataOffset !== inventory.headerBytes || header.tensorCount !== records.length)
    throw invalid('Gemma 4 v2 source header does not match the inventory.');
  for (const tensor of header.tensors) {
    const expected = inventory.tensors[tensor.name];
    if (
      !expected ||
      expected.dtype !== tensor.dtype ||
      expected.offset !== tensor.offset ||
      expected.length !== tensor.length ||
      expected.shape.length !== tensor.shape.length ||
      expected.shape.some((size, index) => size !== tensor.shape[index])
    )
      throw invalid('Gemma 4 v2 source tensor does not match the inventory.');
  }
}

function combineSignals(...signals: (AbortSignal | null | undefined)[]) {
  const active = [...new Set(signals.filter((signal): signal is AbortSignal => signal != null))];
  if (active.length < 2) return {signal: active[0], close() {}};
  const controller = new AbortController();
  const listeners = active.map((signal) => {
    const abort = () => controller.abort(signal.reason);
    signal.addEventListener('abort', abort, {once: true});
    return {signal, abort};
  });
  const aborted = active.find((signal) => signal.aborted);
  if (aborted) controller.abort(aborted.reason);
  return {
    signal: controller.signal,
    close() {
      for (const item of listeners) item.signal.removeEventListener('abort', item.abort);
    }
  };
}

function requiredSourceRanges(inventory: Inventory): RangeCacheSpec[] {
  const requiredIndices = new Set<number>();
  function requireSpan(start: number, length: number): void {
    let offset = start;
    const end = start + length;
    while (offset < end) {
      const index = findChunk(inventory.chunks, offset);
      requiredIndices.add(index);
      const [chunkStart, chunkLength] = inventory.chunks[index]!;
      offset = chunkStart + chunkLength;
    }
  }
  requireSpan(0, inventory.headerBytes);
  for (const [name, tensor] of Object.entries(inventory.tensors)) {
    if (
      name.startsWith('model.language_model.') ||
      name.startsWith('model.vision_tower.') ||
      name.startsWith('model.embed_vision.')
    )
      requireSpan(tensor.offset, tensor.length);
  }
  return [...requiredIndices]
    .sort((a, b) => a - b)
    .map((index) => {
      const [offset, length, digest] = inventory.chunks[index]!;
      return {path: inventory.file, totalBytes: inventory.bytes, offset, length, sha256: digest};
    });
}

type EmbeddingRow = {output: number; offset: number};

function copyLayerEmbeddingRows(
  packed: Uint32Array,
  layerWords: number,
  bytes: Uint8Array,
  rows: readonly EmbeddingRow[]
): void {
  const words = PER_LAYER_WIDTH / 2;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (const {output, offset} of rows) {
    for (let layer = 0; layer < LAYERS; layer++) {
      const from = offset + layer * PER_LAYER_WIDTH * Uint16Array.BYTES_PER_ELEMENT;
      const to = layer * layerWords + output * words;
      for (let word = 0; word < words; word++) packed[to + word] = view.getUint32(from + word * 4, true);
    }
  }
}

function embeddingGroups(
  inventory: Inventory,
  tensor: Tensor,
  ids: readonly number[] | Int32Array,
  width: number,
  layer?: number
) {
  const groups = new Map<number, EmbeddingRow[]>();
  for (let row = 0; row < ids.length; row++) {
    const id = ids[row]!;
    if (!Number.isSafeInteger(id) || id < 0 || id >= tensor.shape[0]!)
      throw new WebDecisionError('INVALID_INPUT', 'Token ID is outside the model vocabulary.');
    const offset = tensor.offset + id * tensor.shape[1]! * 2 + (layer ?? 0) * PER_LAYER_WIDTH * 2;
    const index = findChunk(inventory.chunks, offset);
    const [start, length] = inventory.chunks[index]!;
    if (offset + width * 2 > start + length) throw invalid('An embedding row crosses a source range.');
    const entries = groups.get(index) ?? [];
    entries.push({output: row, offset: offset - start});
    groups.set(index, entries);
  }
  return groups;
}

/** Internal test seam; normal loading uses the pinned source below. */
export async function openGemma4V2SourceFromInventory(
  value: Inventory,
  expectedHash: string,
  base: URL,
  options: Gemma4V2SourceOptions = {}
) {
  options.signal?.throwIfAborted();
  const encoded = new TextEncoder().encode(JSON.stringify(value) + '\n');
  if ((await sha256(encoded)) !== expectedHash) throw invalid('Gemma 4 v2 source inventory digest differs.');
  options.signal?.throwIfAborted();
  const inventory = JSON.parse(new TextDecoder().decode(encoded)) as Inventory;
  validateInventory(inventory);
  if (
    !(base instanceof URL) ||
    (options.offline !== undefined && typeof options.offline !== 'boolean') ||
    (options.offline && !options.cache)
  )
    throw invalid('Invalid Gemma 4 v2 source options.');
  const sourceBase = new URL(base.href);
  const cache = options.cache;
  const offline = options.offline;
  const initialSignal = options.signal;
  const closed = new AbortController();
  const assertOpen = () => {
    if (closed.signal.aborted) throw new WebDecisionError('SESSION_CLOSED', 'The Gemma 4 v2 source is closed.');
  };
  const resident = new Map<number, Uint8Array>();
  let residentGeneration = 0;
  const requiredRanges = requiredSourceRanges(inventory);
  const readChunk = async (index: number, signal?: AbortSignal | null) => {
    assertOpen();
    initialSignal?.throwIfAborted();
    signal?.throwIfAborted();
    const cached = resident.get(index);
    if (cached) {
      resident.delete(index);
      resident.set(index, cached);
      return cached;
    }
    const [offset, length, digest] = inventory.chunks[index]!;
    const generation = residentGeneration;
    const combined = combineSignals(initialSignal, signal, closed.signal);
    try {
      const result = await readVerifiedRange(
        sourceBase,
        {path: inventory.file, totalBytes: inventory.bytes, offset, length, sha256: digest},
        {cache, offline, signal: combined.signal}
      );
      combined.signal?.throwIfAborted();
      assertOpen();
      const bytes = new Uint8Array(result.bytes);
      if (generation === residentGeneration) {
        resident.set(index, bytes);
        if (resident.size > 4) resident.delete(resident.keys().next().value!);
      }
      return bytes;
    } finally {
      combined.close();
    }
  };
  let headerVerified = false;
  async function ensureHeader(signal?: AbortSignal | null): Promise<void> {
    assertOpen();
    initialSignal?.throwIfAborted();
    signal?.throwIfAborted();
    if (headerVerified) return;
    const parts: Uint8Array[] = [];
    let size = 0;
    while (size < inventory.headerBytes) {
      const part = await readChunk(findChunk(inventory.chunks, size), signal);
      parts.push(part);
      size += part.length;
    }
    const header = new Uint8Array(inventory.headerBytes);
    let cursor = 0;
    for (const part of parts) {
      const length = Math.min(part.length, header.length - cursor);
      header.set(part.subarray(0, length), cursor);
      cursor += length;
    }
    verifyHeader(header, inventory);
    headerVerified = true;
  }
  async function readTensorRange(
    name: string,
    start: number,
    length: number,
    signal?: AbortSignal | null
  ): Promise<Uint8Array> {
    const tensor = inventory.tensors[name];
    if (!tensor || name === WORD || name === PER_LAYER) throw invalid('Unknown or row-addressed Gemma 4 v2 tensor.');
    if (
      !Number.isSafeInteger(start) ||
      start < 0 ||
      !Number.isSafeInteger(length) ||
      length < 0 ||
      start > tensor.length - length
    )
      throw invalid('Tensor range is outside the Gemma 4 v2 source.');
    await ensureHeader(signal);
    const output = new Uint8Array(length);
    let position = tensor.offset + start;
    const end = position + length;
    while (position < end) {
      const index = findChunk(inventory.chunks, position);
      const [offset, chunkLength] = inventory.chunks[index]!;
      const bytes = await readChunk(index, signal);
      const count = Math.min(end - position, offset + chunkLength - position);
      output.set(bytes.subarray(position - offset, position - offset + count), position - tensor.offset - start);
      position += count;
    }
    return output;
  }
  async function readTensor(name: string, signal?: AbortSignal | null): Promise<Uint8Array> {
    const tensor = inventory.tensors[name];
    if (!tensor) throw invalid('Unknown Gemma 4 v2 tensor.');
    return readTensorRange(name, 0, tensor.length, signal);
  }
  function embeddingTensor(layer: number | undefined) {
    const tensor = inventory.tensors[layer === undefined ? WORD : PER_LAYER];
    if (
      !tensor ||
      tensor.dtype !== 'BF16' ||
      tensor.shape.length !== 2 ||
      tensor.shape[0] !== 262144 ||
      tensor.shape[1] !== (layer === undefined ? 1536 : LAYERS * PER_LAYER_WIDTH) ||
      (layer !== undefined && (!Number.isInteger(layer) || layer < 0 || layer >= LAYERS))
    )
      throw invalid('Invalid Gemma 4 v2 embedding metadata.');
    return tensor;
  }

  async function readEmbeddingRows(
    ids: readonly number[] | Int32Array,
    layer?: number,
    signal?: AbortSignal | null
  ): Promise<Uint32Array> {
    await ensureHeader(signal);
    const tensor = embeddingTensor(layer);
    const width = layer === undefined ? 1536 : PER_LAYER_WIDTH;
    const groups = embeddingGroups(inventory, tensor, ids, width, layer);
    const words = width / 2;
    const output = new Uint32Array(ids.length * words);
    for (const [index, rows] of groups) {
      signal?.throwIfAborted();
      const bytes = await readChunk(index, signal);
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      for (const row of rows)
        for (let word = 0; word < words; word++)
          output[row.output * words + word] = view.getUint32(row.offset + word * 4, true);
    }
    return output;
  }
  async function readAllLayerEmbeddingRows(
    ids: readonly number[] | Int32Array,
    memoryBudget: number,
    signal?: AbortSignal | null
  ): Promise<readonly Uint32Array[] | undefined> {
    assertOpen();
    initialSignal?.throwIfAborted();
    signal?.throwIfAborted();
    if (ids.length * PLE_ROW_BYTES > memoryBudget) return undefined;
    const selectedIds = Array.from(ids);
    await ensureHeader(signal);
    const groups = embeddingGroups(inventory, embeddingTensor(0), selectedIds, PER_LAYER_WIDTH);
    // The pinned inventory aligns ranges to complete rows. Other layouts can use the per-layer reader.
    for (const [index, rows] of groups) {
      if (rows.some(({offset}) => offset + PLE_ROW_BYTES > inventory.chunks[index]![1])) return undefined;
    }
    const layerWords = selectedIds.length * (PER_LAYER_WIDTH / 2);
    const packed = new Uint32Array(LAYERS * layerWords);
    for (const [index, rows] of groups) {
      const bytes = await readChunk(index, signal);
      copyLayerEmbeddingRows(packed, layerWords, bytes, rows);
    }
    signal?.throwIfAborted();
    return Array.from({length: LAYERS}, (_, layer) => packed.subarray(layer * layerWords, (layer + 1) * layerWords));
  }

  async function withOfflineTransaction<T>(
    signal: AbortSignal | null | undefined,
    operation: (transaction: RangeCacheTransaction, combined: AbortSignal | undefined) => Promise<T>
  ): Promise<T> {
    assertOpen();
    if (!cache) throw new WebDecisionError('ASSET_LOAD_FAILED', 'A cache is required for offline weight preparation.');
    const combined = combineSignals(initialSignal, signal, closed.signal);
    const transaction = new RangeCacheTransaction({
      base: sourceBase,
      cache,
      signal: combined.signal,
      markerId: `strands-gemma4-v2-weights:${expectedHash}`,
      markerPath: 'strands-gemma4-v2-weights-offline-v1.json'
    });
    try {
      await transaction.open();
      transaction.require(requiredRanges);
      transaction.seal();
      return await operation(transaction, combined.signal);
    } finally {
      transaction.dispose();
      combined.close();
    }
  }
  async function prepareWeightsOffline({
    signal,
    onProgress
  }: {signal?: AbortSignal | null; onProgress?: (bytes: number, total: number) => void} = {}): Promise<void> {
    if (offline)
      throw new WebDecisionError('ASSET_LOAD_FAILED', 'Offline weight preparation requires a network connection.');
    await withOfflineTransaction(signal, async (transaction, combined) => {
      await ensureHeader(combined);
      const total = requiredRanges.reduce((sum, range) => sum + range.length, 0);
      let loaded = 0;
      for (const range of requiredRanges) {
        combined?.throwIfAborted();
        if (!(await transaction.read(sourceBase, range, {signal: combined})).cachedByHash)
          throw new WebDecisionError('ASSET_LOAD_FAILED', 'A verified weight range could not be cached.');
        loaded += range.length;
        onProgress?.(loaded, total);
      }
      if (!(await transaction.commit({signal: combined})))
        throw new WebDecisionError('ASSET_LOAD_FAILED', 'The verified weights could not be committed for offline use.');
    });
  }
  // Check the completion marker and range presence; reads still verify each range's hash.
  async function hasCompleteOfflineWeights(signal?: AbortSignal | null): Promise<boolean> {
    return withOfflineTransaction(signal, (transaction, combined) =>
      transaction.hasCommittedRanges({signal: combined})
    );
  }
  // Explicitly rehash the full cached model; this reads approximately 9 GiB.
  async function verifyOfflineWeights(signal?: AbortSignal | null): Promise<boolean> {
    return withOfflineTransaction(signal, async (transaction, combined) => {
      if (!(await transaction.hasCommittedRanges({signal: combined}))) return false;
      for (const range of requiredRanges) {
        try {
          await readVerifiedRange(sourceBase, range, {cache, offline: true, signal: combined});
        } catch {
          combined?.throwIfAborted();
          return false;
        }
      }
      return true;
    });
  }
  function dispose(): void {
    if (closed.signal.aborted) return;
    closed.abort(new WebDecisionError('SESSION_CLOSED', 'The Gemma 4 v2 source is closed.'));
    resident.clear();
  }
  return Object.freeze({
    readTensor,
    readTensorRange,
    readEmbeddingRows,
    readAllLayerEmbeddingRows,
    prepareWeightsOffline,
    hasCompleteOfflineWeights,
    verifyOfflineWeights,
    releaseCachedMemory() {
      assertOpen();
      residentGeneration++;
      resident.clear();
    },
    dispose
  });
}

export function openPinnedGemma4V2Source(options?: Gemma4V2SourceOptions) {
  return openGemma4V2SourceFromInventory(
    pinnedInventory as unknown as Inventory,
    INVENTORY_SHA256,
    new URL(`https://huggingface.co/${GEMMA4_V2_REPOSITORY}/resolve/${GEMMA4_V2_REVISION}/`),
    options
  );
}
