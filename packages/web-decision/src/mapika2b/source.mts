import pinnedInventory from '../../assets/mapika2b/source-inventory.json' with {type: 'json'};
import {sha256, waitForAbort, type AssetCache} from '../assets.js';
import {readAssetBatch} from '../asset-batch.js';
import {WebDecisionError} from '../kev.js';
import {parseSafetensorsHeader} from '../kanana/safetensors.mjs';
import {RangeCacheTransaction, type RangeCacheReadOptions, type RangeCacheSpec} from '../strands/range-cache.mjs';
import {MAPIKA2B_REPOSITORY, MAPIKA2B_REVISION} from './contract.mts';

export const MAPIKA2B_SOURCE_INVENTORY_SHA256 = '151295ce4b89e7dc049e1fb7420fda52670772cf391c27fc741e90182f617a60';
const format = 'web-decision-mapika-2b-v12-source-inventory-v1';
const embeddingName = 'model.language_model.embed_tokens.weight';
const filenames = {text: 'model.safetensors', vision: 'vision/visual.safetensors'} as const;
const assetIds = ['text', 'vision'] as const;
const prefetchCount = 4;
type AssetId = (typeof assetIds)[number];
type SourceRange = Readonly<{offset: number; length: number; sha256: string}>;
type SourceTensor = Readonly<
  SourceRange & {name: string; shape: readonly number[]; dtype: string; chunks: readonly SourceRange[]}
>;
type SourceAsset = Readonly<{
  file: string;
  bytes: number;
  headerRangeBytes: number;
  headerRangeSha256: string;
  tensorCount: number;
  tensors: readonly SourceTensor[];
}>;
type SourceInventory = Readonly<{
  format: string;
  repository: string;
  revision: string;
  maxRangeBytes: number;
  weights: Readonly<Record<AssetId, SourceAsset>>;
}>;
type ParameterBinding = Readonly<{name: string; shape: readonly number[]}>;
type PrefetchOptions = RangeCacheReadOptions & Readonly<{onProgress?: (bytes: number) => void}>;
type AssetPlan = Readonly<{
  header: RangeCacheSpec;
  tensors: ReadonlyMap<string, SourceTensor>;
  specs: readonly RangeCacheSpec[];
}>;
type EmbeddingRow = Readonly<{outputRow: number; sourceRow: number}>;

export interface Mapika2bSourceOpenOptions {
  readonly cache?: AssetCache | null | undefined;
  readonly offline?: boolean | undefined;
  readonly signal?: AbortSignal | null | undefined;
}

export interface Mapika2bSource {
  readonly requiredRangeBytes: number;
  readTensor(asset: AssetId, name: string, options?: RangeCacheReadOptions): Promise<Uint8Array<ArrayBuffer>>;
  readParameter(binding: ParameterBinding, options?: RangeCacheReadOptions): Promise<Uint8Array<ArrayBuffer>>;
  readEmbeddingRows(
    ids: readonly number[] | Int32Array,
    options?: RangeCacheReadOptions
  ): Promise<Uint16Array<ArrayBuffer>>;
  prefetchRequiredRanges(options?: PrefetchOptions): Promise<void>;
  hasCommittedRanges(options?: RangeCacheReadOptions): Promise<boolean>;
  commit(options?: RangeCacheReadOptions): Promise<boolean>;
  finishLoading(): void;
  dispose(): void;
}

const invalid = (message: string) => new WebDecisionError('INVALID_ASSET', message);

function safePositive(value: number): boolean {
  return Number.isSafeInteger(value) && value > 0;
}

function rangeSpec(asset: SourceAsset, chunk: SourceRange): RangeCacheSpec {
  return {path: asset.file, totalBytes: asset.bytes, offset: chunk.offset, length: chunk.length, sha256: chunk.sha256};
}

function tensorRanges(asset: SourceAsset, tensor: SourceTensor, start: number, rangeBytes: number): RangeCacheSpec[] {
  const elements = tensor.shape.reduce((count, dimension) => count * dimension, 1);
  if (
    !tensor.name ||
    tensor.dtype !== 'BF16' ||
    !safePositive(elements) ||
    tensor.offset !== start ||
    tensor.length !== elements * 2
  )
    throw invalid('Invalid Mapika Safetensors entry: ' + tensor.name + '.');
  let cursor = start;
  const ranges = tensor.chunks.map((chunk) => {
    const expected = Math.min(rangeBytes, start + tensor.length - cursor);
    if (chunk.offset !== cursor || chunk.length !== expected || expected < 1)
      throw invalid('Mapika Safetensors ranges do not match ' + tensor.name + '.');
    cursor += chunk.length;
    return rangeSpec(asset, chunk);
  });
  if (cursor !== start + tensor.length) throw invalid('Mapika Safetensors chunks do not cover ' + tensor.name + '.');
  return ranges;
}

function planAsset(asset: SourceAsset, id: AssetId, rangeBytes: number): AssetPlan {
  if (asset.file !== filenames[id] || asset.tensors.length !== asset.tensorCount || !safePositive(asset.tensorCount))
    throw invalid('Unexpected Mapika Safetensors source.');
  const header = rangeSpec(asset, {offset: 0, length: asset.headerRangeBytes, sha256: asset.headerRangeSha256});
  const specs = [header];
  const tensors = new Map<string, SourceTensor>();
  let offset = header.length;
  for (const tensor of [...asset.tensors].sort((left, right) => left.offset - right.offset)) {
    if (tensors.has(tensor.name)) throw invalid('Duplicate Mapika Safetensors name.');
    specs.push(...tensorRanges(asset, tensor, offset, rangeBytes));
    tensors.set(tensor.name, tensor);
    offset += tensor.length;
  }
  if (offset !== asset.bytes) throw invalid('Mapika Safetensors ranges do not cover the file.');
  return {header, tensors, specs};
}

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((size, index) => size === right[index]);
}

function checkHeader(data: ArrayBuffer, plan: AssetPlan): void {
  const parsed = parseSafetensorsHeader(data, plan.header.totalBytes);
  if (parsed.tensorCount !== plan.tensors.size) throw invalid('Mapika Safetensors header count differs.');
  for (const {name, dtype, shape, offset, length} of parsed.tensors) {
    const pin = plan.tensors.get(name);
    if (!pin || pin.dtype !== dtype || pin.offset !== offset || pin.length !== length || !sameShape(pin.shape, shape))
      throw invalid('Mapika Safetensors header differs for ' + name + '.');
  }
}

function groupRows(
  ids: readonly number[] | Int32Array,
  vocabulary: number,
  rowsPerRange: number
): Map<number, EmbeddingRow[]> {
  if (!Array.isArray(ids) && !(ids instanceof Int32Array)) throw new TypeError('Mapika token IDs must be an array.');
  const groups = new Map<number, EmbeddingRow[]>();
  for (let row = 0; row < ids.length; row += 1) {
    const id = ids[row]!;
    if (!Number.isSafeInteger(id) || id < 0 || id >= vocabulary)
      throw new WebDecisionError('INVALID_INPUT', 'Mapika token ID is outside the pinned embedding table.');
    const range = Math.floor(id / rowsPerRange);
    const group = groups.get(range) ?? [];
    group.push({outputRow: row, sourceRow: id % rowsPerRange});
    groups.set(range, group);
  }
  return groups;
}

function tensorByName(plan: AssetPlan, name: string): SourceTensor {
  const tensor = plan.tensors.get(name);
  if (!tensor) throw invalid('Unknown Mapika Safetensors tensor: ' + name + '.');
  return tensor;
}

function sourceReader(
  transaction: RangeCacheTransaction,
  base: URL,
  plans: Readonly<Record<AssetId, AssetPlan>>,
  assets: SourceInventory['weights'],
  rangeBytes: number,
  cacheAvailable: boolean
): Mapika2bSource {
  const headers = new Map<AssetId, Promise<void>>();
  const required = [...plans.text.specs, ...plans.vision.specs];
  const embedding = tensorByName(plans.text, embeddingName);
  const {vocabulary, width, rowsPerRange} = embeddingTable(embedding, rangeBytes);

  async function verifyHeader(id: AssetId, options: RangeCacheReadOptions): Promise<void> {
    options.signal?.throwIfAborted();
    const prior = headers.get(id);
    if (prior) return waitForAbort(() => prior, options.signal);
    const plan = plans[id];
    const pending = transaction.read(base, plan.header).then(({bytes}) => checkHeader(bytes, plan));
    headers.set(id, pending);
    void pending.catch(() => headers.delete(id));
    await waitForAbort(() => pending, options.signal);
  }

  async function readTensor(id: AssetId, name: string, options: RangeCacheReadOptions = {}) {
    const tensor = tensorByName(plans[id], name);
    if (name === embeddingName) throw invalid('Read the Mapika embedding table by selected rows.');
    await verifyHeader(id, options);
    const output = new Uint8Array(tensor.length);
    for (const chunk of tensor.chunks) {
      const {bytes} = await transaction.read(base, rangeSpec(assets[id], chunk), options);
      output.set(new Uint8Array(bytes), chunk.offset - tensor.offset);
    }
    if ((await waitForAbort(() => sha256(output), options.signal)) !== tensor.sha256)
      throw invalid('Mapika tensor digest differs: ' + name + '.');
    return output;
  }

  async function readEmbeddingRows(ids: readonly number[] | Int32Array, options: RangeCacheReadOptions = {}) {
    options.signal?.throwIfAborted();
    const groups = groupRows(ids, vocabulary, rowsPerRange);
    const output = new Uint16Array(ids.length * width);
    if (!groups.size) return output;
    await verifyHeader('text', options);
    for (const [range, rows] of groups) {
      const chunk = embedding.chunks[range]!;
      const {bytes} = await transaction.read(base, rangeSpec(assets.text, chunk), options);
      const view = new DataView(bytes);
      for (const {outputRow, sourceRow} of rows) {
        const offset = sourceRow * width * 2;
        for (let word = 0; word < width; word += 1) {
          output[outputRow * width + word] = view.getUint16(offset + word * 2, true);
        }
      }
    }
    return output;
  }

  return Object.freeze({
    requiredRangeBytes: required.reduce((bytes, spec) => bytes + spec.length, 0),
    readTensor,
    async readParameter(binding: ParameterBinding, options?: RangeCacheReadOptions) {
      const tensor = tensorByName(plans.text, binding.name);
      const packedWords = binding.shape.reduce((count, dimension) => count * dimension, 1);
      if (!safePositive(packedWords) || packedWords * 4 !== tensor.length)
        throw invalid('Mapika packed weight shape differs: ' + binding.name + '.');
      return readTensor('text', binding.name, options);
    },
    readEmbeddingRows,
    async prefetchRequiredRanges(options: PrefetchOptions = {}) {
      if (!cacheAvailable) return;
      for (const id of assetIds) await verifyHeader(id, options);
      for (let start = 0; start < required.length; start += prefetchCount) {
        options.signal?.throwIfAborted();
        const group = required.slice(start, start + prefetchCount);
        const cached = await readAssetBatch(
          group.map((_, index) => String(index)),
          async (index, signal) => {
            const spec = group[Number(index)]!;
            return transaction.isCached(spec) || (await transaction.read(base, spec, {signal})).cachedByHash;
          },
          options.signal
        );
        if (cached.some((value) => !value)) return;
        for (const spec of group) options.onProgress?.(spec.length);
      }
    },
    hasCommittedRanges: (options?: RangeCacheReadOptions) => transaction.hasCommittedRanges(options),
    commit: (options?: RangeCacheReadOptions) => transaction.commit(options),
    finishLoading: () => transaction.detachSignal(),
    dispose() {
      headers.clear();
      transaction.dispose();
    }
  });
}

function embeddingTable(embedding: SourceTensor, rangeBytes: number) {
  const [vocabulary, width] = embedding.shape;
  if (embedding.shape.length !== 2 || !vocabulary || !width) throw invalid('Invalid Mapika embedding shape.');
  const rowsPerRange = rangeBytes / (width * 2);
  if (!safePositive(rowsPerRange)) throw invalid('Mapika embedding rows cross chunk boundaries.');
  return {vocabulary, width, rowsPerRange};
}

/** Internal dependency seam for range/cache tests. Public loading uses the immutable model URL below. */
export async function openMapika2bSourceFromInventory(
  value: SourceInventory,
  expectedSha256: string,
  base: URL,
  {cache, offline = false, signal}: Mapika2bSourceOpenOptions = {}
): Promise<Mapika2bSource> {
  signal?.throwIfAborted();
  const encoded = new TextEncoder().encode(JSON.stringify(value) + '\n');
  if ((await waitForAbort(() => sha256(encoded), signal)) !== expectedSha256)
    throw invalid('Mapika source inventory digest differs.');
  const inventory = JSON.parse(new TextDecoder().decode(encoded)) as SourceInventory;
  const {maxRangeBytes, weights} = inventory;
  if (
    inventory.format !== format ||
    inventory.repository !== MAPIKA2B_REPOSITORY ||
    inventory.revision !== MAPIKA2B_REVISION ||
    !safePositive(maxRangeBytes) ||
    maxRangeBytes > 8 * 1024 * 1024
  )
    throw invalid('Unexpected Mapika source inventory.');
  const plans = {
    text: planAsset(weights.text, 'text', maxRangeBytes),
    vision: planAsset(weights.vision, 'vision', maxRangeBytes)
  };
  const transaction = new RangeCacheTransaction({
    base,
    cache,
    signal,
    offline,
    markerId: 'mapika-2b-v12:' + expectedSha256,
    markerPath: 'mapika-2b-v12-offline-complete-v1.json'
  });
  try {
    await transaction.open();
    for (const id of assetIds) transaction.require(plans[id].specs);
    transaction.seal();
    return sourceReader(transaction, base, plans, weights, maxRangeBytes, Boolean(cache) && !offline);
  } catch (cause) {
    transaction.dispose();
    throw cause;
  }
}

/** Open the original Mapika safetensors directly, with every byte range verified. */
export function openPinnedMapika2bSource(options?: Mapika2bSourceOpenOptions): Promise<Mapika2bSource> {
  const base = new URL('https://huggingface.co/' + MAPIKA2B_REPOSITORY + '/resolve/' + MAPIKA2B_REVISION + '/');
  return openMapika2bSourceFromInventory(pinnedInventory, MAPIKA2B_SOURCE_INVENTORY_SHA256, base, options);
}
