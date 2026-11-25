import {waitForAbort, type AssetCache} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {RangeCacheTransaction, type RangeCacheSpec} from './range-cache.mjs';
import {readVerifiedRange} from './range-reader.mjs';

type SourceId = 'base' | 'adapter';
type Chunk = readonly [offset: number, length: number, sha256: string];
interface SourceTensor {
  readonly offset: number;
  readonly length: number;
  readonly shape: readonly number[];
  readonly dtype: 'BF16' | 'F32';
}
interface CheckpointInventory {
  readonly repository: string;
  readonly revision: string;
  readonly file: string;
  readonly bytes: number;
  readonly sha256: string;
  readonly headerBytes: number;
  readonly chunks: readonly Chunk[];
  readonly tensors: Readonly<Record<string, SourceTensor>>;
}

export interface Gemma4V1Inventory {
  readonly format: 'web-decision-gemma4-e2b-v1-original-ranges-v1';
  readonly base: CheckpointInventory;
  readonly adapter: CheckpointInventory;
}

export interface Gemma4V1Binding {
  readonly source: SourceId;
  readonly tensor: string;
  readonly dtype: 'uint32' | 'float32';
  readonly shape: readonly number[];
}

export const wordEmbedding = 'model.language_model.embed_tokens.weight';
export const perLayerEmbedding = 'model.language_model.embed_tokens_per_layer.weight';

const invalid = (message: string) => new WebDecisionError('INVALID_ASSET', message);
const rangeReader = readVerifiedRange;

function indexOf(chunks: readonly Chunk[], position: number): number {
  let lo = 0;
  let hi = chunks.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (chunks[mid]![0] <= position) lo = mid + 1;
    else hi = mid;
  }
  const i = lo - 1;
  const chunk = chunks[i];
  if (!chunk || position >= chunk[0] + chunk[1]) throw invalid('Source range does not cover the tensor.');
  return i;
}

function listRange(source: CheckpointInventory, position: number, size: number) {
  const end = position + size;
  const items: RangeCacheSpec[] = [];
  for (let i = indexOf(source.chunks, position); position < end; i++) {
    const [offset, length, sha256] = source.chunks[i]!;
    if (offset > position) throw invalid('Source range coverage has a gap.');
    items.push({path: source.file, totalBytes: source.bytes, offset, length, sha256});
    position = offset + length;
  }
  return items;
}

function findTensor(inventory: CheckpointInventory, key: string) {
  const tensor = inventory.tensors[key];
  if (!tensor) throw invalid('Tensor missing from pinned Gemma 4 v1 source.');
  return tensor;
}

function weightPlan(inv: CheckpointInventory, names: readonly string[]) {
  const ranges = new Map<number, RangeCacheSpec>();
  function add(start: number, length: number) {
    for (const range of listRange(inv, start, length)) ranges.set(range.offset, range);
  }
  add(0, inv.headerBytes);
  for (const key of names) {
    const tensor = findTensor(inv, key);
    add(tensor.offset, tensor.length);
  }
  return [...ranges.values()].sort((a, b) => a.offset - b.offset);
}

async function readSpan(
  inv: CheckpointInventory,
  offset: number,
  length: number,
  read: (range: RangeCacheSpec) => Promise<ArrayBuffer>
) {
  const copied = new Uint8Array(length);
  for (const range of listRange(inv, offset, length)) {
    const bytes = await read(range);
    const start = Math.max(range.offset, offset);
    const end = Math.min(range.offset + range.length, offset + length);
    copied.set(new Uint8Array(bytes, start - range.offset, end - start), start - offset);
  }
  return copied;
}

/** Build host inputs from verified bytes of the frozen Hugging Face originals. */
export function createGemma4V1Source(
  inventory: Gemma4V1Inventory,
  bindings: readonly Gemma4V1Binding[],
  options: {cache?: AssetCache | null; offline?: boolean} = {},
  readRange: typeof readVerifiedRange = rangeReader
) {
  if (inventory.format !== 'web-decision-gemma4-e2b-v1-original-ranges-v1')
    throw invalid('Wrong Gemma 4 v1 source inventory.');
  const ids = ['base', 'adapter'] as const;
  const sources = Object.fromEntries(
    ids.map((id) => [
      id,
      new URL(`https://huggingface.co/${inventory[id].repository}/resolve/${inventory[id].revision}/`)
    ])
  ) as Record<SourceId, URL>;
  const plans = ids.map((id) =>
    weightPlan(inventory[id], [
      ...new Set(bindings.filter((binding) => binding.source === id).map((binding) => binding.tensor)),
      ...(id === 'base' ? [wordEmbedding, perLayerEmbedding] : [])
    ])
  );
  async function transaction(id: SourceId, signal?: AbortSignal) {
    const step = new RangeCacheTransaction({
      base: sources[id],
      cache: options.cache,
      signal,
      markerId: `strands-gemma4-v1-${id}:${inventory[id].sha256}`,
      markerPath: 'web-decision-gemma4-v1-offline.json'
    });
    try {
      await step.open();
      step.require(plans[ids.indexOf(id)]!);
      step.seal();
      return step;
    } catch (error) {
      step.dispose();
      throw error;
    }
  }
  const read = (id: SourceId, range: RangeCacheSpec, signal?: AbortSignal) =>
    readRange(sources[id], range, {...options, signal}).then(({bytes}) => bytes);
  let preparation = Promise.resolve();

  async function prepare(signal?: AbortSignal): Promise<void> {
    signal?.throwIfAborted();
    if (!options.cache)
      throw new WebDecisionError('ASSET_LOAD_FAILED', 'Gemma 4 v1 requires a cache for offline preparation.');
    for (const [index, id] of ids.entries()) {
      const step = await transaction(id, signal);
      try {
        for (const range of plans[index]!) {
          const result = options.offline
            ? await readRange(sources[id], range, {...options, signal})
            : await step.read(sources[id], range, {signal});
          if (!result.cachedByHash)
            throw new WebDecisionError('ASSET_LOAD_FAILED', 'Gemma 4 v1 could not cache a verified weight range.');
        }
        if (options.offline ? !(await step.hasCommittedRanges({signal})) : !(await step.commit({signal})))
          throw new WebDecisionError('ASSET_LOAD_FAILED', 'Gemma 4 v1 offline weights are incomplete.');
      } finally {
        step.dispose();
      }
    }
  }

  return {
    async tensor(binding: Gemma4V1Binding, signal?: AbortSignal) {
      const inv = inventory[binding.source];
      const tensor = findTensor(inv, binding.tensor);
      const size = binding.shape.reduce((product, dim) => product * dim, 1) * 4;
      const expected = tensor.dtype === 'BF16' ? 'uint32' : 'float32';
      if (binding.dtype !== expected || size !== Math.ceil(tensor.length / 4) * 4)
        throw invalid('Gemma 4 v1 graph binding differs from the source.');
      const raw = await readSpan(inv, tensor.offset, tensor.length, (range) => read(binding.source, range, signal));
      const bytes = new Uint8Array(size);
      bytes.set(raw);
      return expected === 'uint32' ? new Uint32Array(bytes.buffer) : new Float32Array(bytes.buffer);
    },
    async rows(name: string, tokens: readonly number[], signal?: AbortSignal) {
      const {base} = inventory;
      const tensor = findTensor(base, name);
      const [vocabulary, width] = tensor.shape;
      if (tensor.dtype !== 'BF16' || tensor.shape.length !== 2 || !width || !vocabulary)
        throw invalid('Gemma 4 v1 embedding has an invalid shape.');
      const out = new Uint16Array(tokens.length * width);
      const byChunk = new Map<number, number[]>();
      for (const [row, token] of tokens.entries()) {
        if (!Number.isSafeInteger(token) || token < 0 || token >= vocabulary)
          throw invalid('Token is outside the vocabulary.');
        const at = tensor.offset + token * width * 2;
        const [range] = listRange(base, at, width * 2);
        if (!range || at + width * 2 > range.offset + range.length) throw invalid('Embedding row spans source ranges.');
        if (!byChunk.has(range.offset)) byChunk.set(range.offset, []);
        byChunk.get(range.offset)!.push(row);
      }
      for (const [offset, rows] of byChunk) {
        const [range] = listRange(base, offset, 1);
        const buffer = await read('base', range!, signal);
        for (const row of rows) {
          const position = tensor.offset + tokens[row]! * width * 2 - offset;
          const view = new Uint16Array(buffer, position, width);
          out.set(view, row * width);
        }
      }
      return out;
    },
    async isOfflineReady(signal?: AbortSignal) {
      if (!options.cache) return false;
      for (const id of ids) {
        const step = await transaction(id, signal);
        try {
          if (!(await step.hasCommittedRanges({signal}))) return false;
        } finally {
          step.dispose();
        }
      }
      return true;
    },
    async prepareOffline(signal?: AbortSignal) {
      signal?.throwIfAborted();
      const task = preparation.then(() => prepare(signal));
      preparation = task.catch(() => {});
      return waitForAbort(() => task, signal);
    }
  };
}
