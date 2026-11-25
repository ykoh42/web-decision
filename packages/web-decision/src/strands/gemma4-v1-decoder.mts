import {STRANDS_GEMMA4_MAX_LENGTH} from './gemma4-protocol.mjs';
import type {Gemma4V1OwnedRunner} from './gemma4-v1-session.mjs';
import {
  type Gemma4V1Binding,
  type createGemma4V1Source,
  perLayerEmbedding,
  wordEmbedding
} from './gemma4-v1-source.mjs';

const hiddenSize = 1536;
const pleWidth = 256;
const layerCount = 35;
const tileSize = 64;
const modelHeads = 8;
type Source = Pick<ReturnType<typeof createGemma4V1Source>, 'rows'>;

export interface Gemma4V1HostTensor {
  readonly shape: number[];
  readonly data: Float32Array;
  readonly reuseWithinLayer?: boolean;
}
interface Layer {
  readonly family: 'local_owner' | 'global_owner' | 'local_shared' | 'global_shared';
  readonly producer: number | null;
  readonly window: number | null;
  readonly project: readonly Gemma4V1Binding[];
  readonly tail: readonly Gemma4V1Binding[];
}
export interface Gemma4V1Graph {
  readonly layers: readonly Layer[];
  readonly maxLength: number;
  readonly final: readonly Gemma4V1Binding[];
  readonly ple: readonly Gemma4V1Binding[];
  readonly rope: Readonly<{sliding_attention: readonly number[]; full_attention: readonly number[]}>;
}

export interface Gemma4V1Stages {
  readonly signal?: AbortSignal | undefined;
  withWeights<T>(
    weights: Readonly<Record<string, readonly Gemma4V1Binding[]>>,
    signal: AbortSignal,
    operation: (call: Gemma4V1StageCall) => Promise<T>
  ): Promise<T>;
  releaseCachedMemory(): Promise<void> | void;
  dispose(): Promise<void> | void;
}
type Gemma4V1StageCall = (
  name: string,
  inputs: readonly Gemma4V1HostTensor[],
  releaseReusedInputs?: boolean
) => Promise<readonly Gemma4V1HostTensor[]>;

type LayerKv = {key: Float32Array; value: Float32Array; start: number};
type Prefix = Readonly<{end: number; layers: readonly (LayerKv | undefined)[]}>;

function tensor(data: Float32Array, ...shape: number[]): Gemma4V1HostTensor {
  if (data.length !== shape.reduce((a, b) => a * b, 1)) throw new Error('Unexpected Gemma 4 v1 tensor shape.');
  return {shape, data};
}

function bf16Math() {
  const value = new Float32Array(1);
  const bits = new Uint32Array(value.buffer);
  return {
    round(number: number) {
      value[0] = number;
      const word = bits[0]!;
      bits[0] = (word + 0x7fff + ((word >>> 16) & 1)) & 0xffff0000;
      return value[0]!;
    },
    decode(half: number) {
      bits[0] = half << 16;
      return value[0]!;
    }
  };
}

function scaleEmbedding(words: Uint16Array, scale: number) {
  const bf16 = bf16Math();
  const factor = bf16.round(scale);
  return Float32Array.from(words, (half) => bf16.round(bf16.decode(half) * factor));
}

export function gemma4V1CausalMask(query: number, queries: number, key: number, keys: number, window: number | null) {
  const mask = new Float32Array(queries * keys);
  for (let row = 0; row < queries; row++) {
    for (let col = 0; col < keys; col++) {
      const q = query + row;
      const k = key + col;
      if (k > q || (window !== null && k <= q - window)) mask[row * keys + col] = -3.3895313892515355e38;
    }
  }
  return tensor(mask, 1, 1, queries, keys);
}

function rotary(start: number, rows: number, frequency: readonly number[]) {
  const width = frequency.length * 2;
  const cosine = new Float32Array(rows * width);
  const sine = new Float32Array(rows * width);
  const bf16 = bf16Math();
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < width; col++) {
      const phase = Math.fround((start + row) * frequency[col % frequency.length]!);
      cosine[row * width + col] = bf16.round(Math.cos(phase));
      sine[row * width + col] = bf16.round(Math.sin(phase));
    }
  }
  return [tensor(cosine, 1, rows, width), tensor(sine, 1, rows, width)];
}

function join(first: Float32Array | undefined, second: Float32Array) {
  if (!first) return second;
  const result = new Float32Array(first.length + second.length);
  result.set(first);
  result.set(second, first.length);
  return result;
}

function appendKeyValue(previous: LayerKv | undefined, projected: readonly Gemma4V1HostTensor[], start: number) {
  return {
    key: join(previous?.key, projected[1]!.data),
    value: join(previous?.value, projected[2]!.data),
    start: previous?.start ?? start
  };
}

function compact(kv: LayerKv, width: number, window: number | null): LayerKv {
  const rows = kv.key.length / width;
  if (window === null || rows < window) return kv;
  const cut = rows - window + 1;
  return {start: kv.start + cut, key: kv.key.slice(cut * width), value: kv.value.slice(cut * width)};
}

function queryTile(query: Float32Array, total: number, from: number, rows: number, width: number) {
  const values = new Float32Array(modelHeads * rows * width);
  for (let head = 0; head < modelHeads; head++) {
    const base = (head * total + from) * width;
    values.set(query.subarray(base, base + rows * width), head * rows * width);
  }
  return tensor(values, 1, modelHeads, rows, width);
}

function layerEmbedding(embedding: Float32Array, layer: number, from: number, count: number) {
  const data = new Float32Array(count * pleWidth);
  for (let row = 0; row < count; row++) {
    const start = ((from + row) * layerCount + layer) * pleWidth;
    data.set(embedding.subarray(start, start + pleWidth), row * pleWidth);
  }
  return tensor(data, 1, count, pleWidth);
}

async function layerForward(args: {
  call: Gemma4V1StageCall;
  layer: Layer;
  index: number;
  hidden: Float32Array;
  ple: Float32Array;
  start: number;
  rows: number;
  frequency: readonly number[];
  previous?: LayerKv | undefined;
  shared?: LayerKv | undefined;
}) {
  const {call, layer, rows, start, hidden, ple, frequency} = args;
  const width = frequency.length * 2;
  const locality = layer.window === null ? 'global' : 'local';
  const projected = await call(layer.family + '_project_main', [
    tensor(hidden, 1, rows, hiddenSize),
    ...rotary(start, rows, frequency)
  ]);
  const active = args.shared ?? appendKeyValue(args.previous, projected, start);
  const keys = active.key.length / width;
  if (active.start + keys !== start + rows)
    throw new Error('Gemma 4 v1 K/V positions differ from the visible sequence.');
  const kv = [active.key, active.value].map((data) => ({
    ...tensor(data, 1, 1, keys, width),
    reuseWithinLayer: rows > tileSize
  }));
  const output = new Float32Array(hidden.length);
  for (let from = 0; from < rows; from += tileSize) {
    const tile = Math.min(tileSize, rows - from);
    const query = queryTile(projected[0]!.data, rows, from, tile, width);
    const mask = gemma4V1CausalMask(start + from, tile, active.start, keys, layer.window);
    const [attention] = await call(locality + '_attention_main', [query, ...kv, mask], from + tile === rows);
    const hiddenTile = tensor(hidden.subarray(from * hiddenSize, (from + tile) * hiddenSize), 1, tile, hiddenSize);
    const [result] = await call(layer.family + '_tail_main', [
      hiddenTile,
      layerEmbedding(ple, args.index, from, tile),
      attention!
    ]);
    output.set(result!.data, from * hiddenSize);
  }
  return {output, active, cached: layer.producer === null ? compact(active, width, layer.window) : undefined};
}

async function projectPerLayer(
  graph: Gemma4V1Graph,
  stages: Gemma4V1Stages,
  hidden: Float32Array,
  identity: Float32Array,
  rows: number,
  signal: AbortSignal
) {
  const width = layerCount * pleWidth;
  const ple = new Float32Array(rows * width);
  return stages.withWeights({ple_main: graph.ple}, signal, async (call) => {
    for (let from = 0; from < rows; from += tileSize) {
      const count = Math.min(tileSize, rows - from);
      const [result] = await call('ple_main', [
        tensor(hidden.subarray(from * hiddenSize, (from + count) * hiddenSize), 1, count, hiddenSize),
        tensor(identity.subarray(from * width, (from + count) * width), 1, count, width)
      ]);
      ple.set(result!.data, from * width);
    }
    return ple;
  });
}

export function createGemma4V1Decoder(
  graph: Gemma4V1Graph,
  stages: Gemma4V1Stages,
  source: Source,
  checkDevice: (tokens: number) => void = () => {}
): Gemma4V1OwnedRunner {
  if (graph.layers.length !== layerCount) throw new Error('Expected the 35 pinned Gemma 4 v1 layers.');
  const sharedIndices = new Set(graph.layers.flatMap(({producer}) => (producer === null ? [] : [producer])));

  async function decode(tokens: readonly number[], signal: AbortSignal, previous?: Prefix) {
    const rows = tokens.length;
    const start = previous?.end ?? 0;
    if (!rows || start + rows > graph.maxLength)
      throw new Error('Gemma 4 v1 graph input exceeds its position capacity.');
    let hidden = scaleEmbedding(await source.rows(wordEmbedding, tokens, signal), Math.sqrt(hiddenSize));
    const identity = scaleEmbedding(await source.rows(perLayerEmbedding, tokens, signal), Math.sqrt(pleWidth));
    const ple = await projectPerLayer(graph, stages, hidden, identity, rows, signal);
    const shared = new Map<number, LayerKv>();
    const layers: (LayerKv | undefined)[] = [];
    for (const [index, layer] of graph.layers.entries()) {
      const weights = {[layer.family + '_project_main']: layer.project, [layer.family + '_tail_main']: layer.tail};
      const suffix = await stages.withWeights(weights, signal, (call) =>
        layerForward({
          call,
          layer,
          index,
          hidden,
          ple,
          rows,
          start,
          previous: previous?.layers[index],
          shared: layer.producer === null ? undefined : shared.get(layer.producer),
          frequency: layer.window === null ? graph.rope.full_attention : graph.rope.sliding_attention
        })
      );
      hidden = suffix.output;
      layers.push(suffix.cached);
      if (sharedIndices.has(index)) shared.set(index, suffix.active);
    }
    return {hidden, prefix: {end: start + rows, layers}};
  }

  async function decodeChunks(
    tokens: readonly number[],
    indices: readonly number[],
    signal: AbortSignal,
    prefix?: Prefix
  ) {
    const selected = new Float32Array(indices.length * hiddenSize);
    if (indices.some((index) => !Number.isInteger(index) || index < 0 || index >= tokens.length))
      throw new Error('Invalid Gemma 4 v1 pointer index.');
    // Keep projections and per-layer embedding allocations bounded while K/V
    // carries absolute positions across causal chunks.
    for (let start = 0; start < tokens.length; start += STRANDS_GEMMA4_MAX_LENGTH) {
      const chunk = tokens.slice(start, start + STRANDS_GEMMA4_MAX_LENGTH);
      const {hidden, prefix: next} = await decode(chunk, signal, prefix);
      prefix = next;
      for (const [slot, index] of indices.entries()) {
        const row = index - start;
        if (row >= 0 && row < chunk.length)
          selected.set(hidden.subarray(row * hiddenSize, (row + 1) * hiddenSize), slot * hiddenSize);
      }
    }
    return {selected, prefix};
  }

  return {
    async run(state, questions, caller, usePrefixCache = true) {
      const signal = stages.signal ? AbortSignal.any([caller, stages.signal]) : caller;
      signal.throwIfAborted();
      const largest = state.length + questions.reduce((longest, {ids}) => Math.max(longest, ids.length), 0);
      checkDevice(largest);
      const prefix =
        usePrefixCache && questions.length > 1 ? (await decodeChunks(state, [], signal)).prefix : undefined;
      const answers = [];
      for (const question of questions) {
        const offset = prefix ? 0 : state.length;
        const indices = [question.answerIndex, ...question.optionIndices].map((position) => offset + position);
        const {selected: pointers} = await decodeChunks(
          prefix ? question.ids : [...state, ...question.ids],
          indices,
          signal,
          prefix
        );
        answers.push(
          await stages.withWeights({final_main: graph.final}, signal, async (call) => {
            const [normalized] = await call('final_main', [
              tensor(pointers, 1, question.optionIndices.length + 1, hiddenSize)
            ]);
            const [logits] = await call('pointer_main', [
              tensor(normalized!.data.subarray(0, hiddenSize), 1, hiddenSize),
              tensor(normalized!.data.subarray(hiddenSize), 1, question.optionIndices.length, hiddenSize)
            ]);
            return logits!.data;
          })
        );
      }
      return answers;
    },
    releaseCachedMemory: () => stages.releaseCachedMemory(),
    dispose: () => stages.dispose()
  };
}
