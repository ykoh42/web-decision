import type {DLDevice, Instance, Tensor, VirtualMachine} from '../runtime/tvm-types.js';
import {withGemma4V2Cleanup} from './gemma4-v2-cleanup.mjs';

type OwnedTensor = Tensor & {dispose(): void};
type ScopedRuntime = Instance & {detachFromCurrentScope<T extends OwnedTensor>(value: T): T};
type Tuple = {size(): number; get(index: number): OwnedTensor};

type QueryGraphs = Readonly<{weights: string; chunk: string; rope: string; kv?: string}>;
type GraphNames = Readonly<{
  attention: string;
  unpack: string;
  tail: string;
  slice: string;
  concat: string;
  query?: QueryGraphs;
}>;
type QueryExecution = Readonly<{
  graphs: QueryGraphs;
  cache: readonly Tensor[];
  weights: readonly Tensor[];
  prefixTokens: number;
}>;
export type Gemma4V2ChunkPlan = Readonly<{
  tokens: number;
  chunkTokens: number;
}>;
export type Gemma4V2ChunkedDecoderInput = Readonly<{
  hidden: Tensor;
  perLayer: Tensor;
  cosine: Tensor;
  sine: Tensor;
  split: Tensor;
  prefixTokens?: number;
  valid: Tensor;
  scale: Tensor;
  cache: readonly Tensor[];
  attentionWeights: readonly Tensor[];
  tailWeights: readonly Tensor[];
  retainKeyValues: boolean;
  plan: Gemma4V2ChunkPlan;
}>;
export type Gemma4V2ChunkedDecoderOutput = Readonly<{
  hidden: OwnedTensor;
  key?: OwnedTensor;
  value?: OwnedTensor;
}>;

type Context = Readonly<{
  runtime: ScopedRuntime;
  device: DLDevice;
  vm: VirtualMachine;
  graphs: GraphNames;
  signal?: AbortSignal | null;
}>;

function validate(input: Gemma4V2ChunkedDecoderInput): void {
  const {tokens, chunkTokens} = input.plan;
  if (
    !Number.isSafeInteger(tokens) ||
    tokens < 1 ||
    tokens > 2147483647 ||
    !Number.isSafeInteger(chunkTokens) ||
    chunkTokens < 1 ||
    chunkTokens > tokens
  )
    throw new RangeError('Invalid Gemma 4 chunk execution plan.');
  if (
    input.hidden.dtype !== 'float32' ||
    input.hidden.shape.join(',') !== `1,${tokens},1536` ||
    input.perLayer.dtype !== 'float32' ||
    input.perLayer.shape.join(',') !== `1,${tokens},256` ||
    (input.cache.length !== 0 && input.cache.length !== 2) ||
    input.tailWeights.length !== 8 ||
    (input.cache.length > 0 && input.retainKeyValues)
  )
    throw new RangeError('Gemma 4 chunk inputs do not match the execution plan.');
}

async function scoped<T>(context: Context, work: () => T): Promise<T> {
  const {runtime, device, signal} = context;
  signal?.throwIfAborted();
  runtime.beginScope();
  const result = await withGemma4V2Cleanup(
    async () => work(),
    () =>
      withGemma4V2Cleanup(
        () => device.sync(),
        () => runtime.endScope()
      )
  );
  signal?.throwIfAborted();
  return result;
}

function detach(runtime: ScopedRuntime, owned: Set<OwnedTensor>, tensor: OwnedTensor): OwnedTensor {
  const output = runtime.detachFromCurrentScope(tensor);
  owned.add(output);
  return output;
}

function disposeOwned(owned: Set<OwnedTensor>, tensor: OwnedTensor): void {
  if (owned.delete(tensor)) tensor.dispose();
}

function disposeRemaining(owned: Set<OwnedTensor>, retained: Set<OwnedTensor>): void {
  let failure: unknown;
  for (const tensor of owned) {
    if (retained.has(tensor)) continue;
    try {
      disposeOwned(owned, tensor);
    } catch (error) {
      failure ??= error;
    }
  }
  if (failure !== undefined) throw failure;
}

function attention(context: Context, input: Gemma4V2ChunkedDecoderInput): {hidden: Tensor; tuple?: Tuple} {
  const result = context.vm.getFunction(context.graphs.attention)(
    input.hidden,
    input.cosine,
    input.sine,
    input.split,
    input.valid,
    ...input.cache,
    ...input.attentionWeights
  );
  if (input.cache.length) return {hidden: result as Tensor};
  const tuple = result as Tuple;
  if (tuple.size() !== 3) throw new Error('Unexpected Gemma 4 attention output.');
  return {hidden: tuple.get(0), tuple};
}

function decodedWeights(context: Context, input: Gemma4V2ChunkedDecoderInput): OwnedTensor[] {
  const tuple = context.vm.getFunction(context.graphs.unpack)(...input.tailWeights) as Tuple;
  if (tuple.size() !== 8) throw new Error('Unexpected Gemma 4 unpack output.');
  return [1, 2, 3, 0, 4, 5, 6, 7].map((index) => tuple.get(index));
}

function prepareQuery(context: Context, input: Gemma4V2ChunkedDecoderInput, graphs: QueryGraphs): QueryExecution {
  const prefixTokens = input.prefixTokens;
  if (
    prefixTokens === undefined ||
    !Number.isSafeInteger(prefixTokens) ||
    prefixTokens < 0 ||
    prefixTokens > input.plan.tokens
  )
    throw new RangeError('Invalid Gemma 4 prefix length for query tiling.');
  const prepared = context.vm.getFunction(graphs.weights)(...input.attentionWeights) as Tuple;
  const expected = input.cache.length ? 5 : 8;
  if (prepared.size() !== expected) throw new Error('Unexpected Gemma 4 attention weights.');
  const weights = Array.from({length: 5}, (_, index) => prepared.get(index));
  if (input.cache.length) return {graphs, cache: input.cache, weights, prefixTokens};
  if (!graphs.kv) throw new Error('Missing Gemma 4 K/V preparation graph.');
  const kv = context.vm.getFunction(graphs.kv)(
    input.hidden,
    input.cosine,
    input.sine,
    prepared.get(5),
    prepared.get(6),
    prepared.get(7),
    weights[3]
  ) as Tuple;
  if (kv.size() !== 2) throw new Error('Unexpected Gemma 4 K/V preparation output.');
  return {graphs, cache: [kv.get(0), kv.get(1)], weights, prefixTokens};
}

function queryChunk(
  context: Context,
  input: Gemma4V2ChunkedDecoderInput,
  query: QueryExecution,
  hidden: Tensor,
  offset: Tensor,
  template: Tensor,
  start: number
): Tensor {
  const rope = context.vm.getFunction(query.graphs.rope)(input.cosine, input.sine, offset, template) as Tuple;
  if (rope.size() !== 2) throw new Error('Unexpected Gemma 4 RoPE slice.');
  const metadata = context.runtime
    .empty([2], 'int32', context.device)
    .copyFrom(new Int32Array([query.prefixTokens, start]));
  return context.vm.getFunction(query.graphs.chunk)(
    hidden,
    rope.get(0),
    rope.get(1),
    input.valid,
    metadata,
    ...query.cache,
    ...query.weights
  ) as Tensor;
}

async function makeChunk(
  context: Context,
  input: Gemma4V2ChunkedDecoderInput,
  owned: Set<OwnedTensor>,
  hidden: Tensor,
  weights: readonly Tensor[],
  start: number,
  query?: QueryExecution
): Promise<OwnedTensor> {
  const length = Math.min(input.plan.chunkTokens, input.plan.tokens - start);
  return scoped(context, () => {
    const offset = context.runtime.empty([1], 'int32', context.device).copyFrom(new Int32Array([start]));
    const template = context.runtime.empty([1, length, 1], 'float32', context.device);
    const pair = context.vm.getFunction(context.graphs.slice)(hidden, input.perLayer, offset, template) as Tuple;
    if (pair.size() !== 2) throw new Error('Unexpected Gemma 4 chunk slice.');
    const hiddenChunk = pair.get(0);
    const attended = query ? queryChunk(context, input, query, hiddenChunk, offset, template, start) : hiddenChunk;
    const output = context.vm.getFunction(context.graphs.tail)(
      attended,
      pair.get(1),
      input.scale,
      ...weights
    ) as OwnedTensor;
    if (output.dtype !== 'float32' || output.shape.join(',') !== `1,${length},1536`)
      throw new Error('Unexpected Gemma 4 chunk output.');
    return detach(context.runtime, owned, output);
  });
}

async function joinChunks(context: Context, owned: Set<OwnedTensor>, chunks: OwnedTensor[]): Promise<OwnedTensor> {
  let current = chunks;
  while (current.length > 1) {
    const next: OwnedTensor[] = [];
    for (let index = 0; index < current.length; index += 2) {
      const left = current[index]!;
      const right = current[index + 1];
      if (!right) {
        next.push(left);
        continue;
      }
      const combined = await scoped(context, () => {
        const result = context.vm.getFunction(context.graphs.concat)(left, right) as OwnedTensor;
        return detach(context.runtime, owned, result);
      });
      disposeOwned(owned, left);
      disposeOwned(owned, right);
      next.push(combined);
    }
    current = next;
  }
  return current[0]!;
}

/**
 * Run a private experimental decoder without truncating the requested tokens.
 * The chunk size does not guarantee that device memory will be sufficient.
 * Returned tensors are detached and owned by the caller.
 */
export async function runGemma4V2ChunkedDecoder(
  context: Context,
  input: Gemma4V2ChunkedDecoderInput
): Promise<Gemma4V2ChunkedDecoderOutput> {
  validate(input);
  context.signal?.throwIfAborted();
  const owned = new Set<OwnedTensor>();
  const retained = new Set<OwnedTensor>();
  context.runtime.beginScope();
  try {
    const result = await withGemma4V2Cleanup(
      async () => {
        const query = context.graphs.query ? prepareQuery(context, input, context.graphs.query) : undefined;
        const attended = query ? {hidden: input.hidden} : attention(context, input);
        const weights = decodedWeights(context, input);
        const chunks: OwnedTensor[] = [];
        for (let start = 0; start < input.plan.tokens; start += input.plan.chunkTokens)
          chunks.push(await makeChunk(context, input, owned, attended.hidden, weights, start, query));
        const hidden = await joinChunks(context, owned, chunks);
        if (hidden.shape.join(',') !== `1,${input.plan.tokens},1536`)
          throw new Error('Unexpected Gemma 4 assembled output.');
        context.signal?.throwIfAborted();
        retained.add(hidden);
        if (!input.retainKeyValues) return {hidden};
        const key = detach(context.runtime, owned, query ? (query.cache[0] as OwnedTensor) : attended.tuple!.get(1));
        const value = detach(context.runtime, owned, query ? (query.cache[1] as OwnedTensor) : attended.tuple!.get(2));
        retained.add(key);
        retained.add(value);
        return {hidden, key, value};
      },
      () =>
        withGemma4V2Cleanup(
          () => context.device.sync(),
          () =>
            withGemma4V2Cleanup(
              async () => disposeRemaining(owned, retained),
              () => context.runtime.endScope()
            )
        )
    );
    context.signal?.throwIfAborted();
    return result;
  } catch (error) {
    try {
      disposeRemaining(owned, new Set());
    } catch {
      /* Preserve the original error. */
    }
    throw error;
  }
}
