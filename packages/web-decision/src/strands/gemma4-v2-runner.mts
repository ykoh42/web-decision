import type {DLDevice, Instance, Tensor, VirtualMachine} from '../runtime/tvm-types.js';
import {loadPinnedGemma4V2Bindings} from './gemma4-v2-bindings.mjs';
import {withGemma4V2Cleanup} from './gemma4-v2-cleanup.mjs';
import {assertGemma4V2CompiledContext} from './gemma4-v2-context.mjs';
import type {Gemma4V2RunnerInput, Gemma4V2OwnedRunner} from './gemma4-v2-session.mjs';
import type {openPinnedGemma4V2Source} from './gemma4-v2-source.mjs';
import {runSplitAdapter, transposedSplitGraphs} from './gemma4-v2-split-adapter.mjs';

const LAYER_EMBEDDING_BUDGET = 32 * 1024 * 1024;

type DisposableTensor = Tensor & {dispose(): void};
type ScopedRuntime = Instance & {detachFromCurrentScope<T extends DisposableTensor>(value: T): T};
type Source = Awaited<ReturnType<typeof openPinnedGemma4V2Source>>;
type Tuple = {size(): number; get(index: number): DisposableTensor};
export type Gemma4V2SplitExecution = Readonly<{
  graphs: ReturnType<typeof transposedSplitGraphs>;
  chunkTokens: number;
}>;

export interface Gemma4V2RunnerDependencies {
  readonly runtime: ScopedRuntime;
  readonly device: DLDevice;
  readonly vm: VirtualMachine;
  readonly source: Source;
  readonly readout: Uint8Array;
  /** The pinned graph returns normalized BF16 features and exports a host readout kernel. */
  readonly hostReadout?: boolean;
  readonly localFrequency: Float32Array;
  readonly globalFrequency: Float32Array;
  /** Internal only: enabled exclusively with a matching verified split graph. */
  readonly splitExecution?: Gemma4V2SplitExecution;
  /** Inserts prepared vision embeddings into the word embedding on the same runtime and device. */
  readonly insertImages: ((
    word: Tensor,
    input: Gemma4V2RunnerInput,
    signal?: AbortSignal | null
  ) => Promise<Tensor>) & {
    releaseCachedMemory?: () => void;
  };
  readonly dispose: () => void | Promise<void>;
}

function words(bytes: Uint8Array): Uint32Array {
  if (bytes.byteLength % 4 !== 0) throw new Error('Invalid packed Gemma 4 tensor.');
  if (bytes.byteOffset % 4 === 0) return new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  return new Uint32Array(bytes.slice().buffer);
}

async function assertSplitGraphFunctions(deps: Gemma4V2RunnerDependencies, split: Gemma4V2SplitExecution) {
  deps.runtime.beginScope();
  await withGemma4V2Cleanup(
    async () => {
      for (const family of [0, 4, 15, 19] as const) {
        const graph = split.graphs[family];
        if (!graph) throw new Error('Missing Gemma 4 split graph configuration.');
        for (const name of [graph.attention, graph.unpack, graph.tail, graph.slice, graph.concat]) {
          if (typeof name !== 'string' || !name || typeof deps.vm.getFunction(name) !== 'function')
            throw new Error('The Gemma 4 graph does not contain the required split functions.');
        }
      }
    },
    () => deps.runtime.endScope()
  );
}

function compactIds(input: Gemma4V2RunnerInput) {
  const ids = [...input.stateIds, ...input.questionIds];
  if (input.images.length !== input.imageSpans.length) throw new Error('Image spans do not match images.');
  let previousEnd = 0;
  for (const span of input.imageSpans) {
    if (
      !Number.isInteger(span.start) ||
      !Number.isInteger(span.end) ||
      span.start < previousEnd ||
      span.end <= span.start ||
      span.end > input.stateIds.length
    )
      throw new Error('Invalid image token span.');
    for (let index = span.start; index < span.end; index++) {
      if (ids[index] !== 258880) throw new Error('Image token span does not contain image placeholders.');
      ids[index] = 0;
    }
    previousEnd = span.end;
  }
  if (ids.length === 0 || ids.some((id) => !Number.isInteger(id) || id < 0)) throw new Error('Invalid token IDs.');
  const unique: number[] = [];
  const indices = new Map<number, number>();
  const compact = Int32Array.from(ids, (id) => {
    let index = indices.get(id);
    if (index === undefined) {
      index = unique.length;
      indices.set(id, index);
      unique.push(id);
    }
    return index;
  });
  return {unique, compact};
}

type Bindings = Awaited<ReturnType<typeof loadPinnedGemma4V2Bindings>>;
type TensorFactory = (shape: number[], dtype: string, data: Uint32Array | Int32Array | Float32Array) => Tensor;
interface DecoderState {
  hidden?: DisposableTensor;
  localKey?: Tensor;
  localValue?: Tensor;
  fullKey?: Tensor;
  fullValue?: Tensor;
  readonly saved: DisposableTensor[];
}
interface DecoderInputs {
  readonly deps: Gemma4V2RunnerDependencies;
  readonly bindings: Bindings;
  readonly unique: number[];
  readonly layerEmbeddings: readonly Uint32Array[] | undefined;
  readonly initial: Tensor;
  readonly ids: Tensor;
  readonly norm: Tensor;
  readonly split: Tensor;
  readonly valid: Tensor;
  readonly localCos: Tensor;
  readonly localSin: Tensor;
  readonly fullCos: Tensor;
  readonly fullSin: Tensor;
  readonly tensor: TensorFactory;
}

function retainOutput(runtime: ScopedRuntime, state: DecoderState, index: number, result: Tuple | DisposableTensor) {
  if (index >= 15) return result as DisposableTensor;
  const tuple = result as Tuple;
  if (tuple.size() !== 3) throw new Error('Unexpected Gemma 4 decoder output.');
  const next = tuple.get(0);
  if (index === 13 || index === 14) {
    const key = runtime.detachFromCurrentScope(tuple.get(1));
    state.saved.push(key);
    const value = runtime.detachFromCurrentScope(tuple.get(2));
    state.saved.push(value);
    if (index === 13) {
      state.localKey = key;
      state.localValue = value;
    } else {
      state.fullKey = key;
      state.fullValue = value;
    }
  }
  return next;
}

async function runSplitLayer(
  context: DecoderInputs,
  state: DecoderState,
  index: number,
  perLayer: Tensor,
  scale: Tensor,
  weights: readonly {name: string; value: Tensor}[],
  signal?: AbortSignal | null
): Promise<DisposableTensor> {
  const {runtime, device, vm, splitExecution} = context.deps;
  if (!splitExecution) throw new Error('Split graph execution is unavailable.');
  const global = index % 5 === 4;
  const key = global ? state.fullKey : state.localKey;
  const value = global ? state.fullValue : state.localValue;
  if (index >= 15 && (!key || !value)) throw new Error('Missing Gemma 4 shared KV.');
  const tokens = context.ids.shape[0]!;
  const result = await runSplitAdapter(
    {runtime, device, vm, graphs: splitExecution.graphs, ...(signal === undefined ? {} : {signal})},
    {
      index,
      hidden: state.hidden ?? context.initial,
      perLayer,
      cosine: global ? context.fullCos : context.localCos,
      sine: global ? context.fullSin : context.localSin,
      split: context.split,
      valid: context.valid,
      scale,
      weights,
      ...(index >= 15 ? {sharedCache: [key!, value!] as const} : {}),
      plan: {tokens, chunkTokens: Math.min(tokens, splitExecution.chunkTokens)}
    }
  );
  if (index === 13 || index === 14) {
    if (!result.key || !result.value) {
      result.hidden.dispose();
      result.key?.dispose();
      result.value?.dispose();
      throw new Error('Missing Gemma 4 attention KV output.');
    }
    state.saved.push(result.key, result.value);
    if (global) {
      state.fullKey = result.key;
      state.fullValue = result.value;
    } else {
      state.localKey = result.key;
      state.localValue = result.value;
    }
  }
  return result.hidden;
}

async function runDecoderLayer(
  context: DecoderInputs,
  state: DecoderState,
  index: number,
  signal?: AbortSignal | null
) {
  signal?.throwIfAborted();
  const {
    deps: {runtime, device, vm, source},
    bindings,
    tensor
  } = context;
  runtime.beginScope();
  return withGemma4V2Cleanup(
    async () => {
      const layer = bindings.layer(index);
      const projection = tensor(
        [196608],
        'uint32',
        words(
          await source.readTensorRange(
            'model.language_model.per_layer_model_projection.weight',
            index * 256 * 1536 * 2,
            256 * 1536 * 2,
            signal
          )
        )
      );
      signal?.throwIfAborted();
      const rows = tensor(
        [context.unique.length, 128],
        'uint32',
        context.layerEmbeddings?.[index] ?? (await source.readEmbeddingRows(context.unique, index, signal))
      );
      const perLayer = vm.getFunction('ple_packed_main')(
        context.initial,
        context.ids,
        rows,
        projection,
        context.norm
      ) as Tensor;
      const scalarBytes = await source.readTensor(layer.scalar, signal);
      if (scalarBytes.length !== 2) throw new Error('Invalid Gemma 4 layer scalar.');
      const scalarBits = new Uint32Array([((scalarBytes[0]! | (scalarBytes[1]! << 8)) << 16) >>> 0]);
      const scale = tensor([1], 'float32', new Float32Array(scalarBits.buffer));
      const weights = [];
      for (const parameter of layer.parameters) {
        const bytes = await source.readTensor(parameter.name, signal);
        weights.push({name: parameter.name, value: tensor([parameter.packedWords], 'uint32', words(bytes))});
      }
      const global = index % 5 === 4;
      const args: unknown[] = [
        state.hidden ?? context.initial,
        perLayer,
        global ? context.fullCos : context.localCos,
        global ? context.fullSin : context.localSin,
        context.split,
        context.valid,
        scale
      ];
      if (index >= 15) args.push(global ? state.fullKey : state.localKey, global ? state.fullValue : state.localValue);
      args.push(...weights.map((weight) => weight.value));
      signal?.throwIfAborted();
      const next = context.deps.splitExecution
        ? await runSplitLayer(context, state, index, perLayer, scale, weights, signal)
        : retainOutput(
            runtime,
            state,
            index,
            vm.getFunction(`layer${layer.family}_packed_main`)(...args) as Tuple | DisposableTensor
          );
      const previous = state.hidden;
      state.hidden = context.deps.splitExecution ? next : runtime.detachFromCurrentScope(next);
      await withGemma4V2Cleanup(
        () => device.sync(),
        () => previous?.dispose()
      );
    },
    () => runtime.endScope()
  );
}

function cleanupInference(runtime: ScopedRuntime, state: DecoderState): void {
  let firstError: unknown;
  let failed = false;
  for (const tensor of [state.hidden, ...state.saved]) {
    try {
      tensor?.dispose();
    } catch (error) {
      if (!failed) {
        firstError = error;
        failed = true;
      }
    }
  }
  try {
    runtime.endScope();
  } catch (error) {
    if (!failed) {
      firstError = error;
      failed = true;
    }
  }
  if (failed) throw firstError;
}

async function readoutLogits(
  deps: Gemma4V2RunnerDependencies,
  tensor: TensorFactory,
  hidden: Tensor,
  finalNorm: Tensor,
  signal?: AbortSignal | null
) {
  const {runtime, device, vm} = deps;
  const readout = vm.getFunction('readout_packed_main');
  const packed = words(deps.readout);
  const width = deps.hostReadout ? 1536 : 255;
  const result = (
    deps.hostReadout ? readout(hidden, finalNorm) : readout(hidden, finalNorm, tensor([195840], 'uint32', packed))
  ) as Tensor;
  if (result.dtype !== 'float32' || result.shape.length !== 2 || result.shape[0] !== 1 || result.shape[1] !== width)
    throw new Error('Unexpected Gemma 4 readout shape or dtype.');
  const cpu = runtime.cpu();
  const host = runtime.empty([1, width], 'float32', cpu).copyFrom(result);
  await device.sync();
  signal?.throwIfAborted();
  if (!deps.hostReadout) return (host.toArray() as Float32Array).slice();

  const weights = runtime.empty([195840], 'uint32', cpu).copyFrom(packed);
  const logits = runtime.empty([1, 255], 'float32', cpu);
  runtime.systemLib().getFunction('readout_accumulate')(host, weights, logits);
  return (logits.toArray() as Float32Array).slice();
}

async function runInference(
  deps: Gemma4V2RunnerDependencies,
  bindings: Bindings,
  input: Gemma4V2RunnerInput,
  signal?: AbortSignal | null
) {
  const {runtime, device, vm, source} = deps;
  const state: DecoderState = {saved: []};
  runtime.beginScope();
  return withGemma4V2Cleanup(
    async () => {
      const {unique, compact} = compactIds(input);
      const length = compact.length;
      const tensor: TensorFactory = (shape, dtype, data) => runtime.empty(shape, dtype, device).copyFrom(data);
      const ids = tensor([length], 'int32', compact);
      const wordRows = await source.readEmbeddingRows(unique, undefined, signal);
      const word = tensor([unique.length, 768], 'uint32', wordRows);
      const wordHidden = vm.getFunction('word_packed_main')(word, ids) as Tensor;
      const initial = await deps.insertImages(wordHidden, input, signal);
      signal?.throwIfAborted();
      const positions = tensor(
        [length],
        'int32',
        Int32Array.from({length}, (_, i) => i)
      );
      const localFreq = tensor([128], 'float32', deps.localFrequency);
      const globalFreq = tensor([256], 'float32', deps.globalFrequency);
      const localCos = vm.getFunction('rope_local_cos')(positions, localFreq) as Tensor;
      const localSin = vm.getFunction('rope_local_sin')(positions, localFreq) as Tensor;
      const fullCos = vm.getFunction('rope_global_cos')(positions, globalFreq) as Tensor;
      const fullSin = vm.getFunction('rope_global_sin')(positions, globalFreq) as Tensor;
      const split = tensor([1], 'int32', new Int32Array([input.stateIds.length]));
      const valid = tensor([1, length], 'int32', new Int32Array(length).fill(1));
      const norm = tensor(
        [128],
        'uint32',
        words(await source.readTensor('model.language_model.per_layer_projection_norm.weight', signal))
      );
      const context = {
        deps,
        bindings,
        unique,
        layerEmbeddings: await source.readAllLayerEmbeddingRows?.(unique, LAYER_EMBEDDING_BUDGET, signal),
        initial,
        ids,
        norm,
        split,
        valid,
        localCos,
        localSin,
        fullCos,
        fullSin,
        tensor
      };
      for (let index = 0; index < 35; index++) await runDecoderLayer(context, state, index, signal);
      signal?.throwIfAborted();
      const finalNorm = tensor(
        [768],
        'uint32',
        words(await source.readTensor('model.language_model.norm.weight', signal))
      );
      if (!state.hidden) throw new Error('The Gemma 4 decoder returned no hidden state.');
      return readoutLogits(deps, tensor, state.hidden, finalNorm, signal);
    },
    () => cleanupInference(runtime, state)
  );
}

/** Run all decoder layers in one TVM runtime. The caller owns and disposes the source. */
export async function createGemma4V2Runner(deps: Gemma4V2RunnerDependencies): Promise<Gemma4V2OwnedRunner> {
  const bindings = await loadPinnedGemma4V2Bindings();
  if (
    deps.localFrequency.length !== 128 ||
    deps.globalFrequency.length !== 256 ||
    deps.readout.byteLength !== 255 * 1536 * 2
  )
    throw new Error('Invalid Gemma 4 graph constants.');
  if (deps.splitExecution) {
    if (!Number.isSafeInteger(deps.splitExecution.chunkTokens) || deps.splitExecution.chunkTokens < 1)
      throw new Error('Invalid Gemma 4 chunk size.');
    await assertSplitGraphFunctions(deps, deps.splitExecution);
  }
  let closed = false;
  let busy = false;
  return {
    async run(input, signal) {
      if (closed) throw new Error('Gemma 4 runner is disposed.');
      if (busy) throw new Error('Gemma 4 runner is already in use.');
      signal?.throwIfAborted();
      assertGemma4V2CompiledContext(input.stateIds.length + input.questionIds.length);
      busy = true;
      try {
        return await runInference(deps, bindings, input, signal);
      } finally {
        busy = false;
      }
    },
    releaseCachedMemory() {
      if (closed) throw new Error('Gemma 4 runner is disposed.');
      if (busy) throw new Error('Cannot release memory while the Gemma 4 runner is in use.');
      deps.source.releaseCachedMemory();
      deps.insertImages.releaseCachedMemory?.();
    },
    async dispose() {
      if (closed) return;
      if (busy) throw new Error('Cannot dispose a running Gemma 4 runner.');
      closed = true;
      await deps.dispose();
    }
  };
}
