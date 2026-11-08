/** Diagnostic candidate: combine 3-score head outputs into one GPU readback.
 * Uses pinned TVM WebGPU internals and is not a public API.
 */
import {WebDecisionError, decodeFloat16} from '../kev.js';
import {unpackDebertaScores} from './packing.mjs';
import type {DecisionResult} from '../curated/types.js';
import type {DLDevice, Instance, PackedFunc, Tensor} from '../runtime/tvm.js';
import type {DebertaChunk, ValidatedDebertaTensors} from './validate.mjs';

const noFailure = Symbol('no failure');

type PackedDebertaRequest = ValidatedDebertaTensors & {readonly optionGroups: readonly (readonly string[])[]};
type RuntimeLifecycle = Readonly<{loss: unknown; disposed: boolean}>;
type InternalTensor = Tensor & {
  readonly byteOffset: number;
  readonly dtype: string;
  readonly shape: readonly number[];
  copyFrom(value: Int32Array | Tensor): InternalTensor;
  getDataPtr(): number;
  toRawBytes(): Uint8Array;
};
type WebGPUContext = Readonly<{
  bufferTable: readonly ({readonly size: number} | undefined)[];
  getDeviceAPI(
    name: string
  ): (
    sourcePointer: number,
    sourceOffset: number,
    targetPointer: number,
    targetOffset: number,
    byteLength: number
  ) => void;
}>;
type PackedRuntime = Omit<Instance, 'empty'> & {
  readonly lib: {readonly webGPUContext?: WebGPUContext | null};
  cpu(): DLDevice;
  empty(shape: number[] | number, dtype: string, device: DLDevice): InternalTensor;
};
type PackedDevice = DLDevice & {sync(): Promise<void>};
type PackedReadbackSession = {
  checkDevice(): void;
  runtime: PackedRuntime;
  device: PackedDevice;
  gpu: Pick<GPUDevice, 'pushErrorScope' | 'popErrorScope'>;
  encoder: PackedFunc;
  head: PackedFunc;
  encoderWeights: Tensor[];
  headWeights: Tensor[];
  lifecycle: RuntimeLifecycle;
  uncaptured: string[];
};
type OutputTarget = Readonly<{
  tensor: InternalTensor;
  pointer: number;
  offset: number;
}>;
type ReadbackFlags = {readbackStarted: boolean; syncAttempted: boolean};
type DebertaUnpacker = (packed: PackedDebertaRequest, scores: number[]) => DecisionResult;

const unpackDebertaScoresForReadback = unpackDebertaScores as DebertaUnpacker;

function preserveFirstFailure(current: unknown, cause: unknown): unknown {
  return current === noFailure ? cause : current;
}

function deviceLost(session: PackedReadbackSession, message: string): void {
  if (session.lifecycle.loss || session.lifecycle.disposed)
    throw new WebDecisionError('INFERENCE_ERROR', message, {cause: new Error('The GPU device was lost.')});
}

function tensorFromValues(session: PackedReadbackSession, values: readonly number[]): InternalTensor {
  const tensor = session.runtime.empty([1, values.length], 'int32', session.device);
  tensor.copyFrom(Int32Array.from(values));
  return tensor;
}

function webGpuContext(session: PackedReadbackSession): WebGPUContext {
  const context = session.runtime.lib.webGPUContext;
  if (!context || !Array.isArray(context.bufferTable)) throw new Error('Unsupported TVM WebGPU runtime.');
  return context;
}

function createOutputTarget(session: PackedReadbackSession, context: WebGPUContext, chunkCount: number): OutputTarget {
  const tensor = session.runtime.empty([1, chunkCount * 4], 'float16', session.device);
  const pointer = tensor.getDataPtr();
  const offset = tensor.byteOffset;
  const buffer = context.bufferTable[pointer];
  if (
    !buffer ||
    !Number.isSafeInteger(offset) ||
    offset < 0 ||
    offset % 4 !== 0 ||
    buffer.size < offset + chunkCount * 8
  ) {
    throw new Error('Unexpected destination tensor storage.');
  }
  return {tensor, pointer, offset};
}

function copyHeadOutput(
  session: PackedReadbackSession,
  context: WebGPUContext,
  copyWithin: ReturnType<WebGPUContext['getDeviceAPI']>,
  target: OutputTarget,
  hidden: unknown,
  seg: readonly number[],
  chunk: DebertaChunk,
  index: number
): void {
  const output = session.head(
    hidden,
    tensorFromValues(session, seg),
    tensorFromValues(session, chunk.pair_q),
    tensorFromValues(session, chunk.pair_opt),
    ...session.headWeights
  ) as InternalTensor;
  const pointer = output.getDataPtr();
  const offset = output.byteOffset;
  const buffer = context.bufferTable[pointer];
  if (
    output.dtype !== 'float16' ||
    output.shape.length !== 2 ||
    output.shape[0] !== 1 ||
    output.shape[1] !== 3 ||
    !Number.isSafeInteger(offset) ||
    offset < 0 ||
    offset % 4 !== 0 ||
    !buffer ||
    buffer.size < offset + 8
  ) {
    throw new Error('Unexpected head tensor storage.');
  }
  copyWithin(pointer, offset, target.pointer, target.offset + index * 8, 8);
}

function copyHeadOutputs(
  session: PackedReadbackSession,
  context: WebGPUContext,
  copyWithin: ReturnType<WebGPUContext['getDeviceAPI']>,
  target: OutputTarget,
  hidden: unknown,
  seg: readonly number[],
  chunks: readonly DebertaChunk[]
): void {
  for (const [index, chunk] of chunks.entries())
    copyHeadOutput(session, context, copyWithin, target, hidden, seg, chunk, index);
}

function decodePackedScores(raw: Uint8Array, packed: PackedDebertaRequest): number[] {
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const scores: number[] = [];
  for (const [index, chunk] of packed.chunks.entries()) {
    for (let scoreIndex = 0; scoreIndex < chunk.count; scoreIndex++)
      scores.push(decodeFloat16(view.getUint16(index * 8 + scoreIndex * 2, true)));
  }
  return scores;
}

async function closePackedReadback(
  session: PackedReadbackSession,
  flags: ReadbackFlags,
  scopeStarted: boolean,
  gpuScopeStarted: boolean,
  failure: unknown
): Promise<unknown> {
  failure = await syncReadbackIfNeeded(session, flags.readbackStarted && !flags.syncAttempted, failure);
  failure = await popGpuErrorScope(session, gpuScopeStarted, failure);
  return endRuntimeScope(session, scopeStarted, failure);
}

async function syncReadbackIfNeeded(
  session: PackedReadbackSession,
  shouldSync: boolean,
  failure: unknown
): Promise<unknown> {
  if (!shouldSync) return failure;
  try {
    await session.device.sync();
    session.checkDevice();
  } catch (cause) {
    return preserveFirstFailure(failure, cause);
  }
  return failure;
}

async function popGpuErrorScope(
  session: PackedReadbackSession,
  scopeStarted: boolean,
  failure: unknown
): Promise<unknown> {
  if (!scopeStarted) return failure;
  try {
    const error = await session.gpu.popErrorScope();
    return error ? preserveFirstFailure(failure, error) : failure;
  } catch (cause) {
    return preserveFirstFailure(failure, cause);
  }
}

function endRuntimeScope(session: PackedReadbackSession, scopeStarted: boolean, failure: unknown): unknown {
  if (!scopeStarted || session.lifecycle.disposed) return failure;
  try {
    session.runtime.endScope();
  } catch (cause) {
    return preserveFirstFailure(failure, cause);
  }
  return failure;
}

export async function runWithPackedReadback(
  this: PackedReadbackSession,
  packed: PackedDebertaRequest
): Promise<DecisionResult> {
  let failure: unknown = noFailure;
  let result!: DecisionResult;
  let scopeStarted = false;
  let gpuScopeStarted = false;
  const flags: ReadbackFlags = {readbackStarted: false, syncAttempted: false};
  try {
    this.checkDevice();
    this.runtime.beginScope();
    scopeStarted = true;
    this.gpu.pushErrorScope('validation');
    gpuScopeStarted = true;
    const context = webGpuContext(this);
    const copyWithin = context.getDeviceAPI('deviceCopyWithinGPU');
    const {input_ids, attention_mask, seg} = packed.inputs;
    const hidden = this.encoder(
      tensorFromValues(this, input_ids),
      tensorFromValues(this, attention_mask),
      ...this.encoderWeights
    );
    const outputTarget = createOutputTarget(this, context, packed.chunks.length);
    copyHeadOutputs(this, context, copyWithin, outputTarget, hidden, seg, packed.chunks);
    const host = this.runtime.empty([1, packed.chunks.length * 4], 'float16', this.runtime.cpu());
    flags.readbackStarted = true;
    host.copyFrom(outputTarget.tensor);
    flags.syncAttempted = true;
    await this.device.sync();
    this.checkDevice();
    const raw = host.toRawBytes();
    if (raw.byteLength !== packed.chunks.length * 8) throw new Error('Unexpected packed output length.');
    result = unpackDebertaScoresForReadback(packed, decodePackedScores(raw, packed));
  } catch (cause) {
    failure = cause;
  } finally {
    failure = await closePackedReadback(this, flags, scopeStarted, gpuScopeStarted, failure);
  }
  if (failure !== noFailure)
    throw new WebDecisionError('INFERENCE_ERROR', 'DeBERTa inference failed.', {cause: failure});
  if (this.uncaptured.length) throw new WebDecisionError('INFERENCE_ERROR', this.uncaptured[0]!);
  // Device loss may arrive while the asynchronous error scope is closing.
  deviceLost(this, 'DeBERTa inference failed.');
  return result;
}
