import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
/** Experimental, preloaded DeBERTa session. Not yet part of the public package. */
import {WebDecisionError, decodeFloat16} from '../kev.js';
import {DEBERTA_MAX_OPTIONS, packDebertaRequest, unpackDebertaScores} from './packing.mjs';
import {finishDebertaSystemOne, prepareDebertaSystemOne} from './typed.mjs';
import {waitForAbort} from '../assets.js';
import type {DecisionRequest, DecisionResult} from '../curated/types.js';
import type {DebertaSystemOneQuestion, DebertaSystemOneResult} from '../deberta-single-preview.js';
import type {KevTokenizer} from '../kev.js';
import type {DLDevice, Instance, PackedFunc, Tensor} from '../runtime/tvm.js';
import type {DebertaChunk, ValidatedDebertaTensors} from './validate.mjs';

const noFailure = Symbol('no failure');

type DecisionOptions = Readonly<{signal?: AbortSignal | undefined}>;
type RuntimeLifecycle = Readonly<{
  loss: unknown;
  disposed: boolean;
  lossSignal?: AbortSignal | undefined;
}>;
type DebertaTensor = Tensor & {
  copyFrom(value: Int32Array | Tensor): DebertaTensor;
  toRawBytes(): Uint8Array;
};
type DebertaRuntime = Omit<Instance, 'empty'> & {
  cpu(): DLDevice;
  empty(shape: number[] | number, dtype: string, device: DLDevice): DebertaTensor;
  getGlobalFunc(name: string): () => unknown;
};
type DebertaDevice = DLDevice & {sync(): Promise<void>};
type PackedDebertaRequest = ValidatedDebertaTensors & {readonly optionGroups: readonly (readonly string[])[]};
type DebertaQuestion = Readonly<{name: string; type: 'choice' | 'score' | 'noul'; options: readonly string[]}>;
type PreparedSystemOne = Readonly<{request: unknown; metadata: readonly DebertaQuestion[]}>;
type NamedQuestions = Readonly<Record<string, DebertaSystemOneQuestion>>;
type ValidateRequestFields = (
  request: unknown,
  maxQuestions?: number,
  maxInputCharacters?: number,
  maxInputTokens?: number,
  maxOptions?: number,
  maxTotalOptions?: number,
  validation?: {readonly dataPropertiesOnly?: boolean}
) => unknown;
type ValidateDecisionFields = (options: unknown) => void;
type WaitForAbort = <Value>(operation: () => Value | Promise<Value>, signal?: AbortSignal) => Promise<Value>;
type DebertaPacker = (tokenizer: KevTokenizer, request: DecisionRequest) => PackedDebertaRequest;
type DebertaUnpacker = (packed: PackedDebertaRequest, scores: number[]) => DecisionResult;
type PrepareSystemOne = (state: string, questions: NamedQuestions) => PreparedSystemOne;
type FinishSystemOne = (prepared: PreparedSystemOne, result: DecisionResult) => DebertaSystemOneResult;

const validateRequestFieldsForModel = validateRequestFields as ValidateRequestFields;
const validateDecisionFieldsForModel = validateDecisionFields as ValidateDecisionFields;
const waitForAbortOperation = waitForAbort as WaitForAbort;
const packDebertaRequestForModel = packDebertaRequest as DebertaPacker;
const unpackDebertaScoresForModel = unpackDebertaScores as DebertaUnpacker;
const prepareDebertaSystemOneForModel = prepareDebertaSystemOne as PrepareSystemOne;
const finishDebertaSystemOneForModel = finishDebertaSystemOne as FinishSystemOne;

function closedError(): WebDecisionError {
  return new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
}

function checkDeviceAfterCleanup(session: ExperimentalDebertaSession, message: string): void {
  if (session.lifecycle!.loss || session.lifecycle!.disposed)
    throw new WebDecisionError('INFERENCE_ERROR', message, {cause: new Error('The GPU device was lost.')});
}

async function appendChunk(
  session: ExperimentalDebertaSession,
  hidden: unknown,
  seg: readonly number[],
  chunk: DebertaChunk,
  tensor: (values: readonly number[]) => DebertaTensor,
  scores: number[]
): Promise<void> {
  session.runtime!.beginScope();
  let failure: unknown = noFailure;
  try {
    const out = session.head!(
      hidden,
      tensor(seg),
      tensor(chunk.pair_q),
      tensor(chunk.pair_opt),
      ...session.headWeights
    ) as Tensor;
    const host = session.runtime!.empty([1, 3], 'float16', session.runtime!.cpu()).copyFrom(out);
    await session.device!.sync();
    session.checkDevice();
    const raw = host.toRawBytes();
    if (raw.byteLength !== 6) throw new Error('Unexpected output length.');
    const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    for (let index = 0; index < chunk.count; index++) scores.push(decodeFloat16(view.getUint16(index * 2, true)));
  } catch (cause) {
    failure = cause;
  } finally {
    if (!session.lifecycle!.disposed) {
      try {
        session.runtime!.endScope();
      } catch (cause) {
        if (failure === noFailure) failure = cause;
      }
    }
  }
  if (failure !== noFailure) throw failure;
}

async function closeValidationScope(
  session: ExperimentalDebertaSession,
  scopeStarted: boolean,
  gpuScopeStarted: boolean,
  failure: unknown
): Promise<unknown> {
  try {
    const error = gpuScopeStarted ? await session.gpu!.popErrorScope() : null;
    if (error && failure === noFailure) failure = error;
  } catch (cause) {
    if (failure === noFailure) failure = cause;
  }
  if (scopeStarted && !session.lifecycle!.disposed) {
    try {
      session.runtime!.endScope();
    } catch (cause) {
      if (failure === noFailure) failure = cause;
    }
  }
  return failure;
}

async function withValidationScope<Value>(
  session: ExperimentalDebertaSession,
  failureMessage: string,
  operation: () => Value | Promise<Value>
): Promise<Value> {
  let result!: Value;
  let failure: unknown = noFailure;
  let scopeStarted = false;
  let gpuScopeStarted = false;
  try {
    session.checkDevice();
    session.runtime!.beginScope();
    scopeStarted = true;
    session.gpu!.pushErrorScope('validation');
    gpuScopeStarted = true;
    result = await operation();
  } catch (cause) {
    failure = cause;
  } finally {
    failure = await closeValidationScope(session, scopeStarted, gpuScopeStarted, failure);
  }
  if (failure !== noFailure) throw new WebDecisionError('INFERENCE_ERROR', failureMessage, {cause: failure});
  if (session.uncaptured.length) throw new WebDecisionError('INFERENCE_ERROR', session.uncaptured[0]!);
  checkDeviceAfterCleanup(session, failureMessage);
  return result;
}

/** The caller supplies an initialized runtime with an open root scope and checked weights. */
export class ExperimentalDebertaSession {
  runtime: DebertaRuntime | null;
  device: DebertaDevice | null;
  gpu: GPUDevice | null;
  encoder: PackedFunc | null;
  head: PackedFunc | null;
  encoderWeights: Tensor[];
  headWeights: Tensor[];
  tokenizer: KevTokenizer | null;
  uncaptured: string[];
  lifecycle: RuntimeLifecycle | null;
  queue: Promise<unknown>;
  closed: boolean;
  disposal: Promise<void> | null;

  constructor({
    runtime,
    device,
    gpu,
    encoder,
    head,
    encoderWeights,
    headWeights,
    tokenizer,
    uncaptured,
    lifecycle
  }: {
    runtime: DebertaRuntime;
    device: DebertaDevice;
    gpu: GPUDevice;
    encoder: PackedFunc;
    head: PackedFunc;
    encoderWeights: Tensor[];
    headWeights: Tensor[];
    tokenizer: KevTokenizer;
    uncaptured: string[];
    lifecycle: RuntimeLifecycle;
  }) {
    this.runtime = runtime;
    this.device = device;
    this.gpu = gpu;
    this.encoder = encoder;
    this.head = head;
    this.encoderWeights = encoderWeights;
    this.headWeights = headWeights;
    this.tokenizer = tokenizer;
    this.uncaptured = uncaptured;
    this.lifecycle = lifecycle;
    this.queue = Promise.resolve();
    this.closed = false;
    this.disposal = null;
  }

  async decide(request: DecisionRequest, options: DecisionOptions | null = {}): Promise<DecisionResult> {
    if (this.closed) throw closedError();
    const signal = options?.signal;
    signal?.throwIfAborted();
    validateDecisionFieldsForModel(options);
    request = validateRequestFieldsForModel(request, Infinity, Infinity, Infinity, DEBERTA_MAX_OPTIONS, Infinity, {
      dataPropertiesOnly: true
    }) as DecisionRequest;
    if (this.closed) throw closedError();
    let packed: PackedDebertaRequest;
    try {
      packed = packDebertaRequestForModel(this.tokenizer!, request);
    } catch (cause) {
      throw new WebDecisionError('INVALID_INPUT', 'Invalid DeBERTa request.', {cause});
    }
    if (this.closed) throw closedError();
    const schedule = () => {
      if (this.closed) throw closedError();
      const task = this.queue.then(() => {
        signal?.throwIfAborted();
        return this.run(packed);
      });
      this.queue = task.catch(() => {});
      return task;
    };
    return signal ? waitForAbortOperation(schedule, signal) : schedule();
  }

  async systemOne(
    state: string,
    questions: NamedQuestions,
    options: DecisionOptions | null = {}
  ): Promise<DebertaSystemOneResult> {
    if (this.closed) throw closedError();
    const prepared = prepareDebertaSystemOneForModel(state, questions);
    const result = await this.decide(prepared.request as DecisionRequest, options);
    return finishDebertaSystemOneForModel(prepared, result);
  }

  checkDevice(): void {
    if (this.lifecycle!.loss || this.lifecycle!.disposed) throw new Error('The GPU device was lost.');
    if (this.uncaptured.length) throw new Error(this.uncaptured[0]!);
  }

  async run(packed: PackedDebertaRequest): Promise<DecisionResult> {
    return withValidationScope(this, 'DeBERTa inference failed.', async () => {
      const tensor = (values: readonly number[]): DebertaTensor => {
        const value = this.runtime!.empty([1, values.length], 'int32', this.device!);
        value.copyFrom(Int32Array.from(values));
        return value;
      };
      const {input_ids, attention_mask, seg} = packed.inputs;
      const hidden = this.encoder!(tensor(input_ids), tensor(attention_mask), ...this.encoderWeights);
      const scores: number[] = [];
      for (const chunk of packed.chunks) await appendChunk(this, hidden, seg, chunk, tensor, scores);
      return unpackDebertaScoresForModel(packed, scores);
    });
  }

  async releaseCachedMemory(): Promise<void> {
    if (this.closed) throw closedError();
    const task = this.queue.then(() =>
      withValidationScope(this, 'Could not release DeBERTa cached memory.', async () => {
        this.runtime!.getGlobalFunc('vm.builtin.memory_manager.clear')();
        await this.device!.sync();
        this.checkDevice();
      })
    );
    this.queue = task.catch(() => {});
    return task;
  }

  async dispose(): Promise<void> {
    if (this.disposal) return this.disposal;
    this.closed = true;
    const runtime = this.runtime!;
    const lifecycle = this.lifecycle!;
    this.disposal = (async () => {
      await this.queue;
      try {
        if (!lifecycle.disposed) {
          try {
            if (!lifecycle.loss) runtime.endScope();
          } finally {
            runtime.dispose();
          }
        }
      } finally {
        this.encoderWeights = [];
        this.headWeights = [];
        this.runtime = this.device = this.gpu = this.encoder = this.head = this.tokenizer = this.lifecycle = null;
        this.uncaptured = [];
      }
    })();
    return this.disposal;
  }
}
