import {validateDecisionFields, validateRequestFields} from './request-fields.mjs';
import {KEV_MAX_OPTIONS, WebDecisionError, decodeFloat16, packKevRequest, unpackKevScores} from './kev.js';
import {assertKevAttentionFits, type KevAttentionProfile} from './kev-gpu.js';
import {waitForAbort} from './assets.js';
import {finishKevSystemOne, prepareKevSystemOne} from './kev/system-one.mjs';
import type {KevLimits, KevPackOptions, KevScoreResult, KevTokenizer} from './kev.js';
import type {DecisionRequest, DecisionResult, KevQuestion, KevResult, KevValue} from './curated/types.js';
import type {DLDevice, Instance, PackedFunc, Tensor} from './runtime/tvm.js';

const noFailure = Symbol('no failure');

type DecisionOptions = Readonly<{signal?: AbortSignal | undefined; [key: string]: unknown}>;
type DecisionModelLimits = KevLimits &
  Readonly<{
    attentionProfile?: KevAttentionProfile;
    maxInputTokens: number | undefined;
    maxStateTokens: number;
    maxBranchTokens: number;
    maxQuestions: number | undefined;
    maxOptions: number | undefined;
    maxInputCharacters: number | undefined;
  }>;
type RuntimeLifecycle = Readonly<{
  lossSignal: AbortSignal | undefined;
  disposed: boolean;
  loss: unknown;
}>;
type NamedKevQuestions = Readonly<Record<string, KevQuestion>> | Map<string, KevQuestion>;
type PreparedKevSystemOne = Readonly<{
  request: unknown;
  metadata: readonly unknown[];
  packing: KevPackOptions;
}>;
type DecisionModelState = Readonly<{
  runtime: Instance;
  device: DLDevice;
  gpu: GPUDevice;
  fn: PackedFunc | null;
  weights: Tensor[];
  tokenizer: KevTokenizer | null;
  lifecycle: RuntimeLifecycle;
}>;
type ValidateRequestFields = (
  request: unknown,
  maxQuestions?: number,
  maxInputCharacters?: number,
  maxInputTokens?: number,
  maxOptions?: number
) => unknown;
type ValidateDecisionFields = (options: unknown) => void;
type WaitForAbort = <Value>(operation: () => Value | Promise<Value>, signal?: AbortSignal) => Promise<Value>;
type AssertKevAttentionFits = (
  sequenceLength: number,
  limits?: GPUSupportedLimits | undefined,
  profile?: KevAttentionProfile | undefined
) => void;
type PrepareKevSystemOne = (
  state: KevValue,
  questions: NamedKevQuestions,
  maxInputCharacters?: number
) => PreparedKevSystemOne;
type FinishKevSystemOne = (prepared: PreparedKevSystemOne, result: KevScoreResult, model?: string) => KevResult;

const validateRequestFieldsForModel = validateRequestFields as ValidateRequestFields;
const validateDecisionFieldsForModel = validateDecisionFields as ValidateDecisionFields;
const waitForAbortOperation = waitForAbort as WaitForAbort;
const assertKevAttentionFitsForModel: AssertKevAttentionFits = assertKevAttentionFits;
const prepareKevSystemOneForModel = prepareKevSystemOne as PrepareKevSystemOne;
const finishKevSystemOneForModel = finishKevSystemOne as FinishKevSystemOne;

function sessionClosed(): WebDecisionError {
  return new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
}

function deviceLost(message = 'The GPU device was lost. Load the model again.'): WebDecisionError {
  return new WebDecisionError('INFERENCE_ERROR', message);
}

function captureModelState(model: DecisionModel): DecisionModelState {
  const {runtime, device, gpu, fn, tokenizer, lifecycle} = model;
  if (lifecycle === null || lifecycle.loss || lifecycle.disposed) throw deviceLost();
  if (runtime === null || device === null || gpu === null) throw deviceLost();
  return {runtime, device, gpu, fn, weights: model.weights, tokenizer, lifecycle};
}

function assertNoUncapturedErrors(model: DecisionModel): void {
  if (model.uncaptured.length) throw new WebDecisionError('INFERENCE_ERROR', model.uncaptured[0]!);
}

function checkDeviceAfterCleanup(session: DecisionModelState, message: string): void {
  if (session.lifecycle.loss || session.lifecycle.disposed)
    throw new WebDecisionError('INFERENCE_ERROR', message, {cause: new Error('The GPU device was lost.')});
}

async function closeValidationScope(
  session: DecisionModelState,
  scopeStarted: boolean,
  scopePushed: boolean,
  failure: unknown
): Promise<unknown> {
  try {
    const gpuError = scopePushed
      ? await waitForAbortOperation(() => session.gpu.popErrorScope(), session.lifecycle.lossSignal)
      : null;
    if (gpuError && failure === noFailure) failure = gpuError;
  } catch (cause) {
    if (failure === noFailure) failure = cause;
  }
  if (
    scopeStarted &&
    !session.lifecycle.loss &&
    !session.lifecycle.lossSignal?.aborted &&
    !session.lifecycle.disposed
  ) {
    try {
      session.runtime.endScope();
    } catch (cause) {
      if (failure === noFailure) failure = cause;
    }
  }
  return failure;
}

async function withValidationScope<Value>(
  session: DecisionModelState,
  failureMessage: string,
  operation: () => Value | Promise<Value>
): Promise<Value> {
  let result!: Value;
  let failure: unknown = noFailure;
  let scopeStarted = false;
  let scopePushed = false;
  try {
    session.runtime.beginScope();
    scopeStarted = true;
    session.gpu.pushErrorScope('validation');
    scopePushed = true;
    result = await operation();
  } catch (cause) {
    failure = cause;
  } finally {
    failure = await closeValidationScope(session, scopeStarted, scopePushed, failure);
  }
  if (failure !== noFailure) throw new WebDecisionError('INFERENCE_ERROR', failureMessage, {cause: failure});
  return result;
}

async function inferDecisionModel(
  model: DecisionModel,
  request: DecisionRequest,
  options: DecisionOptions | null = {},
  packing: KevPackOptions = {}
): Promise<KevScoreResult> {
  if (model.closed) throw sessionClosed();
  const signal = options?.signal;
  signal?.throwIfAborted();
  validateDecisionFieldsForModel(options);
  request = validateRequestFieldsForModel(
    request,
    model.limits?.maxQuestions,
    model.limits?.maxInputCharacters,
    model.limits?.maxInputTokens,
    model.limits?.maxOptions ?? KEV_MAX_OPTIONS
  ) as DecisionRequest;
  if (model.closed) throw sessionClosed();
  // Pack synchronously before entering the queue so later caller mutations do
  // not affect an already-submitted request.
  const packed = packKevRequest(model.tokenizer!, request, model.limits!, packing);
  if (model.closed) throw sessionClosed();
  const schedule = () => {
    if (model.closed) throw sessionClosed();
    const task = model.queue.then(() => {
      signal?.throwIfAborted();
      return model.run(packed);
    });
    model.queue = task.catch(() => {});
    return task;
  };
  // Cancellation stops the caller waiting, but the internal queue must
  // still wait for any GPU work already in progress before using the device.
  return signal ? waitForAbortOperation(schedule, signal) : schedule();
}

export class DecisionModel {
  runtime: Instance | null;
  device: DLDevice | null;
  gpu: GPUDevice | null;
  fn: PackedFunc | null;
  weights: Tensor[];
  tokenizer: KevTokenizer | null;
  limits: DecisionModelLimits | null | undefined;
  uncaptured: string[];
  lifecycle: RuntimeLifecycle | null;
  supportsSystemOne: boolean;
  modelId: string;
  closed: boolean;
  queue: Promise<unknown>;
  disposal: Promise<void> | null;

  constructor(
    runtime: Instance,
    device: DLDevice,
    gpu: GPUDevice,
    fn: PackedFunc,
    weights: Tensor[],
    tokenizer: KevTokenizer,
    limits: DecisionModelLimits,
    uncaptured: string[],
    lifecycle: RuntimeLifecycle,
    supportsSystemOne = false,
    modelId = 'kev-0.6b-q4f16'
  ) {
    this.runtime = runtime;
    this.device = device;
    this.gpu = gpu;
    this.fn = fn;
    this.weights = weights;
    this.tokenizer = tokenizer;
    this.limits = limits;
    this.uncaptured = uncaptured;
    this.lifecycle = lifecycle;
    this.supportsSystemOne = supportsSystemOne;
    this.modelId = modelId;
    this.closed = false;
    this.queue = Promise.resolve();
    this.disposal = null;
  }

  async decide(request: DecisionRequest, options: DecisionOptions | null = {}): Promise<DecisionResult> {
    return inferDecisionModel(this, request, options);
  }

  async systemOne(
    state: KevValue,
    questions: NamedKevQuestions,
    options: DecisionOptions | null = {}
  ): Promise<KevResult> {
    if (!this.supportsSystemOne)
      throw new WebDecisionError(
        'UNSUPPORTED_ENVIRONMENT',
        'Typed System One input is unavailable for this model route.'
      );
    if (this.closed) throw sessionClosed();
    options?.signal?.throwIfAborted();
    validateDecisionFieldsForModel(options);
    const prepared = prepareKevSystemOneForModel(state, questions, this.limits?.maxInputCharacters);
    const result = await inferDecisionModel(this, prepared.request as DecisionRequest, options, prepared.packing);
    return finishKevSystemOneForModel(prepared, result, this.modelId);
  }

  async run(packed: ReturnType<typeof packKevRequest>): Promise<KevScoreResult> {
    const session = captureModelState(this);
    assertNoUncapturedErrors(this);
    const n = packed.ids.length;
    assertKevAttentionFitsForModel(n, session.gpu.limits, this.limits?.attentionProfile);
    const result = await withValidationScope(session, 'Model inference failed.', async () => {
      const ids = session.runtime.empty([1, n], 'int32', session.device);
      const mask = session.runtime.empty([1, n], 'int32', session.device);
      ids.copyFrom(Int32Array.from(packed.ids));
      mask.copyFrom(new Int32Array(n).fill(1));
      const y = session.fn!(ids, mask, ...session.weights) as Tensor;
      const host = session.runtime.empty([1, n], 'float16', session.runtime.cpu()).copyFrom(y);
      await waitForAbortOperation(() => session.device.sync(), session.lifecycle.lossSignal);
      if (session.lifecycle.loss || session.lifecycle.disposed) throw new Error('The GPU device was lost.');
      const raw = host.toRawBytes();
      if (raw.byteLength !== n * 2) throw new Error('Unexpected output length.');
      const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
      return unpackKevScores(
        Array.from({length: n}, (_, i) => decodeFloat16(view.getUint16(i * 2, true))),
        packed
      );
    });
    assertNoUncapturedErrors(this);
    checkDeviceAfterCleanup(session, 'Model inference failed.');
    return result;
  }

  /** Release unused TVM allocator buffers after earlier queued requests finish. */
  async releaseCachedMemory(): Promise<void> {
    if (this.closed) throw sessionClosed();
    const task = this.queue.then(async () => {
      const session = captureModelState(this);
      assertNoUncapturedErrors(this);
      await withValidationScope(session, 'Could not release cached runtime memory.', async () => {
        session.runtime.getGlobalFunc('vm.builtin.memory_manager.clear')();
        await waitForAbortOperation(() => session.device.sync(), session.lifecycle.lossSignal);
        if (session.lifecycle.loss || session.lifecycle.disposed) throw new Error('The GPU device was lost.');
      });
      assertNoUncapturedErrors(this);
      checkDeviceAfterCleanup(session, 'Could not release cached runtime memory.');
    });
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
        // Do not retain 394 tensor wrappers and the runtime after closing.
        this.weights = [];
        this.fn = this.runtime = this.device = this.gpu = this.tokenizer = this.lifecycle = null;
        this.uncaptured = [];
      }
    })();
    return this.disposal;
  }
}
