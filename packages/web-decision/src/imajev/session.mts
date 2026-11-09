// Isolated session integration against the experimental five-input graph ABI.
// The diagnostic graph is separate; no public loader or model ID is provided here.
import {waitForAbort} from '../abort.js';
import {checkGpuDevice, createCausalInputs} from '../gpu-session-utils.js';
import {assertUsable, withStrandsScope} from '../strands/scope.mjs';
import {WebDecisionError} from '../kev.js';
import {validateDecisionFields} from '../request-fields.mjs';
import {validateH2oSignal} from '../h2o/signal.mjs';
import type {
  ImajevProcessedResult,
  ImajevResultProcessor,
  PreparedImajevQuestion,
  PreparedImajevRequest
} from './result-processor.mjs';

interface ImajevTensor {
  copyFrom(value: unknown): ImajevTensor;
  toArray(): unknown;
}

interface ImajevRuntime {
  empty(shape: number[], dtype: 'int32' | 'float32', device: unknown): ImajevTensor;
  cpu(): unknown;
  getGlobalFunc(name: string): () => unknown;
  endScope(): void;
  dispose(): void;
}

interface ImajevLifecycle {
  readonly loss: unknown;
  readonly disposed: unknown;
}

interface ImajevDisposalGuard {
  readonly blocked: boolean;
  retain(): void;
}

interface ImajevSessionOptions {
  readonly runtime: unknown;
  readonly device: unknown;
  readonly gpu: unknown;
  readonly fn: unknown;
  readonly weights: unknown;
  readonly processor: unknown;
  readonly maxInputTokens: unknown;
  readonly uncaptured: unknown;
  readonly lifecycle: unknown;
  readonly disposalGuard: unknown;
}

interface ImajevDecisionOptions {
  readonly signal?: unknown;
  readonly [field: string]: unknown;
}

interface ImajevPreparedQuestionCandidate {
  readonly inputIds?: unknown;
  readonly lastIndex?: unknown;
}

interface ImajevGraphOutputCandidate {
  readonly dtype?: unknown;
  readonly shape?: unknown;
}

interface ImajevScopeLease {
  prepareReadback(): void;
  captureReadback(): void;
  sync(): Promise<void>;
  syncWithoutReadback(): Promise<void>;
}

type ImajevGraphFunction = (...args: unknown[]) => unknown;

interface ImajevCausalInputs {
  readonly ids: unknown;
  readonly positions: unknown;
  readonly mask: unknown;
  readonly valid: unknown;
}

// Failed disposal must retain the whole session, including weight wrappers,
// until the realm is discarded. A guarded runtime alone may not own them.
const retainedSessions = new Set<ImajevSession>();

function validateQuestion(question: unknown, maxTokens: number): asserts question is PreparedImajevQuestion {
  const ids = (question as ImajevPreparedQuestionCandidate | null | undefined)?.inputIds;
  if (
    !Array.isArray(ids) ||
    ids.length < 2 ||
    ids.length > maxTokens ||
    Array.from(ids as unknown[]).some(
      (id: unknown) => !Number.isSafeInteger(id) || (id as number) < 0 || (id as number) >= 248320
    ) ||
    !Number.isSafeInteger((question as ImajevPreparedQuestionCandidate).lastIndex) ||
    (question as ImajevPreparedQuestionCandidate).lastIndex !== ids.length - 1
  )
    throw new WebDecisionError('INVALID_INPUT', 'Invalid Imajev tokens or final position.');
}

function validatePrepared(prepared: unknown, maxTokens: number): asserts prepared is PreparedImajevRequest {
  const candidate = prepared as {readonly questions?: unknown} | null | undefined;
  if (!Array.isArray(candidate?.questions) || candidate.questions.length < 1 || candidate.questions.length > 8)
    throw new WebDecisionError('INVALID_INPUT', 'Invalid Imajev question count.');
  for (const question of candidate.questions as unknown[]) validateQuestion(question, maxTokens);
}

function checkAfterScope(session: ImajevSession, message: string): void {
  try {
    checkGpuDevice(session);
  } catch (cause) {
    throw new WebDecisionError('INFERENCE_ERROR', message, {cause});
  }
}

/** Experiment only. The caller must install the Strands disposal guard after
 * tracking the runtime and before initWebGPU registers TVM's loss callback. */
export class ImajevSession {
  declare runtime: ImajevRuntime | null;
  declare device: unknown;
  declare gpu: unknown;
  declare fn: ImajevGraphFunction | null;
  declare weights: readonly unknown[];
  declare processor: ImajevResultProcessor | null;
  declare maxInputTokens: number;
  declare uncaptured: unknown[];
  declare lifecycle: ImajevLifecycle | null;
  declare disposalGuard: ImajevDisposalGuard | null;
  declare queue: Promise<unknown>;
  declare closed: boolean;
  declare disposal: Promise<void> | null;

  constructor({
    runtime,
    device,
    gpu,
    fn,
    weights,
    processor,
    maxInputTokens,
    uncaptured,
    lifecycle,
    disposalGuard
  }: ImajevSessionOptions) {
    if (
      !processor ||
      typeof (processor as ImajevResultProcessor).prepare !== 'function' ||
      typeof (processor as ImajevResultProcessor).finish !== 'function' ||
      !Number.isSafeInteger(maxInputTokens) ||
      (maxInputTokens as number) < 2 ||
      (maxInputTokens as number) > 4096 ||
      !disposalGuard ||
      typeof (disposalGuard as ImajevDisposalGuard).retain !== 'function' ||
      typeof (disposalGuard as ImajevDisposalGuard).blocked !== 'boolean'
    )
      throw new WebDecisionError('INVALID_ASSET', 'Invalid Imajev processor or graph token limit.');
    Object.assign(this, {
      runtime: runtime as ImajevRuntime,
      device,
      gpu,
      fn: fn as ImajevGraphFunction,
      weights: weights as readonly unknown[],
      processor: processor as ImajevResultProcessor,
      maxInputTokens: maxInputTokens as number,
      uncaptured,
      lifecycle: lifecycle as ImajevLifecycle,
      disposalGuard: disposalGuard as ImajevDisposalGuard
    });
    this.queue = Promise.resolve();
    this.closed = false;
    this.disposal = null;
  }

  async decide(request: unknown, options: ImajevDecisionOptions = {}): Promise<ImajevProcessedResult> {
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    assertUsable(this);
    const signal = (options as ImajevDecisionOptions | null | undefined)?.signal as AbortSignal | undefined;
    if (signal !== undefined) validateH2oSignal(signal);
    signal?.throwIfAborted();
    validateDecisionFields(options);
    // Snapshot and validate before enqueue, even if another request is running.
    const prepared = this.processor!.prepare(request);
    validatePrepared(prepared, this.maxInputTokens);
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    const schedule = (): Promise<ImajevProcessedResult> => {
      if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
      const task = this.queue.then(() => {
        signal?.throwIfAborted();
        return this.run(prepared, signal);
      });
      this.queue = task.catch(() => {});
      return task;
    };
    return signal ? waitForAbort(schedule, signal) : schedule();
  }

  async run(prepared: PreparedImajevRequest, signal: AbortSignal | undefined): Promise<ImajevProcessedResult> {
    const rows: Float32Array[] = [];
    for (const question of prepared.questions) {
      signal?.throwIfAborted();
      validateQuestion(question, this.maxInputTokens);
      const row = await withStrandsScope(this, 'Imajev inference failed.', async (lease: ImajevScopeLease) => {
        const {ids, positions, mask, valid} = createCausalInputs(this, question.inputIds) as ImajevCausalInputs;
        const last = this.runtime!.empty([1], 'int32', this.device).copyFrom(Int32Array.of(question.lastIndex));
        signal?.throwIfAborted();
        const output = this.fn!(ids, positions, mask, valid, last, ...this.weights);
        const graphOutput = output as ImajevGraphOutputCandidate | null | undefined;
        if (
          graphOutput?.dtype !== 'float32' ||
          !Array.isArray(graphOutput.shape) ||
          (graphOutput.shape as unknown[]).length !== 2 ||
          (graphOutput.shape as unknown[])[0] !== 1 ||
          (graphOutput.shape as unknown[])[1] !== 256
        )
          throw new Error('The Imajev graph output does not match [1, 256] FP32.');
        const host = this.runtime!.empty([1, 256], 'float32', this.runtime!.cpu());
        lease.prepareReadback();
        try {
          host.copyFrom(output);
        } finally {
          lease.captureReadback();
        }
        await lease.sync();
        checkGpuDevice(this);
        const readout = host.toArray();
        if (
          !(readout instanceof Float32Array) ||
          readout.length !== 256 ||
          (typeof SharedArrayBuffer !== 'undefined' && readout.buffer instanceof SharedArrayBuffer) ||
          readout.some((value) => !Number.isFinite(value))
        )
          throw new Error('The Imajev graph returned an invalid 256-score FP32 row.');
        // Copy before this scope ends or another question reuses host storage.
        return new Float32Array(readout);
      });
      // Loss can occur while async GPU error scopes are being popped.
      checkAfterScope(this, 'Imajev inference failed.');
      rows.push(row);
    }
    signal?.throwIfAborted();
    return this.processor!.finish(prepared, rows);
  }

  async releaseCachedMemory(): Promise<void> {
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    assertUsable(this);
    const task = this.queue.then(async () => {
      await withStrandsScope(this, 'Could not release Imajev cached memory.', async (lease: ImajevScopeLease) => {
        this.runtime!.getGlobalFunc('vm.builtin.memory_manager.clear')();
        await lease.syncWithoutReadback();
      });
      checkAfterScope(this, 'Could not release Imajev cached memory.');
    });
    this.queue = task.catch(() => {});
    return task;
  }

  async dispose(): Promise<void> {
    if (this.disposal) return this.disposal;
    this.closed = true;
    this.disposal = (async () => {
      await this.queue;
      try {
        assertUsable(this);
        if (this.lifecycle!.loss || this.lifecycle!.disposed)
          throw new Error('The GPU device was lost before disposal.');
        this.runtime!.endScope();
        if (this.lifecycle!.loss || this.lifecycle!.disposed || this.disposalGuard!.blocked)
          throw new Error('The GPU device changed during disposal.');
        this.runtime!.dispose();
        if (this.lifecycle!.loss || this.disposalGuard!.blocked)
          throw new Error('The GPU device changed during disposal.');
      } catch (cause) {
        retainedSessions.add(this);
        this.disposalGuard!.retain();
        throw new WebDecisionError('INFERENCE_ERROR', 'Could not safely dispose the Imajev session.', {cause});
      }
      this.weights = [];
      this.runtime = this.device = this.gpu = this.fn = this.processor = this.lifecycle = null;
      this.disposalGuard = null;
      this.uncaptured = [];
    })();
    return this.disposal;
  }
}
