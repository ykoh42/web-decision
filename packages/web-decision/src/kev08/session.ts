import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {waitForAbort} from '../assets.js';
import {checkGpuDevice, createCausalInputs, withGpuScope} from '../gpu-session-utils.js';
import {WebDecisionError} from '../kev.js';
import type {KevTokenizer} from '../kev.js';
import {KEV08_MARKERS, KEV08_MAX_OPTIONS, packKev08Request, unpackKev08Scores} from '../kev08.js';
import {finishKev08SystemOne, prepareKev08SystemOne} from './system-one.mjs';

interface Kev08Tensor {
  copyFrom(value: unknown): Kev08Tensor;
  toArray(): unknown;
}
interface Kev08Runtime {
  empty(shape: number[], dtype: string, device: unknown): Kev08Tensor;
  cpu(): unknown;
  beginScope(): unknown;
  endScope(): unknown;
  dispose(): unknown;
  getGlobalFunc(name: string): () => unknown;
}
interface Kev08Gpu {
  pushErrorScope(filter: 'out-of-memory' | 'validation'): unknown;
  popErrorScope(): PromiseLike<unknown>;
}
interface Kev08Device {
  sync(): unknown;
}
interface Kev08Lifecycle {
  disposed: boolean;
  loss: unknown;
}
interface Kev08Row {
  ids: number[];
  questionSlot: number;
  optionSlots: number[];
  options: string[];
  allowSingleOption?: boolean;
  allowEmptyOptions?: boolean;
  allowDuplicateOptions?: boolean;
}
interface RawRow {
  ids?: unknown;
  questionSlot?: unknown;
  optionSlots?: unknown;
  options?: unknown;
}
interface RowFlags {
  allowEmptyOptions: boolean;
  allowDuplicateOptions: boolean;
  allowSingleOption: boolean;
}
interface Kev08SessionOptions {
  runtime: Kev08Runtime;
  device: Kev08Device;
  gpu: Kev08Gpu;
  fn: (...args: unknown[]) => unknown;
  weights: unknown[];
  tokenizer: KevTokenizer;
  limits: {maxInputTokens: number; maxStateTokens: number};
  uncaptured: string[];
  lifecycle: Kev08Lifecycle;
}
interface DecisionOptions {
  signal?: AbortSignal | null;
}

const invalidRow = () => new WebDecisionError('INVALID_INPUT', 'Invalid Kev 0.8B token or answer positions.');

function snapshotIds(source: unknown, maxTokens: number) {
  if (!Array.isArray(source) || source.length < 2 || source.length > maxTokens) throw invalidRow();
  const ids = [],
    count = source.length;
  for (let i = 0; i < count; i++) {
    if (!Object.hasOwn(source, i)) throw invalidRow();
    const id = source[i];
    if (!Number.isSafeInteger(id) || id < 0 || id >= 248320) throw invalidRow();
    ids.push(id);
  }
  if (ids[0] !== KEV08_MARKERS.state) throw invalidRow();
  return ids;
}

function snapshotChoices(
  sourceSlots: unknown,
  sourceOptions: unknown,
  ids: number[],
  questionSlot: number,
  {allowEmptyOptions, allowDuplicateOptions, allowSingleOption}: RowFlags
) {
  const minimum = allowSingleOption ? 1 : 2;
  if (
    !Array.isArray(sourceSlots) ||
    !Array.isArray(sourceOptions) ||
    sourceSlots.length !== sourceOptions.length ||
    sourceSlots.length < minimum ||
    sourceSlots.length > KEV08_MAX_OPTIONS
  )
    throw invalidRow();
  const optionSlots = [],
    options = [],
    seen = new Set<string>(),
    count = sourceSlots.length;
  for (let i = 0; i < count; i++) {
    if (!Object.hasOwn(sourceSlots, i) || !Object.hasOwn(sourceOptions, i)) throw invalidRow();
    const slot = sourceSlots[i],
      option = sourceOptions[i];
    if (
      !Number.isSafeInteger(slot) ||
      slot < 0 ||
      slot >= questionSlot ||
      ids[slot] !== KEV08_MARKERS.close ||
      (i > 0 && slot <= optionSlots[i - 1]!) ||
      typeof option !== 'string' ||
      (!allowEmptyOptions && !option.trim()) ||
      (!allowDuplicateOptions && seen.has(option))
    )
      throw invalidRow();
    optionSlots.push(slot);
    options.push(option);
    if (!allowDuplicateOptions) seen.add(option);
  }
  return {optionSlots, options};
}

/** Snapshot validated tokens and gather indices before invoking the TVM GPU graph. */
export function validateKev08Row(row: RawRow | null | undefined, maxTokens: number) {
  if (!Number.isSafeInteger(maxTokens) || maxTokens < 2) throw invalidRow();
  const allowSingleOption = snapshotRowFlag(row, 'allowSingleOption');
  const allowEmptyOptions = snapshotRowFlag(row, 'allowEmptyOptions');
  const allowDuplicateOptions = snapshotRowFlag(row, 'allowDuplicateOptions');
  const ids = snapshotIds(row?.ids, maxTokens),
    questionSlot = row?.questionSlot;
  if (
    !Number.isSafeInteger(questionSlot) ||
    questionSlot !== ids.length - 1 ||
    ids[questionSlot as number] !== KEV08_MARKERS.decide
  )
    throw invalidRow();
  const {optionSlots, options} = snapshotChoices(row?.optionSlots, row?.options, ids, questionSlot as number, {
    allowEmptyOptions,
    allowDuplicateOptions,
    allowSingleOption
  });
  const validated: Kev08Row = {ids, questionSlot: questionSlot as number, optionSlots, options};
  if (allowSingleOption) validated.allowSingleOption = true;
  if (allowEmptyOptions) validated.allowEmptyOptions = true;
  if (allowDuplicateOptions) validated.allowDuplicateOptions = true;
  return validated;
}

function snapshotRowFlag(row: RawRow | null | undefined, key: string) {
  const descriptor = row && Object.getOwnPropertyDescriptor(row, key);
  if (!descriptor) return false;
  if (!Object.hasOwn(descriptor, 'value') || typeof descriptor.value !== 'boolean') throw invalidRow();
  return descriptor.value;
}

export class ExperimentalKev08Session {
  declare runtime: Kev08Runtime | null;
  declare device: Kev08Device | null;
  declare gpu: Kev08Gpu | null;
  declare fn: ((...args: unknown[]) => unknown) | null;
  declare weights: unknown[];
  declare tokenizer: KevTokenizer | null;
  declare limits: {maxInputTokens: number; maxStateTokens: number};
  declare uncaptured: string[];
  declare lifecycle: Kev08Lifecycle | null;
  declare queue: Promise<unknown>;
  declare closed: boolean;
  declare disposal: Promise<void> | null;
  constructor({runtime, device, gpu, fn, weights, tokenizer, limits, uncaptured, lifecycle}: Kev08SessionOptions) {
    Object.assign(this, {runtime, device, gpu, fn, weights, tokenizer, limits, uncaptured, lifecycle});
    this.queue = Promise.resolve();
    this.closed = false;
    this.disposal = null;
  }

  async decide(request: unknown, options: DecisionOptions | null = {}) {
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    const signal = options?.signal;
    signal?.throwIfAborted();
    validateDecisionFields(options);
    request = validateRequestFields(request, Infinity, Infinity, Infinity, KEV08_MAX_OPTIONS);
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    const packed = packKev08Request(this.tokenizer!, request as Parameters<typeof packKev08Request>[1], this.limits);
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    return this.enqueue(packed, signal);
  }

  async systemOne(state: unknown, questions: unknown, options: DecisionOptions | null = {}) {
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    const signal = options?.signal;
    signal?.throwIfAborted();
    validateDecisionFields(options);
    const prepared = prepareKev08SystemOne(state, questions);
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    const packed = packKev08Request(this.tokenizer!, prepared.request, this.limits, prepared.packing);
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    const result = await this.enqueue(packed, signal);
    return finishKev08SystemOne(prepared, result);
  }

  enqueue(packed: Parameters<ExperimentalKev08Session['run']>[0], signal?: AbortSignal | null) {
    const schedule = () => {
      if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
      const task = this.queue.then(() => {
        signal?.throwIfAborted();
        return this.run(packed);
      });
      this.queue = task.catch(() => {});
      return task;
    };
    return signal ? waitForAbort(schedule, signal) : schedule();
  }

  async run(packed: {rows: Kev08Row[]; tokenCount: number}) {
    const sourceRows = packed?.rows;
    if (!Array.isArray(sourceRows) || !sourceRows.length) throw invalidRow();
    const rows = [],
      scores = [],
      rowCount = sourceRows.length;
    let tokenCount = 0;
    for (let i = 0; i < rowCount; i++) {
      if (!Object.hasOwn(sourceRows, i)) throw invalidRow();
      const row = validateKev08Row(sourceRows[i], this.limits.maxInputTokens);
      rows.push(row);
      tokenCount += row.ids.length;
    }
    for (const row of rows) {
      scores.push(
        await withGpuScope(this, 'Kev 0.8B inference failed.', async () => {
          const {ids, positions, mask, valid} = createCausalInputs(this, row.ids);
          const question = this.runtime!.empty([1], 'int32', this.device).copyFrom(Int32Array.of(row.questionSlot));
          const choices = this.runtime!.empty([row.optionSlots.length], 'int32', this.device).copyFrom(
            Int32Array.from(row.optionSlots)
          );
          const output = this.fn!(ids, positions, mask, valid, question, choices, ...this.weights);
          const host = this.runtime!.empty([row.optionSlots.length], 'float32', this.runtime!.cpu()).copyFrom(output);
          await this.device!.sync();
          checkGpuDevice(this);
          return host.toArray();
        })
      );
    }
    return unpackKev08Scores(scores, {rows, tokenCount});
  }

  async releaseCachedMemory() {
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    const task = this.queue.then(() =>
      withGpuScope(this, 'Could not release Kev 0.8B cached memory.', async () => {
        this.runtime!.getGlobalFunc('vm.builtin.memory_manager.clear')();
        await this.device!.sync();
      })
    );
    this.queue = task.catch(() => {});
    return task;
  }

  async dispose() {
    if (this.disposal) return this.disposal;
    this.closed = true;
    this.disposal = (async () => {
      await this.queue;
      try {
        if (!this.lifecycle!.disposed) {
          try {
            if (!this.lifecycle!.loss) this.runtime!.endScope();
          } finally {
            this.runtime!.dispose();
          }
        }
      } finally {
        this.weights = [];
        this.runtime = this.device = this.gpu = this.fn = this.tokenizer = this.lifecycle = null;
        this.uncaptured = [];
      }
    })();
    return this.disposal;
  }
}
