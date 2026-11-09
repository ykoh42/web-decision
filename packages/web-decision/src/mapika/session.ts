import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {waitForAbort} from '../assets.js';
import {checkGpuDevice, createCausalInputs, withGpuScope} from '../gpu-session-utils.js';
import {WebDecisionError} from '../kev.js';
import {MAPIKA_MAX_OPTIONS, packMapikaRequest, unpackMapikaScores} from '../mapika.js';
import {finishMapikaSystemOne, prepareMapikaSystemOne} from './system-one.mjs';

interface MapikaTensor {
  copyFrom(value: unknown): MapikaTensor;
  toArray(): unknown;
}
interface MapikaRuntime {
  empty(shape: number[], dtype: string, device: unknown): MapikaTensor;
  cpu(): unknown;
  beginScope(): unknown;
  endScope(): unknown;
  dispose(): unknown;
  getGlobalFunc(name: string): () => unknown;
}
interface MapikaGpu {
  pushErrorScope(filter: 'out-of-memory' | 'validation'): unknown;
  popErrorScope(): PromiseLike<unknown>;
}
interface MapikaDevice {
  sync(): unknown;
}
interface MapikaLifecycle {
  disposed: boolean;
  loss: unknown;
}
interface MapikaLabels {
  readonly names: readonly string[];
  readonly ids: readonly number[];
  readonly open: readonly number[];
}
interface MapikaLimits {
  readonly maxStateTokens: number;
  readonly maxInputTokens: number;
  readonly controlIds: readonly number[];
  readonly temperature: number;
}
interface MapikaSessionOptions {
  runtime: MapikaRuntime;
  device: MapikaDevice;
  gpu: MapikaGpu;
  fn: (...args: unknown[]) => unknown;
  weights: unknown[];
  tokenizer: unknown;
  labels: MapikaLabels;
  limits: MapikaLimits;
  uncaptured: string[];
  lifecycle: MapikaLifecycle;
  modelId?: string | undefined;
}
interface DecisionOptions {
  signal?: AbortSignal | null;
}
type PackedMapikaRequest = Parameters<typeof unpackMapikaScores>[1];

export class ExperimentalMapikaSession {
  declare runtime: MapikaRuntime | null;
  declare device: MapikaDevice | null;
  declare gpu: MapikaGpu | null;
  declare fn: ((...args: unknown[]) => unknown) | null;
  declare weights: unknown[];
  declare tokenizer: unknown;
  declare labels: MapikaLabels | null;
  declare limits: MapikaLimits;
  declare uncaptured: string[];
  declare lifecycle: MapikaLifecycle | null;
  declare modelId: string;
  declare queue: Promise<unknown>;
  declare closed: boolean;
  declare disposal: Promise<void> | null;
  constructor({
    runtime,
    device,
    gpu,
    fn,
    weights,
    tokenizer,
    labels,
    limits,
    uncaptured,
    lifecycle,
    modelId
  }: MapikaSessionOptions) {
    Object.assign(this, {
      runtime,
      device,
      gpu,
      fn,
      weights,
      tokenizer,
      labels,
      limits,
      uncaptured,
      lifecycle,
      modelId: modelId ?? 'mapika-0.8b-f16'
    });
    this.queue = Promise.resolve();
    this.closed = false;
    this.disposal = null;
  }

  async decide(request: unknown, options: DecisionOptions | null = {}) {
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    const signal = options?.signal;
    signal?.throwIfAborted();
    validateDecisionFields(options);
    request = validateRequestFields(request, Infinity, Infinity, Infinity, MAPIKA_MAX_OPTIONS);
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    const packed = packMapikaRequest(
      this.tokenizer,
      this.labels!,
      request as Parameters<typeof packMapikaRequest>[2],
      this.limits
    );
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
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

  async systemOne(state: unknown, questions: unknown, options: DecisionOptions | null = {}) {
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    const signal = options?.signal;
    signal?.throwIfAborted();
    validateDecisionFields(options);
    const prepared = prepareMapikaSystemOne(state, questions, this.tokenizer, this.labels, this.limits);
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');

    const schedule = () => {
      if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
      const task = this.queue.then(async () => {
        const results = [];
        for (const row of prepared.rows) {
          signal?.throwIfAborted();
          results.push(await this.run(row.packed));
        }
        return finishMapikaSystemOne(prepared, results, this.modelId);
      });
      this.queue = task.catch(() => {});
      return task;
    };
    return signal ? waitForAbort(schedule, signal) : schedule();
  }

  async run(packed: PackedMapikaRequest) {
    return withGpuScope(this, 'Mapika inference failed.', async () => {
      const {ids, positions, mask, valid} = createCausalInputs(this, packed.ids);
      const output = this.fn!(ids, positions, mask, valid, ...this.weights);
      const host = this.runtime!.empty(
        [1, packed.ids.length, MAPIKA_MAX_OPTIONS],
        'float32',
        this.runtime!.cpu()
      ).copyFrom(output);
      await this.device!.sync();
      checkGpuDevice(this);
      const values = host.toArray() as {length: number; slice(start: number, end: number): Float32Array};
      if (values.length !== packed.ids.length * MAPIKA_MAX_OPTIONS) throw new Error('Unexpected Mapika output length.');
      const rows = packed.groups.map(({slot}) =>
        values.slice(slot * MAPIKA_MAX_OPTIONS, (slot + 1) * MAPIKA_MAX_OPTIONS)
      );
      return unpackMapikaScores(rows, packed, this.limits.temperature);
    });
  }

  async releaseCachedMemory() {
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    const task = this.queue.then(() =>
      withGpuScope(this, 'Could not release Mapika cached memory.', async () => {
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
        this.runtime = this.device = this.gpu = this.fn = this.tokenizer = this.labels = this.lifecycle = null;
        this.uncaptured = [];
      }
    })();
    return this.disposal;
  }
}
