import {checkGpuDevice, createCausalInputs} from '../gpu-session-utils.js';
import {assertUsable, withStrandsScope} from './scope.mjs';
import {validateStrandsRow} from './choice.mjs';

interface TensorLike {
  copyFrom(value: unknown): TensorLike;
  toArray(): unknown;
}

interface RuntimeLike {
  empty(shape: number[], dtype: string, device: unknown): TensorLike;
  cpu(): unknown;
  getGlobalFunc(name: string): () => unknown;
  endScope(): void;
  dispose(): unknown;
}

interface DeviceLike {
  sync(): MaybePromise<unknown>;
}

interface GpuLike {
  pushErrorScope(filter: 'out-of-memory' | 'validation'): void;
  popErrorScope(): PromiseLike<unknown>;
}

interface LifecycleState {
  loss: unknown;
  disposed: boolean;
}

interface DisposalGuard {
  readonly blocked: boolean;
  retain(): void;
}

interface RunnerOptions {
  readonly runtime: RuntimeLike;
  readonly device: DeviceLike;
  readonly gpu: GpuLike;
  readonly fn: (...args: unknown[]) => unknown;
  readonly weights: unknown[];
  readonly lifecycle: LifecycleState;
  readonly uncaptured: string[];
  readonly disposalGuard?: DisposalGuard | null | undefined;
}

type MaybePromise<Value> = Value | PromiseLike<Value>;

interface ValidatedStrandsRow {
  readonly inputIds: readonly number[];
  readonly answerIndex: number;
  readonly optionIndices: readonly number[];
  readonly options: readonly string[];
}

interface ReadbackLease {
  prepareReadback(): void;
  captureReadback(): void;
  sync(): PromiseLike<void>;
  syncWithoutReadback(): PromiseLike<void>;
}

/** One owned TVM runtime/device and its already verified fused parameter list. */
export class ExperimentalFusedRunner {
  declare runtime: RuntimeLike | null;
  declare device: DeviceLike | null;
  declare gpu: GpuLike | null;
  declare fn: ((...args: unknown[]) => unknown) | null;
  declare weights: unknown[];
  declare lifecycle: LifecycleState | null;
  declare uncaptured: string[];
  declare disposalGuard: DisposalGuard | null | undefined;
  declare poisoned?: boolean;

  constructor({runtime, device, gpu, fn, weights, lifecycle, uncaptured, disposalGuard}: RunnerOptions) {
    if (
      !runtime ||
      !device ||
      !gpu ||
      typeof fn !== 'function' ||
      !Array.isArray(weights) ||
      weights.length !== 328 ||
      !lifecycle ||
      !Array.isArray(uncaptured)
    )
      throw new TypeError('Invalid fused runtime ownership');
    Object.assign(this, {runtime, device, gpu, fn, weights, lifecycle, uncaptured, disposalGuard});
  }

  async run(row: unknown): Promise<unknown> {
    const snapshot: ValidatedStrandsRow = validateStrandsRow(row);
    return withStrandsScope(this, 'Strands inference failed.', async (lease: ReadbackLease) => {
      const {
        ids,
        positions,
        mask,
        valid
      }: {ids: TensorLike; positions: TensorLike; mask: TensorLike; valid: TensorLike} = createCausalInputs(
        this,
        snapshot.inputIds
      );
      const answer: TensorLike = this.runtime!.empty([1], 'int32', this.device!).copyFrom(
        Int32Array.of(snapshot.answerIndex)
      );
      const options: TensorLike = this.runtime!.empty([snapshot.optionIndices.length], 'int32', this.device!).copyFrom(
        Int32Array.from(snapshot.optionIndices)
      );
      const output = this.fn!(ids, positions, mask, valid, answer, options, ...this.weights);
      const host: TensorLike = this.runtime!.empty([1, snapshot.options.length], 'float32', this.runtime!.cpu());
      lease.prepareReadback();
      try {
        host.copyFrom(output);
      } finally {
        lease.captureReadback();
      }
      await lease.sync();
      checkGpuDevice(this);
      return host.toArray();
    });
  }

  async releaseCachedMemory(): Promise<void> {
    return withStrandsScope(this, 'Could not release Strands cached memory.', async (lease: ReadbackLease) => {
      this.runtime!.getGlobalFunc('vm.builtin.memory_manager.clear')();
      await lease.syncWithoutReadback();
    });
  }

  async dispose(): Promise<void> {
    assertUsable(this);
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
      this.runtime = this.device = this.gpu = this.fn = this.lifecycle = null;
      this.uncaptured = [];
    }
  }
}
