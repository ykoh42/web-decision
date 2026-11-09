import {H2oTvmRunner} from '../h2o/runner.mjs';
import {WebDecisionError} from '../kev.js';
import {withStrandsScope} from '../strands/scope.mjs';
import {graphInputs, validateGraphInputs} from './graph-inputs.mjs';
import type {ElephantGraphInput} from './graph-inputs.mjs';
import type {ElephantPackedRequest} from './protocol.mjs';

interface ElephantTvmTensor {
  readonly dtype: string;
  readonly shape: number[];
  readonly byteOffset: number;
  copyFrom(data: unknown): unknown;
  getDataPtr(): number;
  toArray(): unknown;
}

interface ElephantWebGpuContext {
  copyRawBytesToBuffer?(bytes: Uint8Array, pointer: number, offset: number, length: number): void;
}

interface ElephantTvmRuntime {
  readonly lib?: {readonly webGPUContext?: ElephantWebGpuContext};
  empty(shape: number[], dtype: string, device: unknown): ElephantTvmTensor;
  cpu(): unknown;
}

interface ElephantRunnerArgs extends Record<string, unknown> {
  readonly weights: unknown[];
  readonly gpu: unknown;
  readonly lifecycle: unknown;
  readonly disposalGuard: unknown;
  readonly uncaptured: unknown[];
  readonly teardownObserver: unknown;
  readonly teardownTimeoutMs: number;
  readonly runtime: ElephantTvmRuntime;
  readonly device: unknown;
  readonly fn: (...args: unknown[]) => ElephantTvmTensor | null | undefined;
  readonly maxBranchTokens?: number;
}

function inputTensor(runner: ElephantTvmRunner, input: ElephantGraphInput): unknown {
  const tensor = runner.runtime.empty(input.shape, input.dtype, runner.device);
  if (input.dtype === 'int32') return tensor.copyFrom(input.data);
  const bytes = new Uint8Array(Math.ceil(input.data.byteLength / 4) * 4);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < input.data.length; i++) view.setUint16(i * 2, input.data[i]!, true);
  const context = runner.runtime.lib?.webGPUContext;
  if (typeof context?.copyRawBytesToBuffer !== 'function' || typeof tensor.byteOffset !== 'number')
    throw new Error('Pinned TVM aligned upload API unavailable.');
  context.copyRawBytesToBuffer(bytes, tensor.getDataPtr(), tensor.byteOffset, bytes.byteLength);
  return tensor;
}

interface ElephantReadbackLease {
  prepareReadback(): void;
  captureReadback(): void;
  sync(): Promise<void>;
}

export class ElephantTvmRunner extends H2oTvmRunner {
  declare runtime: ElephantTvmRuntime;
  declare device: unknown;
  declare fn: (...args: unknown[]) => ElephantTvmTensor | null | undefined;
  declare weights: unknown[];
  declare exclusive: <T>(operation: () => Promise<T>) => Promise<T>;

  declare private readonly maxBranchTokens: number;

  constructor(args: ElephantRunnerArgs) {
    const {weights, maxBranchTokens = 128} = args;
    if (!Array.isArray(weights) || weights.length !== 684 || !weights[682] || !weights[683])
      throw new TypeError('Elephant requires 684 ordered tensors.');
    if (![128, 512].includes(maxBranchTokens)) throw new TypeError('Invalid Elephant branch limit.');
    super({...args, weights: weights.slice(0, 682)});
    this.weights.push(weights[682], weights[683]);
    this.maxBranchTokens = maxBranchTokens;
  }

  override async run(packed: ElephantPackedRequest, {signal}: {signal?: AbortSignal} = {}): Promise<Float32Array[]> {
    const prefixLength = packed.prefixIds.length;
    let branches: Array<{ids: number[]; optionCount: number}>;
    try {
      branches = packed.branches.map((branch) => {
        const ids = validateGraphInputs(branch.ids, prefixLength, this.maxBranchTokens);
        return {ids, optionCount: branch.options.length};
      });
    } catch (cause) {
      throw new WebDecisionError('INFERENCE_ERROR', 'Invalid Elephant graph input.', {cause});
    }
    return this.exclusive(async () => {
      const rows: Float32Array[] = [];
      for (const branch of branches) {
        signal?.throwIfAborted();
        const inputs = graphInputs(branch.ids, prefixLength, this.maxBranchTokens);
        const scores = await withStrandsScope(
          this,
          'Elephant inference failed.',
          async (lease: ElephantReadbackLease) => {
            const output = this.fn(...inputs.map((input) => inputTensor(this, input)), ...this.weights);
            if (
              output?.dtype !== 'float32' ||
              output.shape?.length !== 2 ||
              output.shape[0] !== 1 ||
              output.shape[1] !== 256
            )
              throw new Error('Unexpected Elephant graph output.');
            const host = this.runtime.empty([1, 256], 'float32', this.runtime.cpu());
            lease.prepareReadback();
            try {
              host.copyFrom(output);
            } finally {
              lease.captureReadback();
            }
            await lease.sync();
            const data = host.toArray();
            if (!(data instanceof Float32Array) || data.length !== 256 || !data.every(Number.isFinite))
              throw new Error('Invalid Elephant score readback.');
            return data.slice(0, branch.optionCount);
          }
        );
        rows.push(scores);
      }
      return rows;
    });
  }
}
