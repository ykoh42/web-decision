import {checkGpuDevice} from '../gpu-session-utils.js';
import {WebDecisionError} from '../kev.js';
import {assertUsable, withStrandsScope} from './scope.mjs';
import {STRANDS_GEMMA4_E2B_CONTEXT} from './gemma4-e2b-contract.mjs';

interface Gemma4Tensor {
  copyFrom(value: unknown): Gemma4Tensor;
  toArray(): unknown;
}

interface Gemma4Runtime {
  empty(shape: number[], dtype: string, device: unknown): Gemma4Tensor;
  cpu(): unknown;
  getGlobalFunc(name: string): () => unknown;
  endScope(): unknown;
  dispose(): unknown;
}

interface Gemma4Lifecycle {
  readonly loss: unknown;
  readonly disposed: boolean;
}

interface Gemma4DisposalGuard {
  readonly blocked?: boolean;
}

interface Gemma4RunnerOptions {
  readonly runtime: Gemma4Runtime;
  readonly device: unknown;
  readonly gpu: unknown;
  readonly fn: (...args: unknown[]) => Gemma4Tensor;
  readonly weights: unknown[];
  readonly lifecycle: Gemma4Lifecycle;
  readonly uncaptured: string[];
  readonly disposalGuard?: Gemma4DisposalGuard | null;
}

interface Gemma4RowCandidate {
  readonly inputIds?: unknown;
  readonly options?: unknown;
  readonly optionIndices?: unknown;
  readonly answerIndex?: unknown;
  readonly type?: unknown;
  readonly descriptions?: unknown;
}

interface Gemma4GraphInput {
  readonly inputIds: number[];
  readonly optionIndices: number[];
  readonly answerIndex: number;
  readonly optionCount: number;
}

interface Gemma4ReadbackLease {
  prepareReadback(): void;
  captureReadback(): void;
  sync(): PromiseLike<void>;
  syncWithoutReadback(): PromiseLike<void>;
}

function isWhitespaceCodePoint(codePoint: number): boolean {
  return (
    (codePoint >= 0x9 && codePoint <= 0xd) ||
    (codePoint >= 0x1c && codePoint <= 0x20) ||
    codePoint === 0x85 ||
    codePoint === 0xa0 ||
    codePoint === 0x1680 ||
    (codePoint >= 0x2000 && codePoint <= 0x200a) ||
    codePoint === 0x2028 ||
    codePoint === 0x2029 ||
    codePoint === 0x202f ||
    codePoint === 0x205f ||
    codePoint === 0x3000
  );
}

function isWhitespaceOnly(value: string): boolean {
  for (const character of value) {
    if (!isWhitespaceCodePoint(character.codePointAt(0) as number)) return false;
  }
  return true;
}

function snapshotRow(row: unknown): Gemma4GraphInput {
  const inputIds = (row as Gemma4RowCandidate | null | undefined)?.inputIds;
  const options = (row as Gemma4RowCandidate | null | undefined)?.options;
  const optionIndices = (row as Gemma4RowCandidate | null | undefined)?.optionIndices;
  if (
    !Array.isArray(inputIds) ||
    inputIds.length < 2 ||
    inputIds.length > STRANDS_GEMMA4_E2B_CONTEXT ||
    !Array.isArray(options) ||
    !Array.isArray(optionIndices) ||
    options.length < 2 ||
    options.length !== optionIndices.length ||
    (row as Gemma4RowCandidate).answerIndex !== (inputIds as unknown[]).length - 1 ||
    !['choice', 'score', 'noul'].includes((row as Gemma4RowCandidate).type as string) ||
    ((row as Gemma4RowCandidate).type === 'score' &&
      (options.length > 10 ||
        !Array.isArray((row as Gemma4RowCandidate).descriptions) ||
        ((row as Gemma4RowCandidate).descriptions as unknown[]).length !== options.length)) ||
    ((row as Gemma4RowCandidate).type === 'noul' && options.length !== 2)
  )
    throw new WebDecisionError('INVALID_INPUT', 'Invalid Strands Gemma 4 graph input.');
  const ids: number[] = [],
    positions: number[] = [],
    labels = new Set<string>();
  for (let index = 0; index < (inputIds as unknown[]).length; index++) {
    const id = (inputIds as unknown[])[index] as number;
    if (!Object.hasOwn(inputIds, index) || !Number.isSafeInteger(id) || id < 0 || id >= 262144)
      throw new WebDecisionError('INVALID_INPUT', 'Invalid Strands Gemma 4 token IDs.');
    ids.push(id);
  }
  for (let index = 0; index < options.length; index++) {
    const label = (options as unknown[])[index];
    const position = (optionIndices as unknown[])[index] as number;
    if (
      !Object.hasOwn(options, index) ||
      !Object.hasOwn(optionIndices, index) ||
      typeof label !== 'string' ||
      isWhitespaceOnly(label) ||
      labels.has(label) ||
      !Number.isSafeInteger(position) ||
      position < 0 ||
      position >= ((row as Gemma4RowCandidate).answerIndex as number) ||
      (index > 0 && position <= positions[index - 1]!)
    )
      throw new WebDecisionError('INVALID_INPUT', 'Invalid Strands Gemma 4 option positions.');
    labels.add(label);
    positions.push(position);
  }
  return {
    inputIds: ids,
    optionIndices: positions,
    answerIndex: (row as Gemma4RowCandidate).answerIndex as number,
    optionCount: options.length
  };
}

function isGpuOutOfMemory(error: unknown): boolean {
  let cause = error;
  for (let depth = 0; depth < 4 && cause; depth++, cause = (cause as {readonly cause?: unknown}).cause)
    if ((cause as {readonly name?: unknown}).name === 'GPUOutOfMemoryError') return true;
  return false;
}

/** Owns the v26 VM, all 1,248 graph values, and one WebGPU device. */
export class StrandsGemma4Runner {
  declare runtime: Gemma4Runtime | null;
  declare device: unknown;
  declare gpu: unknown;
  declare fn: ((...args: unknown[]) => Gemma4Tensor) | null;
  declare weights: unknown[];
  declare lifecycle: Gemma4Lifecycle | null;
  declare uncaptured: string[];
  declare disposalGuard: Gemma4DisposalGuard | null | undefined;

  constructor({runtime, device, gpu, fn, weights, lifecycle, uncaptured, disposalGuard}: Gemma4RunnerOptions) {
    if (
      !runtime ||
      !device ||
      !gpu ||
      typeof fn !== 'function' ||
      !Array.isArray(weights) ||
      weights.length !== 1248 ||
      weights.some((tensor) => !tensor) ||
      !lifecycle ||
      !Array.isArray(uncaptured)
    )
      throw new TypeError('Invalid Strands Gemma 4 runtime ownership.');
    Object.assign(this, {runtime, device, gpu, fn, weights, lifecycle, uncaptured, disposalGuard});
  }

  async run(row: unknown): Promise<Float32Array> {
    const input = snapshotRow(row);
    try {
      return await withStrandsScope(this, 'Strands Gemma 4 inference failed.', async (lease: Gemma4ReadbackLease) => {
        const ids = this.runtime!.empty([1, input.inputIds.length], 'int32', this.device).copyFrom(
          Int32Array.from(input.inputIds)
        );
        const answer = this.runtime!.empty([1], 'int32', this.device).copyFrom(Int32Array.of(input.answerIndex));
        const options = this.runtime!.empty([input.optionIndices.length], 'int32', this.device).copyFrom(
          Int32Array.from(input.optionIndices)
        );
        const output = this.fn!(ids, answer, options, ...this.weights);
        const host = this.runtime!.empty([1, input.optionCount], 'float32', this.runtime!.cpu());
        lease.prepareReadback();
        try {
          host.copyFrom(output);
        } finally {
          lease.captureReadback();
        }
        await lease.sync();
        checkGpuDevice(this);
        return host.toArray() as Float32Array;
      });
    } catch (cause) {
      if (isGpuOutOfMemory(cause))
        throw new WebDecisionError(
          'DEVICE_RESOURCE_LIMIT',
          'This WebGPU device ran out of resources for the Gemma 4 request; no input was truncated.',
          {cause}
        );
      throw cause;
    }
  }

  async releaseCachedMemory(): Promise<void> {
    return withStrandsScope(this, 'Could not release Gemma 4 cached memory.', async (lease: Gemma4ReadbackLease) => {
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
