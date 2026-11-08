import {checkGpuDevice} from '../gpu-session-utils.js';
import {WebDecisionError} from '../kev.js';
import {assertUsable, withStrandsScope} from './scope.mjs';
import {STRANDS_GEMMA4_E2B_CONTEXT} from './gemma4-e2b-contract.mjs';

function isWhitespaceCodePoint(codePoint) {
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

function isWhitespaceOnly(value) {
  for (const character of value) {
    if (!isWhitespaceCodePoint(character.codePointAt(0))) return false;
  }
  return true;
}

function snapshotRow(row) {
  const inputIds = row?.inputIds;
  const options = row?.options;
  const optionIndices = row?.optionIndices;
  if (
    !Array.isArray(inputIds) ||
    inputIds.length < 2 ||
    inputIds.length > STRANDS_GEMMA4_E2B_CONTEXT ||
    !Array.isArray(options) ||
    !Array.isArray(optionIndices) ||
    options.length < 2 ||
    options.length !== optionIndices.length ||
    row.answerIndex !== inputIds.length - 1 ||
    !['choice', 'score', 'noul'].includes(row.type) ||
    (row.type === 'score' &&
      (options.length > 10 || !Array.isArray(row.descriptions) || row.descriptions.length !== options.length)) ||
    (row.type === 'noul' && options.length !== 2)
  )
    throw new WebDecisionError('INVALID_INPUT', 'Invalid Strands Gemma 4 graph input.');
  const ids = [],
    positions = [],
    labels = new Set();
  for (let index = 0; index < inputIds.length; index++) {
    const id = inputIds[index];
    if (!Object.hasOwn(inputIds, index) || !Number.isSafeInteger(id) || id < 0 || id >= 262144)
      throw new WebDecisionError('INVALID_INPUT', 'Invalid Strands Gemma 4 token IDs.');
    ids.push(id);
  }
  for (let index = 0; index < options.length; index++) {
    const label = options[index];
    const position = optionIndices[index];
    if (
      !Object.hasOwn(options, index) ||
      !Object.hasOwn(optionIndices, index) ||
      typeof label !== 'string' ||
      isWhitespaceOnly(label) ||
      labels.has(label) ||
      !Number.isSafeInteger(position) ||
      position < 0 ||
      position >= row.answerIndex ||
      (index > 0 && position <= positions[index - 1])
    )
      throw new WebDecisionError('INVALID_INPUT', 'Invalid Strands Gemma 4 option positions.');
    labels.add(label);
    positions.push(position);
  }
  return {inputIds: ids, optionIndices: positions, answerIndex: row.answerIndex, optionCount: options.length};
}

function isGpuOutOfMemory(error) {
  let cause = error;
  for (let depth = 0; depth < 4 && cause; depth++, cause = cause.cause)
    if (cause.name === 'GPUOutOfMemoryError') return true;
  return false;
}

/** Owns the v26 VM, all 1,248 graph values, and one WebGPU device. */
export class StrandsGemma4Runner {
  constructor({runtime, device, gpu, fn, weights, lifecycle, uncaptured, disposalGuard}) {
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

  async run(row) {
    const input = snapshotRow(row);
    try {
      return await withStrandsScope(this, 'Strands Gemma 4 inference failed.', async (lease) => {
        const ids = this.runtime
          .empty([1, input.inputIds.length], 'int32', this.device)
          .copyFrom(Int32Array.from(input.inputIds));
        const answer = this.runtime.empty([1], 'int32', this.device).copyFrom(Int32Array.of(input.answerIndex));
        const options = this.runtime
          .empty([input.optionIndices.length], 'int32', this.device)
          .copyFrom(Int32Array.from(input.optionIndices));
        const output = this.fn(ids, answer, options, ...this.weights);
        const host = this.runtime.empty([1, input.optionCount], 'float32', this.runtime.cpu());
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

  async releaseCachedMemory() {
    return withStrandsScope(this, 'Could not release Gemma 4 cached memory.', async (lease) => {
      this.runtime.getGlobalFunc('vm.builtin.memory_manager.clear')();
      await lease.syncWithoutReadback();
    });
  }

  async dispose() {
    assertUsable(this);
    try {
      if (!this.lifecycle.disposed) {
        try {
          if (!this.lifecycle.loss) this.runtime.endScope();
        } finally {
          this.runtime.dispose();
        }
      }
    } finally {
      this.weights = [];
      this.runtime = this.device = this.gpu = this.fn = this.lifecycle = null;
      this.uncaptured = [];
    }
  }
}
