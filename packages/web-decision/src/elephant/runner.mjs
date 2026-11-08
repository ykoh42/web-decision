import {H2oTvmRunner} from '../h2o/runner.mjs';
import {WebDecisionError} from '../kev.js';
import {withStrandsScope} from '../strands/scope.mjs';
import {graphInputs, validateGraphInputs} from './graph-inputs.mjs';

function inputTensor(runner, input) {
  const tensor = runner.runtime.empty(input.shape, input.dtype, runner.device);
  if (input.dtype === 'int32') return tensor.copyFrom(input.data);
  const bytes = new Uint8Array(Math.ceil(input.data.byteLength / 4) * 4);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < input.data.length; i++) view.setUint16(i * 2, input.data[i], true);
  const context = runner.runtime.lib?.webGPUContext;
  if (typeof context?.copyRawBytesToBuffer !== 'function' || typeof tensor.byteOffset !== 'number')
    throw new Error('Pinned TVM aligned upload API unavailable.');
  context.copyRawBytesToBuffer(bytes, tensor.getDataPtr(), tensor.byteOffset, bytes.byteLength);
  return tensor;
}

export class ElephantTvmRunner extends H2oTvmRunner {
  constructor(args) {
    const {weights, maxBranchTokens = 128} = args;
    if (!Array.isArray(weights) || weights.length !== 684 || !weights[682] || !weights[683])
      throw new TypeError('Elephant requires 684 ordered tensors.');
    if (![128, 512].includes(maxBranchTokens)) throw new TypeError('Invalid Elephant branch limit.');
    super({...args, weights: weights.slice(0, 682)});
    this.weights.push(weights[682], weights[683]);
    this.maxBranchTokens = maxBranchTokens;
  }

  async run(packed, {signal} = {}) {
    const prefixLength = packed.prefixIds.length;
    let branches;
    try {
      branches = packed.branches.map((branch) => {
        const ids = validateGraphInputs(branch.ids, prefixLength, this.maxBranchTokens);
        return {ids, optionCount: branch.options.length};
      });
    } catch (cause) {
      throw new WebDecisionError('INFERENCE_ERROR', 'Invalid Elephant graph input.', {cause});
    }
    return this.exclusive(async () => {
      const rows = [];
      for (const branch of branches) {
        signal?.throwIfAborted();
        const inputs = graphInputs(branch.ids, prefixLength, this.maxBranchTokens);
        const scores = await withStrandsScope(this, 'Elephant inference failed.', async (lease) => {
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
        });
        rows.push(scores);
      }
      return rows;
    });
  }
}
