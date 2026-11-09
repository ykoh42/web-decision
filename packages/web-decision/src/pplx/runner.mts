import {H2oTvmRunner} from '../h2o/runner.mjs';
import {withStrandsScope} from '../strands/scope.mjs';
import {createFlashGraphInputs} from './graph-inputs.mjs';
import type {FlashGraphInput, FlashTensor, FlashTvmRuntime} from './types.js';

interface FlashReadbackLease {
  prepareReadback(): void;
  captureReadback(): void;
  sync(): Promise<void>;
}

function inputTensor(runner: FlashTvmRunner, input: FlashGraphInput): unknown {
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

/** Owns the runtime and produces one 255-score row for each joint marker. */
export class FlashTvmRunner extends H2oTvmRunner {
  declare runtime: FlashTvmRuntime;
  declare device: unknown;
  declare fn: (...args: unknown[]) => FlashTensor;
  declare weights: unknown[];

  override async run(packed: unknown): Promise<Float32Array[]> {
    const inputs = createFlashGraphInputs(packed, 248062);
    const count = inputs[4]!.data.length;
    return this.exclusive(() =>
      withStrandsScope(this, 'Perplexity inference failed.', async (lease: FlashReadbackLease) => {
        const args = inputs.map((input) => inputTensor(this, input));
        const output = this.fn(...args, ...this.weights);
        const shape = [1, count, 255];
        if (
          output?.dtype !== 'float32' ||
          !Array.isArray(output.shape) ||
          output.shape.length !== shape.length ||
          shape.some((n, index) => output.shape[index] !== n)
        )
          throw new Error('Unexpected Perplexity graph output.');
        const host = this.runtime.empty(shape, 'float32', this.runtime.cpu());
        lease.prepareReadback();
        try {
          host.copyFrom(output);
        } finally {
          lease.captureReadback();
        }
        await lease.sync();
        const data = host.toArray();
        if (!(data instanceof Float32Array) || data.length !== count * 255 || !data.every(Number.isFinite))
          throw new Error('Invalid Perplexity score readback.');
        return Array.from({length: count}, (_, i) => data.slice(i * 255, (i + 1) * 255));
      })
    );
  }
}
