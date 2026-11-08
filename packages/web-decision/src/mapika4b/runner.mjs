import {H2oTvmRunner} from '../h2o/runner.mjs';
import {WebDecisionError} from '../kev.js';
import {withStrandsScope} from '../strands/scope.mjs';
import {MAPIKA4_MAX_OPTIONS} from './contract.mjs';

/** Create unpadded, causal graph inputs with bounded int32 indices. */
function createMapika4Inputs(packed) {
  const invalid = () => new WebDecisionError('INVALID_INPUT', 'Invalid Mapika 4B input.');
  if (!Array.isArray(packed?.ids) || !Array.isArray(packed?.groups)) throw invalid();
  const ids = Array.from(packed.ids),
    groups = Array.from(packed.groups);
  const n = ids.length,
    slots = groups.map((group) => group?.slot);
  if (
    n < 2 ||
    n > 128 ||
    slots.length < 1 ||
    slots.length > 255 ||
    ids.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= 248320) ||
    slots.some(
      (slot, index) => !Number.isSafeInteger(slot) || slot < 0 || slot >= n || (index > 0 && slots[index - 1] >= slot)
    )
  )
    throw invalid();
  const mask = new Uint16Array(n * n);
  for (let row = 0; row < n; row++) for (let column = row + 1; column < n; column++) mask[row * n + column] = 0xfbff;
  return [
    {dtype: 'int32', shape: [1, n], data: Int32Array.from(ids)},
    {dtype: 'int32', shape: [1, n], data: Int32Array.from({length: n}, (_, index) => index)},
    {dtype: 'float16', shape: [1, 1, n, n], data: mask},
    {dtype: 'float16', shape: [1, n], data: new Uint16Array(n).fill(0x3c00)},
    {dtype: 'int32', shape: [slots.length], data: Int32Array.from(slots)}
  ];
}

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

export class Mapika4TvmRunner extends H2oTvmRunner {
  async run(packed) {
    const inputs = createMapika4Inputs(packed);
    const count = inputs[4].data.length;
    return this.exclusive(() =>
      withStrandsScope(this, 'Mapika 4B inference failed.', async (lease) => {
        const output = this.fn(...inputs.map((input) => inputTensor(this, input)), ...this.weights);
        const shape = [1, count, MAPIKA4_MAX_OPTIONS];
        if (
          output?.dtype !== 'float32' ||
          !Array.isArray(output.shape) ||
          output.shape.length !== shape.length ||
          shape.some((value, index) => output.shape[index] !== value)
        )
          throw new Error('Unexpected Mapika 4B graph output.');
        const host = this.runtime.empty(shape, 'float32', this.runtime.cpu());
        lease.prepareReadback();
        try {
          host.copyFrom(output);
        } finally {
          lease.captureReadback();
        }
        await lease.sync();
        const scores = host.toArray();
        if (
          !(scores instanceof Float32Array) ||
          scores.length !== count * MAPIKA4_MAX_OPTIONS ||
          !scores.every(Number.isFinite)
        )
          throw new Error('Invalid Mapika 4B score readback.');
        return Array.from({length: count}, (_, index) =>
          scores.slice(index * MAPIKA4_MAX_OPTIONS, (index + 1) * MAPIKA4_MAX_OPTIONS)
        );
      })
    );
  }
}
