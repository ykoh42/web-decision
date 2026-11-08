import {H2oTvmRunner} from '../h2o/runner.mjs';
import {checkGpuDevice} from '../gpu-session-utils.js';
import {WebDecisionError} from '../kev.js';
import {withStrandsScope} from '../strands/scope.mjs';
import {buildChoiceInputs, splitChoiceLogits} from './graph-inputs.mjs';
import {selectClefRows} from './rows.mjs';

function checkRunnerDevice(runner) {
  try {
    checkGpuDevice(runner);
  } catch (cause) {
    throw new WebDecisionError('INFERENCE_ERROR', 'Clef GPU is unavailable.', {cause});
  }
}

function inputTensor(runner, input) {
  const tensor = runner.runtime.empty(input.shape, input.dtype, runner.device);
  if (input.dtype !== 'float16') return tensor.copyFrom(input.data);
  const bytes = new Uint8Array(Math.ceil((input.data.length * 2) / 4) * 4),
    view = new DataView(bytes.buffer);
  for (let i = 0; i < input.data.length; i++) view.setUint16(i * 2, input.data[i], true);
  const context = runner.runtime.lib?.webGPUContext;
  if (
    typeof context?.copyRawBytesToBuffer !== 'function' ||
    typeof context?.gpuBufferFromPtr !== 'function' ||
    typeof tensor?.getDataPtr !== 'function' ||
    !Number.isSafeInteger(tensor.byteOffset) ||
    tensor.byteOffset < 0 ||
    tensor.byteOffset % 4
  )
    throw new Error('Pinned TVM aligned upload API unavailable.');
  const buffer = context.gpuBufferFromPtr(tensor.getDataPtr());
  if (!Number.isSafeInteger(buffer?.size) || buffer.size < tensor.byteOffset + bytes.length)
    throw new Error('Clef input allocation is too small.');
  context.copyRawBytesToBuffer(bytes, tensor.getDataPtr(), tensor.byteOffset, bytes.length);
  return tensor;
}

function selectedInputs(graph, input, output) {
  const inputs = graph.inputs.slice();
  for (const [index, selected] of [
    [0, input],
    [4, output]
  ]) {
    if (
      !(selected.local instanceof Int32Array) ||
      !(selected.rowBanks instanceof Int32Array) ||
      !Array.isArray(selected.unique) ||
      !selected.unique.length ||
      selected.unique.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= 248320) ||
      new Set(selected.unique).size !== selected.unique.length ||
      selected.local.length !== inputs[index].data.length ||
      selected.local.some(
        (value, i) => value < 0 || value >= selected.unique.length || selected.unique[value] !== inputs[index].data[i]
      ) ||
      selected.rowBanks.length !== selected.unique.length ||
      selected.rowBanks.some((value, i) => value !== Math.floor(selected.unique[i] / 65536)) ||
      new Set(selected.local).size !== selected.unique.length
    )
      throw new Error('Invalid Clef local ID mapping.');
    inputs[index] = {...inputs[index], data: selected.local};
  }
  for (const [prefix, selected] of [
    ['input', input],
    ['output', output]
  ]) {
    const rows = selected.unique.length;
    if (
      !(selected.packed instanceof Uint32Array) ||
      !(selected.scales instanceof Uint16Array) ||
      selected.packed.length !== rows * 512 ||
      selected.scales.length !== rows * 128
    )
      throw new Error('Invalid Clef selected row shape.');
    inputs.push(
      {name: `${prefix}_banks`, dtype: 'int32', shape: [rows], data: selected.rowBanks},
      {name: `${prefix}_packed`, dtype: 'uint32', shape: [rows, 512], data: selected.packed},
      {name: `${prefix}_scales`, dtype: 'float16', shape: [rows, 128], data: selected.scales}
    );
  }
  return inputs;
}

export class ClefTvmRunner extends H2oTvmRunner {
  constructor(args) {
    const {weights, source} = args;
    if (
      !Array.isArray(weights) ||
      weights.length !== 797 ||
      Array.from({length: 797}, (_, i) => i).some((i) => !Object.hasOwn(weights, i) || !weights[i]) ||
      typeof source?.readRange !== 'function' ||
      !Array.isArray(source.inputBanks) ||
      !Array.isArray(source.outputBanks)
    )
      throw new TypeError('Clef requires its source and 797 ordered tensors.');
    super({...args, weights: weights.slice(0, 682)});
    this.weights.push(...weights.slice(682));
    this.source = source;
    this.selectRows = args.fake === true ? args.selectRows : selectClefRows;
    if (typeof this.selectRows !== 'function') throw new TypeError('Clef row selector is required.');
  }

  async run(packed, {signal} = {}) {
    let graph;
    try {
      graph = buildChoiceInputs(packed);
    } catch (cause) {
      throw new WebDecisionError('INFERENCE_ERROR', 'Invalid Clef graph input.', {cause});
    }
    return this.exclusive(async () => {
      signal?.throwIfAborted();
      checkRunnerDevice(this);
      const readRange = async (entry, part, rangeSignal) => {
        checkRunnerDevice(this);
        const value = await this.source.readRange(entry, part, rangeSignal);
        checkRunnerDevice(this);
        return value;
      };
      const input = await this.selectRows(graph.inputs[0].data, this.source.inputBanks, {
        readRange,
        signal
      });
      signal?.throwIfAborted();
      checkRunnerDevice(this);
      const output = await this.selectRows(graph.inputs[4].data, this.source.outputBanks, {
        readRange,
        signal
      });
      signal?.throwIfAborted();
      const inputs = selectedInputs(graph, input, output);
      return withStrandsScope(this, 'Clef inference failed.', async (lease) => {
        signal?.throwIfAborted();
        const result = this.fn(...inputs.map((item) => inputTensor(this, item)), ...this.weights);
        const count = graph.offsets.at(-1);
        if (
          result?.dtype !== 'float32' ||
          !Array.isArray(result.shape) ||
          result.shape.length !== 1 ||
          result.shape[0] !== count
        )
          throw new Error('Unexpected Clef graph output.');
        const host = this.runtime.empty([count], 'float32', this.runtime.cpu());
        lease.prepareReadback();
        try {
          host.copyFrom(result);
        } finally {
          lease.captureReadback();
        }
        await lease.sync();
        const values = host.toArray();
        if (!(values instanceof Float32Array) || values.length !== count || values.some((x) => !Number.isFinite(x)))
          throw new Error('Invalid Clef score readback.');
        return splitChoiceLogits(graph, values);
      });
    });
  }
}
