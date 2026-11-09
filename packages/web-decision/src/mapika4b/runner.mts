import {H2oTvmRunner} from '../h2o/runner.mjs';
import {WebDecisionError} from '../kev.js';
import {withStrandsScope} from '../strands/scope.mjs';
import {MAPIKA4_MAX_OPTIONS} from './contract.mjs';

interface PackedInputSource {
  readonly ids?: unknown;
  readonly groups?: unknown;
}

interface MapikaGroup {
  readonly slot?: unknown;
}

type GraphInput =
  | {readonly dtype: 'int32'; readonly shape: number[]; readonly data: Int32Array}
  | {readonly dtype: 'float16'; readonly shape: number[]; readonly data: Uint16Array};

type GraphInputs = [GraphInput, GraphInput, GraphInput, GraphInput, GraphInput];

interface GraphOutput {
  readonly dtype?: unknown;
  readonly shape?: readonly unknown[];
}

interface TensorLike {
  readonly byteOffset?: unknown;
  copyFrom(data: unknown): TensorLike;
  getDataPtr(): unknown;
  toArray(): unknown;
}

interface WebGpuContext {
  copyRawBytesToBuffer?:
    | ((data: Uint8Array, pointer: unknown, byteOffset: number, byteLength: number) => unknown)
    | undefined;
}

interface MapikaRuntime {
  readonly lib?: {readonly webGPUContext?: WebGpuContext | null | undefined} | null | undefined;
  empty(shape: number[], dtype: string, device: unknown): TensorLike;
  cpu(): unknown;
}

interface MapikaRunnerContext {
  readonly runtime: MapikaRuntime | null;
  readonly device: unknown;
}

interface ReadbackLease {
  prepareReadback(): void;
  captureReadback(): void;
  sync(): PromiseLike<void>;
}

/** Create unpadded, causal graph inputs with bounded int32 indices. */
function createMapika4Inputs(packed: unknown): GraphInputs {
  const invalid = () => new WebDecisionError('INVALID_INPUT', 'Invalid Mapika 4B input.');
  if (
    !Array.isArray((packed as PackedInputSource | null | undefined)?.ids) ||
    !Array.isArray((packed as PackedInputSource | null | undefined)?.groups)
  )
    throw invalid();
  const ids: unknown[] = Array.from((packed as PackedInputSource).ids as ArrayLike<unknown>),
    groups: unknown[] = Array.from((packed as PackedInputSource).groups as ArrayLike<unknown>);
  const n = ids.length;
  const slots = groups.map((group) => (group as MapikaGroup | null | undefined)?.slot);
  if (
    n < 2 ||
    n > 128 ||
    slots.length < 1 ||
    slots.length > 255 ||
    ids.some((id) => !Number.isSafeInteger(id) || (id as number) < 0 || (id as number) >= 248320) ||
    slots.some(
      (slot, index) =>
        !Number.isSafeInteger(slot) ||
        (slot as number) < 0 ||
        (slot as number) >= n ||
        (index > 0 && (slots[index - 1] as number) >= (slot as number))
    )
  )
    throw invalid();
  const mask = new Uint16Array(n * n);
  for (let row = 0; row < n; row++) for (let column = row + 1; column < n; column++) mask[row * n + column] = 0xfbff;
  return [
    {dtype: 'int32', shape: [1, n], data: Int32Array.from(ids as number[])},
    {dtype: 'int32', shape: [1, n], data: Int32Array.from({length: n}, (_, index) => index)},
    {dtype: 'float16', shape: [1, 1, n, n], data: mask},
    {dtype: 'float16', shape: [1, n], data: new Uint16Array(n).fill(0x3c00)},
    {dtype: 'int32', shape: [slots.length], data: Int32Array.from(slots as number[])}
  ] as GraphInputs;
}

function inputTensor(runner: MapikaRunnerContext, input: GraphInput): TensorLike {
  const tensor = runner.runtime!.empty(input.shape, input.dtype, runner.device);
  if (input.dtype === 'int32') return tensor.copyFrom(input.data);
  const bytes = new Uint8Array(Math.ceil(input.data.byteLength / 4) * 4);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < input.data.length; i++) view.setUint16(i * 2, input.data[i]!, true);
  const context = runner.runtime!.lib?.webGPUContext;
  if (typeof context?.copyRawBytesToBuffer !== 'function' || typeof tensor.byteOffset !== 'number')
    throw new Error('Pinned TVM aligned upload API unavailable.');
  context.copyRawBytesToBuffer!(bytes, tensor.getDataPtr(), tensor.byteOffset, bytes.byteLength);
  return tensor;
}

export class Mapika4TvmRunner extends H2oTvmRunner {
  declare runtime: MapikaRuntime | null;
  declare device: unknown;
  declare fn: ((...args: unknown[]) => unknown) | null;
  declare weights: unknown[];
  declare exclusive: <Value>(operation: () => Value | PromiseLike<Value>) => Promise<Awaited<Value>>;

  override async run(packed: unknown): Promise<Float32Array[]> {
    const inputs = createMapika4Inputs(packed);
    const count = inputs[4].data.length;
    return this.exclusive(() =>
      withStrandsScope(this, 'Mapika 4B inference failed.', async (lease: ReadbackLease) => {
        const output = this.fn!(...inputs.map((input) => inputTensor(this, input)), ...this.weights) as
          | GraphOutput
          | null
          | undefined;
        const shape = [1, count, MAPIKA4_MAX_OPTIONS];
        if (
          output?.dtype !== 'float32' ||
          !Array.isArray(output.shape) ||
          output.shape.length !== shape.length ||
          shape.some((value, index) => output.shape![index] !== value)
        )
          throw new Error('Unexpected Mapika 4B graph output.');
        const host = this.runtime!.empty(shape, 'float32', this.runtime!.cpu());
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
