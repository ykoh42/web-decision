import {withGpuScope} from '../gpu-session-utils.js';
import {readAssetBatch} from '../asset-batch.js';
import type {DLDevice, Instance, PackedFunc, Tensor} from '../runtime/tvm-types.js';

const hiddenSize = 2048;
const letterCount = 255;
const packedLetterRows = 256;
const decoderParameterCount = 319;
const prefetchCount = 4;

export interface Mapika2bTextBinding {
  readonly name: string;
  readonly shape: readonly number[];
}

export interface Mapika2bTextRow {
  readonly embeddings: Float32Array;
  readonly positionIds: Int32Array;
  readonly slotIndices: Int32Array;
}

export interface Mapika2bTextRuntime {
  readonly runtime: Instance;
  readonly device: DLDevice;
  readonly gpu: GPUDevice;
  readonly main: PackedFunc;
  readonly parameters: readonly Tensor[];
  readonly lifecycle: {readonly loss: unknown; readonly disposed: boolean};
  readonly uncaptured: readonly string[];
}

function rawBytes(value: Uint16Array): Uint8Array {
  return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
}

export function widenMapika2bEmbeddingRows(source: Uint16Array): Float32Array<ArrayBuffer> {
  const output = new Float32Array(source.length);
  const words = new Uint32Array(output.buffer);
  for (let index = 0; index < source.length; index += 1) words[index] = source[index]! << 16;
  return output;
}

async function uploadPacked(
  runtime: Instance,
  device: DLDevice,
  shape: readonly number[],
  bytes: Uint8Array
): Promise<Tensor> {
  const wordCount = shape.reduce((product, dimension) => product * dimension, 1);
  if (bytes.byteLength !== wordCount * 4) throw new TypeError('Mapika text parameter size differs from its ABI.');
  const tensor = runtime.empty([...shape], 'int32', device).copyFromRawBytes(bytes);
  await device.sync();
  return tensor;
}

/** Upload the tied answer rows and native packed checkpoint tensors in VM argument order. */
export async function uploadMapika2bTextParameters(
  runtime: Instance,
  device: DLDevice,
  bindings: readonly Mapika2bTextBinding[],
  answerRows: Uint16Array,
  readParameter: (binding: Mapika2bTextBinding, signal: AbortSignal) => Promise<Uint8Array>,
  signal?: AbortSignal,
  onBytes?: (bytes: number) => void
): Promise<readonly Tensor[]> {
  if (bindings.length !== decoderParameterCount || answerRows.length !== letterCount * hiddenSize)
    throw new TypeError('Mapika text graph expects 319 decoder parameters and 255 tied answer rows.');
  const paddedAnswers = new Uint16Array(packedLetterRows * hiddenSize);
  paddedAnswers.set(answerRows);
  signal?.throwIfAborted();
  const parameters = [await uploadPacked(runtime, device, [packedLetterRows, hiddenSize / 2], rawBytes(paddedAnswers))];
  onBytes?.(answerRows.byteLength);
  for (let start = 0; start < bindings.length; start += prefetchCount) {
    signal?.throwIfAborted();
    const group = bindings.slice(start, start + prefetchCount);
    const names = group.map((_, index) => String(index));
    const payloads = await readAssetBatch(
      names,
      (index, batchSignal) => readParameter(group[Number(index)]!, batchSignal),
      signal
    );
    for (const [index, binding] of group.entries()) {
      signal?.throwIfAborted();
      const payload = payloads[index]!;
      parameters.push(await uploadPacked(runtime, device, binding.shape, payload));
      onBytes?.(payload.byteLength);
    }
  }
  signal?.throwIfAborted();
  return Object.freeze(parameters);
}

function validateRow({embeddings, positionIds, slotIndices}: Mapika2bTextRow): number {
  const length = embeddings.length / hiddenSize;
  if (
    !Number.isSafeInteger(length) ||
    length < 1 ||
    positionIds.length !== length * 4 ||
    slotIndices.length === 0 ||
    slotIndices.some((slot) => slot < 0 || slot >= length)
  )
    throw new TypeError('Mapika text row does not match the decoder input shapes.');
  return length;
}

/** Run one unpadded row so callers can schedule rows within device memory. */
export function runMapika2bTextRow(runner: Mapika2bTextRuntime, row: Mapika2bTextRow): Promise<Float32Array> {
  const length = validateRow(row);
  const slots = row.slotIndices.length;
  return withGpuScope(runner, 'Mapika 2B v12 text inference failed.', async () => {
    const {runtime, device, main, parameters} = runner;
    const embeddings = runtime.empty([1, length, hiddenSize], 'float32', device).copyFrom(row.embeddings);
    const mask = runtime.empty([1, length], 'int32', device).copyFrom(new Int32Array(length).fill(1));
    const positions = runtime.empty([4, 1, length], 'int32', device).copyFrom(row.positionIds);
    const batchIndices = runtime.empty([slots], 'int32', device).copyFrom(new Int32Array(slots));
    const slotIndices = runtime.empty([slots], 'int32', device).copyFrom(row.slotIndices);
    const output = main(embeddings, mask, positions, batchIndices, slotIndices, ...parameters) as Tensor;
    const host = runtime.empty([slots, letterCount], 'float32', runtime.cpu());
    host.copyFrom(output);
    await device.sync();
    const logits = host.toArray();
    if (!(logits instanceof Float32Array)) throw new TypeError('Mapika text graph returned an invalid readout.');
    return logits;
  });
}
