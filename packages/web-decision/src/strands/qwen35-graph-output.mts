/**
 * Read the experimental R6/R28 graph ABI output tuple (logits, probabilities).
 * The probabilities are already temperature-scaled and n_slots-masked.
 */

import {QWEN35_MAX_CHOICE_OPTIONS} from './qwen35-protocol.mts';

const TVMJS_CPU_DEVICE_TYPE = 1;
const MIN_GRAPH_OPTION_WIDTH = 2;

type TvmJsTensor = Readonly<{
  dtype: unknown;
  shape: unknown;
  device?: unknown;
  toArray?: unknown;
}>;

type TvmJsTuple = Readonly<{
  size: () => unknown;
  get: (index: number) => unknown;
}>;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isFloat32Array(value: unknown): value is Float32Array {
  return ArrayBuffer.isView(value) && Object.prototype.toString.call(value) === '[object Float32Array]';
}

function isTvmJsTuple(value: unknown): value is TvmJsTuple {
  return isRecord(value) && typeof value.size === 'function' && typeof value.get === 'function';
}

function graphTupleValues(output: unknown): readonly unknown[] {
  if (Array.isArray(output)) return output;
  if (!isTvmJsTuple(output))
    throw new TypeError('Qwen graph output must be the two-tensor (logits, probabilities) tuple.');
  const size = output.size();
  if (!Number.isSafeInteger(size) || size !== 2)
    throw new TypeError('Qwen graph output must be the two-tensor (logits, probabilities) tuple.');
  return [output.get(0), output.get(1)];
}

function validateTensorMetadata(value: unknown, label: string): TvmJsTensor {
  if (!isRecord(value)) throw new TypeError(`Qwen graph ${label} output must be a TVM.js tensor.`);
  if (value.dtype !== 'float32') throw new TypeError(`Qwen graph ${label} output must have dtype float32.`);
  if (
    !Array.isArray(value.shape) ||
    value.shape.length !== 2 ||
    !value.shape.every((dimension) => Number.isSafeInteger(dimension) && dimension > 0)
  )
    throw new TypeError(`Qwen graph ${label} output must have a positive [batch, options] shape.`);
  return value as TvmJsTensor;
}

function readProbabilityStorage(probabilities: TvmJsTensor, length: number): Float32Array {
  const device = probabilities.device;
  if (!isRecord(device) || device.deviceType !== TVMJS_CPU_DEVICE_TYPE)
    throw new TypeError('Qwen graph probabilities must be copied to a CPU-readable TVM.js tensor.');
  if (typeof probabilities.toArray !== 'function')
    throw new TypeError('Qwen graph probabilities must expose TVM.js toArray().');
  const values: unknown = probabilities.toArray();
  if (!isFloat32Array(values)) throw new TypeError('Qwen graph probabilities toArray() must return Float32Array.');
  if (values.length !== length)
    throw new TypeError('Qwen graph probability storage length must match its [batch, options] shape.');

  return values;
}

/**
 * Extract CPU-readable, padded probability rows from the pinned `(logits, probabilities)` tuple.
 * Rows preserve graph values exactly; this function does not apply temperature or softmax.
 */
export function readQwen35GraphProbabilityRows(output: unknown, nSlots: readonly number[]): readonly Float32Array[] {
  const tuple = graphTupleValues(output);
  if (tuple.length !== 2)
    throw new TypeError('Qwen graph output must be the two-tensor (logits, probabilities) tuple.');
  if (
    !Array.isArray(nSlots) ||
    nSlots.length === 0 ||
    !nSlots.every(
      (count) => Number.isSafeInteger(count) && count >= MIN_GRAPH_OPTION_WIDTH && count <= QWEN35_MAX_CHOICE_OPTIONS
    )
  )
    throw new TypeError('Qwen graph n_slots must contain option counts from 2 to 255.');

  const logits = validateTensorMetadata(tuple[0], 'logits');
  const probabilities = validateTensorMetadata(tuple[1], 'probabilities');
  const logitsShape = logits.shape as readonly number[];
  const probabilityShape = probabilities.shape as readonly number[];
  if (logitsShape[0] !== probabilityShape[0] || logitsShape[1] !== probabilityShape[1])
    throw new TypeError('Qwen graph logits and probability outputs must have matching shapes.');

  const batchSize = probabilityShape[0]!;
  const optionWidth = probabilityShape[1]!;
  if (optionWidth < MIN_GRAPH_OPTION_WIDTH || optionWidth > QWEN35_MAX_CHOICE_OPTIONS)
    throw new TypeError('Qwen graph option width must be from 2 to 255.');
  if (batchSize !== nSlots.length)
    throw new TypeError(`Qwen graph probability batch must match n_slots (${nSlots.length}).`);
  if (!nSlots.every((count) => count <= optionWidth))
    throw new TypeError('Qwen graph n_slots cannot exceed the padded option width.');

  const values = readProbabilityStorage(probabilities, batchSize * optionWidth);

  const rows: Float32Array[] = [];
  for (let rowIndex = 0; rowIndex < batchSize; rowIndex += 1) {
    const offset = rowIndex * optionWidth;
    const row = values.subarray(offset, offset + optionWidth);
    for (let optionIndex = nSlots[rowIndex]!; optionIndex < optionWidth; optionIndex += 1) {
      if (row[optionIndex] !== 0)
        throw new TypeError(`Qwen graph probability row ${rowIndex} must be zero after n_slots in padded columns.`);
    }
    rows.push(row);
  }
  return Object.freeze(rows);
}
