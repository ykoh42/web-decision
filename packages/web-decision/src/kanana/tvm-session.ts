import {KANANA_MODEL_MAX_LENGTH, KANANA_VOCAB_SIZE} from './contract.mjs';
import type {KananaTvmSetup, TvmRuntimeLike} from './tvm-runtime.js';

type ByteInput = ArrayBuffer | ArrayBufferView;
type TvmPackedFunction = (...args: unknown[]) => unknown;

interface TvmTensorLike {
  readonly shape: readonly number[];
  readonly dtype: string;
  copyFrom(source: unknown): this;
  copyFromRawBytes(bytes: Uint8Array): this;
  toRawBytes(): Uint8Array;
}

interface TvmDeviceLike {
  sync(): void | PromiseLike<void>;
}

interface TvmVirtualMachineLike {
  getFunction(name: string): TvmPackedFunction;
}

interface KananaTvmRuntimeLike extends TvmRuntimeLike<TvmVirtualMachineLike> {
  cpu(): object;
  empty(shape: number[], dtype: string, device: object): TvmTensorLike;
  scalar(value: number, dtype: string): unknown;
  webgpu(): TvmDeviceLike & object;
}

export type KananaTvmWeightRecord = Readonly<{
  name: string;
  shape: readonly number[];
  dtype: string;
  length: number;
}>;

export type KananaTvmVerifiedWeightRecord = KananaTvmWeightRecord & Readonly<{bytes: ByteInput}>;

export type KananaTvmWeightManifest = Readonly<{
  weights: readonly KananaTvmWeightRecord[];
}>;

type VerifiedWeightSource = Readonly<{
  weights: readonly KananaTvmVerifiedWeightRecord[];
}>;

type ReadWeightSource = Readonly<{
  manifest: KananaTvmWeightManifest;
  readWeight(record: KananaTvmWeightRecord, index: number): ByteInput | PromiseLike<ByteInput>;
}>;

export type KananaTvmWeightSource = VerifiedWeightSource | ReadWeightSource;

export type KananaTvmInferenceSessionOptions = Readonly<{
  setup: KananaTvmSetup<KananaTvmRuntimeLike>;
  weightSource: KananaTvmWeightSource;
}>;

export type KananaTvmInferenceResult = Readonly<{
  shape: readonly number[];
  dtype: 'float32';
  values: Float32Array;
}>;

export interface KananaTvmInferenceSession {
  infer(inputIds: ArrayLike<number>, attentionMask: ArrayLike<number>): Promise<KananaTvmInferenceResult>;
  dispose(): Promise<void>;
}

type IndexedWeightRecord = {
  record: KananaTvmWeightRecord;
  sourceIndex: number;
  parameterName: string;
};

class KananaTvmSessionError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = 'KananaTvmSessionError';
    this.code = code;
  }
}

function fail(code: string, message: string): never {
  throw new KananaTvmSessionError(code, message);
}

function normalizeParameterName(name: string): string {
  return name.replace(/[^A-Za-z0-9_]/g, '_');
}

function byteView(value: ByteInput, label: string): Uint8Array {
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (!ArrayBuffer.isView(value) || value instanceof DataView)
    fail('weight_bytes', `${label} bytes must be an ArrayBuffer or typed-array view.`);
  return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
}

function weightRecords(source: KananaTvmWeightSource): KananaTvmWeightRecord[] {
  const sourceRecords: unknown = 'weights' in source ? source.weights : source.manifest?.weights;
  if (!Array.isArray(sourceRecords)) fail('weight_manifest', 'Kanana TVM weights must be provided as an array.');

  const names = new Set<string>();
  return (sourceRecords as unknown[]).map((item, index) => {
    if (!item || typeof item !== 'object' || Array.isArray(item))
      fail('weight_manifest', `Kanana TVM weight record ${index} is invalid.`);
    const record = item as Record<string, unknown>;
    const shape = record.shape;
    if (
      typeof record.name !== 'string' ||
      !record.name ||
      typeof record.dtype !== 'string' ||
      !record.dtype ||
      !Number.isSafeInteger(record.length) ||
      (record.length as number) < 1 ||
      !Array.isArray(shape) ||
      !shape.length ||
      (shape as unknown[]).some((dimension: unknown) => !Number.isSafeInteger(dimension) || (dimension as number) < 1)
    )
      fail('weight_manifest', `Kanana TVM weight record ${index} is invalid.`);

    const name = record.name as string;
    const dtype = record.dtype as string;
    const length = record.length as number;
    const dimensions = shape as number[];
    const parameterName = normalizeParameterName(name);
    if (!parameterName || names.has(parameterName))
      fail('weight_manifest', `Kanana TVM weight parameter name ${name} is empty or duplicated.`);
    names.add(parameterName);

    return {
      name,
      shape: [...dimensions],
      dtype,
      length
    };
  });
}

function orderedWeightRecords(parameterNames: string[], records: KananaTvmWeightRecord[]): IndexedWeightRecord[] {
  if (parameterNames.length !== records.length + 2)
    fail(
      'parameter_mismatch',
      `TVM main expects ${parameterNames.length} parameters, but the weight manifest has ${records.length} records.`
    );
  if (parameterNames[0] !== 'input_ids' || parameterNames[1] !== 'attention_mask')
    fail('parameter_mismatch', 'TVM main must begin with input_ids and attention_mask in that order.');

  const byParameterName = new Map<string, IndexedWeightRecord>();
  records.forEach((record, sourceIndex) => {
    const parameterName = normalizeParameterName(record.name);
    byParameterName.set(parameterName, {record, sourceIndex, parameterName});
  });

  return parameterNames.slice(2).map((parameterName) => {
    const entry = byParameterName.get(parameterName);
    if (!entry) fail('parameter_mismatch', `TVM main parameter ${parameterName} has no matching weight record.`);
    return entry;
  });
}

function inspectMain(setup: KananaTvmInferenceSessionOptions['setup'], weightCount: number) {
  const runtime = setup.runtime as KananaTvmRuntimeLike;
  const vm = setup.vm as TvmVirtualMachineLike;
  const main = vm.getFunction('main');
  if (typeof main !== 'function') fail('parameter_mismatch', 'TVM VM does not expose main().');

  const arity = vm.getFunction('get_function_arity')('main');
  if (!Number.isSafeInteger(arity) || (arity as number) !== weightCount + 2)
    fail('parameter_mismatch', `TVM main arity must be ${weightCount + 2}; received ${String(arity)}.`);

  const getParameterName = vm.getFunction('get_function_param_name');
  const parameterNames = Array.from({length: arity as number}, (_, index) =>
    getParameterName('main', runtime.scalar(index, 'int'))
  );
  if (parameterNames.some((name) => typeof name !== 'string' || !name))
    fail('parameter_mismatch', 'TVM main parameter names must be nonempty strings.');
  return {main, parameterNames: parameterNames as string[]};
}

function indexedValues(value: unknown, label: string): number[] {
  if (!Array.isArray(value) && !(ArrayBuffer.isView(value) && !(value instanceof DataView)))
    fail('invalid_input', `${label} must be an array of numbers.`);
  const source = value as ArrayLike<unknown>;
  const values: number[] = [];
  values.length = source.length;
  for (let index = 0; index < source.length; index++) {
    if (Array.isArray(value) && !Object.hasOwn(value, index))
      fail('invalid_input', `${label} must not contain empty entries.`);
    const item = source[index];
    if (typeof item !== 'number' || !Number.isSafeInteger(item))
      fail('invalid_input', `${label} must contain safe integer values.`);
    values[index] = item;
  }
  return values;
}

function inputTensorData(inputIds: ArrayLike<number>, attentionMask: ArrayLike<number>) {
  const ids = indexedValues(inputIds, 'input_ids');
  if (!ids.length) fail('invalid_input', 'Kanana TVM input_ids must not be empty.');
  if (ids.length > KANANA_MODEL_MAX_LENGTH)
    fail(
      'input_too_long',
      `Kanana TVM input has ${ids.length} tokens, exceeding the pinned limit of ${KANANA_MODEL_MAX_LENGTH}.`
    );
  if (ids.some((id) => id < 0 || id >= KANANA_VOCAB_SIZE))
    fail('invalid_input', 'Kanana TVM input_ids contain an ID outside the pinned vocabulary.');

  const mask = indexedValues(attentionMask, 'attention_mask');
  if (mask.length !== ids.length) fail('invalid_input', 'attention_mask length must match input_ids.');
  if (mask.some((value) => value !== 0 && value !== 1)) fail('invalid_input', 'attention_mask values must be 0 or 1.');
  if (!mask.includes(1)) fail('invalid_input', 'Kanana TVM input must contain an attended token.');
  return {ids: Int32Array.from(ids), mask: Int32Array.from(mask)};
}

function outputShape(value: unknown): number[] {
  if (
    !Array.isArray(value) ||
    !value.length ||
    value.some((dimension) => !Number.isSafeInteger(dimension) || dimension < 1)
  )
    fail('output_shape', 'Kanana TVM main returned an invalid hidden-state shape.');
  return [...value] as number[];
}

function float32Values(bytes: Uint8Array, shape: number[]): Float32Array {
  const elementCount = shape.reduce((product, dimension) => product * dimension, 1);
  if (!Number.isSafeInteger(elementCount) || bytes.byteLength !== elementCount * Float32Array.BYTES_PER_ELEMENT)
    fail('output_shape', 'Kanana TVM hidden-state bytes do not match the returned float32 shape.');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Float32Array.from({length: elementCount}, (_, index) => view.getFloat32(index * 4, true));
}

async function readWeightBytes(
  source: KananaTvmWeightSource,
  record: KananaTvmWeightRecord,
  index: number
): Promise<Uint8Array> {
  let bytes: ByteInput;
  if ('weights' in source) {
    const weight = source.weights[index];
    if (!weight) fail('weight_bytes', `Verified Kanana TVM weight ${record.name} is missing.`);
    bytes = weight.bytes;
  } else {
    bytes = await source.readWeight(record, index);
  }
  const view = byteView(bytes, record.name);
  if (view.byteLength !== record.length)
    fail('weight_bytes', `${record.name} has ${view.byteLength} bytes; expected ${record.length}.`);
  return view;
}

function releaseSetupAfterFailure(setup: KananaTvmInferenceSessionOptions['setup'], error: unknown): never {
  try {
    setup.dispose();
  } catch (cleanupError) {
    throw new AggregateError([error, cleanupError], 'Kanana TVM session setup and cleanup both failed.', {
      cause: error
    });
  }
  throw error;
}

/**
 * Load caller-verified weights into an already scoped TVM setup and create a text-graph inference session.
 * The weight reader must return verified bytes; transport, caching, and asset verification stay with the caller.
 */
export async function createKananaTvmInferenceSession({
  setup,
  weightSource
}: KananaTvmInferenceSessionOptions): Promise<KananaTvmInferenceSession> {
  try {
    if (!weightSource || typeof weightSource !== 'object')
      fail('weight_manifest', 'A verified Kanana TVM weight source is required.');
    const records = weightRecords(weightSource);
    const {main, parameterNames} = inspectMain(setup, records.length);
    const ordered = orderedWeightRecords(parameterNames, records);
    const runtime = setup.runtime as KananaTvmRuntimeLike;
    const device = runtime.webgpu();
    const weights: TvmTensorLike[] = [];

    for (const entry of ordered) {
      const bytes = await readWeightBytes(weightSource, entry.record, entry.sourceIndex);
      const tensor = runtime.empty([...entry.record.shape], entry.record.dtype, device).copyFromRawBytes(bytes);
      weights.push(tensor);
      if (weights.length % 4 === 0 || weights.length === ordered.length) await device.sync();
    }

    let disposed = false;
    let pendingInference = Promise.resolve();
    let disposal: Promise<void> | undefined;

    async function runInference(inputs: {ids: Int32Array; mask: Int32Array}): Promise<KananaTvmInferenceResult> {
      const {ids, mask} = inputs;
      runtime.beginScope();
      let inferenceFailed = false;
      let inferenceError: unknown;
      let scopeCleanupFailed = false;
      let scopeCleanupError: unknown;
      let result: KananaTvmInferenceResult | undefined;
      try {
        const idsTensor = runtime.empty([1, ids.length], 'int32', device).copyFrom(ids);
        const maskTensor = runtime.empty([1, mask.length], 'int32', device).copyFrom(mask);
        const output = (await main(idsTensor, maskTensor, ...weights)) as TvmTensorLike;
        await device.sync();

        if (!output || output.dtype !== 'float32' || typeof output.toRawBytes !== 'function')
          fail('output_dtype', 'Kanana TVM main must return float32 hidden-state values.');
        const shape = outputShape(output.shape);
        const host = runtime.empty(shape, 'float32', runtime.cpu()).copyFrom(output);
        await device.sync();
        const values = float32Values(host.toRawBytes(), shape);
        result = Object.freeze({shape: Object.freeze(shape), dtype: 'float32' as const, values});
      } catch (error) {
        inferenceFailed = true;
        inferenceError = error;
      }
      try {
        runtime.endScope();
      } catch (error) {
        scopeCleanupFailed = true;
        scopeCleanupError = error;
      }
      if (inferenceFailed && scopeCleanupFailed)
        throw new AggregateError(
          [inferenceError, scopeCleanupError],
          'Kanana TVM inference and scope cleanup both failed.',
          {cause: inferenceError}
        );
      if (inferenceFailed) throw inferenceError;
      if (scopeCleanupFailed) throw scopeCleanupError;
      if (!result) fail('inference_result', 'Kanana TVM inference did not produce hidden-state values.');
      return result;
    }

    return Object.freeze({
      infer(inputIds: ArrayLike<number>, attentionMask: ArrayLike<number>) {
        if (disposed)
          return Promise.reject(
            new KananaTvmSessionError('session_disposed', 'Kanana TVM inference session has been disposed.')
          );
        let inputs: {ids: Int32Array; mask: Int32Array};
        try {
          inputs = inputTensorData(inputIds, attentionMask);
        } catch (error) {
          return Promise.reject(error);
        }
        const result = pendingInference.then(() => runInference(inputs));
        pendingInference = result.then(
          () => undefined,
          () => undefined
        );
        return result;
      },
      dispose() {
        if (disposal) return disposal;
        disposed = true;
        disposal = pendingInference.then(() => setup.dispose());
        return disposal;
      }
    });
  } catch (error) {
    return releaseSetupAfterFailure(setup, error);
  }
}
