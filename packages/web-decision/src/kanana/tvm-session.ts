import {KANANA_EMBEDDING_SIZE, KANANA_IMAGE_TOKEN_ID, KANANA_MODEL_MAX_LENGTH, KANANA_VOCAB_SIZE} from './contract.mjs';
import {kananaVisionPatchCount, prepareKananaVisionGraphInputs} from './vision-inputs.js';
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
  parameterName?: string;
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
  /** Override only for an explicitly compiled alternate graph, such as a reduced smoke-test graph. */
  embeddingSize?: number;
  /** Override only for an explicitly compiled alternate graph, such as a reduced smoke-test graph. */
  imageTokenId?: number;
}>;

export type KananaTvmInferenceResult = Readonly<{
  shape: readonly number[];
  dtype: 'float32';
  values: Float32Array;
}>;

export type KananaTvmMultimodalInput = Readonly<{
  imageEmbeddings: Readonly<{
    shape: readonly [tokens: number, dimension: number];
    values: ArrayLike<number>;
  }>;
  imagePositions: ArrayLike<number>;
  /** Four axis-major rows with shape [4, 1, sequenceLength]. */
  positionIds: ArrayLike<number>;
}>;

export interface KananaTvmInferenceSession {
  infer(
    inputIds: ArrayLike<number>,
    attentionMask: ArrayLike<number>,
    multimodal?: KananaTvmMultimodalInput
  ): Promise<KananaTvmInferenceResult>;
  dispose(): Promise<void>;
}

export type KananaTvmVisionInferenceSessionOptions = Readonly<{
  setup: KananaTvmSetup<KananaTvmRuntimeLike>;
  weightSource: KananaTvmWeightSource;
  patchValues: number;
  numPositionEmbeddings: number;
  spatialMergeSize: number;
  /** Maximum patch count encoded in the compiled vision graph's dynamic shape range. */
  maxPatchCount: number;
}>;

export interface KananaTvmVisionInferenceSession {
  infer(
    pixelValues: ArrayLike<number>,
    visionGridThw: readonly [number, number, number]
  ): Promise<KananaTvmInferenceResult>;
  dispose(): Promise<void>;
}

type IndexedWeightRecord = {
  record: KananaTvmWeightRecord;
  sourceIndex: number;
  parameterName: string;
};

type KananaTvmInputData = {
  ids: Int32Array;
  mask: Int32Array;
  imageEmbeddings?: Float32Array;
  imagePositions?: Int32Array;
  positionIds?: Int32Array;
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

  const sourceNames = new Set<string>();
  const parameterNames = new Set<string>();
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
    const parameterName = Object.hasOwn(record, 'parameterName') ? record.parameterName : normalizeParameterName(name);
    if (
      typeof parameterName !== 'string' ||
      !parameterName ||
      sourceNames.has(name) ||
      parameterNames.has(parameterName)
    )
      fail('weight_manifest', `Kanana TVM weight parameter name ${name} is empty or duplicated.`);
    sourceNames.add(name);
    parameterNames.add(parameterName);

    return {
      name,
      parameterName,
      shape: [...dimensions],
      dtype,
      length
    };
  });
}

function inputCount(parameterNames: string[], records: KananaTvmWeightRecord[]): 2 | 5 {
  const count = parameterNames.length - records.length;
  if (count !== 2 && count !== 5)
    fail(
      'parameter_mismatch',
      `TVM main must have two text inputs or five multimodal inputs before ${records.length} weights.`
    );
  if (parameterNames[0] !== 'input_ids' || parameterNames[1] !== 'attention_mask')
    fail('parameter_mismatch', 'TVM main must begin with input_ids and attention_mask in that order.');
  if (
    count === 5 &&
    !['image_embeddings', 'image_positions', 'position_ids'].every((name, index) => parameterNames[index + 2] === name)
  )
    fail(
      'parameter_mismatch',
      'Multimodal TVM main inputs must be image_embeddings, image_positions, and position_ids.'
    );
  return count;
}

function orderedWeightRecords(
  parameterNames: string[],
  records: KananaTvmWeightRecord[],
  inputParameterCount: 2 | 5
): IndexedWeightRecord[] {
  if (parameterNames.length !== records.length + inputParameterCount)
    fail('parameter_mismatch', 'TVM main parameter count does not match its input and weight manifests.');

  const byParameterName = new Map<string, IndexedWeightRecord>();
  records.forEach((record, sourceIndex) => {
    const parameterName = record.parameterName ?? normalizeParameterName(record.name);
    byParameterName.set(parameterName, {record, sourceIndex, parameterName});
  });

  return parameterNames.slice(inputParameterCount).map((parameterName) => {
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
  const inputArity = (arity as number) - weightCount;
  if (!Number.isSafeInteger(arity) || (inputArity !== 2 && inputArity !== 5))
    fail(
      'parameter_mismatch',
      `TVM main must accept two or five inputs before its weights; received ${String(arity)} parameters.`
    );

  const getParameterName = vm.getFunction('get_function_param_name');
  const parameterNames = Array.from({length: arity as number}, (_, index) =>
    getParameterName('main', runtime.scalar(index, 'int'))
  );
  if (parameterNames.some((name) => typeof name !== 'string' || !name))
    fail('parameter_mismatch', 'TVM main parameter names must be nonempty strings.');
  return {main, parameterNames: parameterNames as string[]};
}

function indexedValues(
  value: unknown,
  label: string,
  options: {expectedLength?: number; maximumLength?: number; maximumLengthCode?: string} = {}
): number[] {
  if (!Array.isArray(value) && !(ArrayBuffer.isView(value) && !(value instanceof DataView)))
    fail('invalid_input', `${label} must be an array of numbers.`);
  const source = value as ArrayLike<unknown>;
  const length = source.length;
  if (!Number.isSafeInteger(length) || length < 0) fail('invalid_input', `${label} must have a valid length.`);
  if (options.expectedLength !== undefined && length !== options.expectedLength)
    fail('invalid_input', `${label} must contain ${options.expectedLength} values.`);
  if (options.maximumLength !== undefined && length > options.maximumLength)
    fail(
      options.maximumLengthCode ?? 'invalid_input',
      `${label} contains ${length} values, exceeding the limit of ${options.maximumLength}.`
    );
  const values: number[] = [];
  values.length = length;
  for (let index = 0; index < length; index++) {
    if (Array.isArray(value) && !Object.hasOwn(value, index))
      fail('invalid_input', `${label} must not contain empty entries.`);
    const item = source[index];
    if (typeof item !== 'number' || !Number.isSafeInteger(item))
      fail('invalid_input', `${label} must contain safe integer values.`);
    values[index] = item;
  }
  return values;
}

function validateTokenIds(ids: readonly number[]): void {
  if (!ids.length) fail('invalid_input', 'Kanana TVM input_ids must not be empty.');
  if (ids.length > KANANA_MODEL_MAX_LENGTH)
    fail(
      'input_too_long',
      `Kanana TVM input has ${ids.length} tokens, exceeding the pinned limit of ${KANANA_MODEL_MAX_LENGTH}.`
    );
  if (ids.some((id) => id < 0 || id >= KANANA_VOCAB_SIZE))
    fail('invalid_input', 'Kanana TVM input_ids contain an ID outside the pinned vocabulary.');
}

function validateAttentionMask(mask: readonly number[], sequenceLength: number): void {
  if (mask.length !== sequenceLength) fail('invalid_input', 'attention_mask length must match input_ids.');
  if (mask.some((value) => value !== 0 && value !== 1)) fail('invalid_input', 'attention_mask values must be 0 or 1.');
  if (!mask.includes(1)) fail('invalid_input', 'Kanana TVM input must contain an attended token.');
}

function linearPositionIds(sequenceLength: number): Int32Array {
  const linearPositions = Int32Array.from({length: sequenceLength}, (_, index) => index);
  const positionIds = new Int32Array(4 * sequenceLength);
  for (let axis = 0; axis < 4; axis++) positionIds.set(linearPositions, axis * sequenceLength);
  return positionIds;
}

function validateMultimodalPositions(positionValues: readonly number[], sequenceLength: number): void {
  if (positionValues.length !== 4 * sequenceLength || positionValues.some((value) => value < 0 || value > 0x7fffffff))
    fail('invalid_input', 'position_ids must contain four nonnegative int32 axis rows matching the input sequence.');
}

function validateImageEmbeddingShape(
  shape: readonly number[] | undefined,
  values: ArrayLike<number> | undefined,
  embeddingSize: number
): asserts shape is readonly [number, number] {
  if (
    !Array.isArray(shape) ||
    shape.length !== 2 ||
    !Number.isSafeInteger(shape[0]) ||
    shape[0] < 0 ||
    shape[1] !== embeddingSize ||
    !values ||
    !Number.isSafeInteger(values.length) ||
    values.length !== shape[0] * embeddingSize
  )
    fail('invalid_input', `image_embeddings must have shape [image tokens, ${embeddingSize}].`);
}

function validateImageTokenPositions(
  ids: readonly number[],
  imagePositions: readonly number[],
  imageTokenId: number
): void {
  const tokenPositions = ids.flatMap((id, index) => (id === imageTokenId ? [index] : []));
  if (
    imagePositions.length !== tokenPositions.length ||
    imagePositions.some((position, index) => position !== tokenPositions[index])
  )
    fail('invalid_input', 'image_positions must match image-token positions in prompt order.');
}

function copyFloat32Values(values: ArrayLike<number>, label: string): Float32Array {
  const copy = new Float32Array(values.length);
  for (let index = 0; index < values.length; index++) {
    const value = values[index];
    const float32Value = typeof value === 'number' ? Math.fround(value) : Number.NaN;
    if (!Number.isFinite(value) || !Number.isFinite(float32Value))
      fail('invalid_input', `${label} must contain finite float32 values.`);
    copy[index] = float32Value;
  }
  return copy;
}

function multimodalInputData(
  ids: readonly number[],
  multimodal: KananaTvmMultimodalInput,
  positionValues: readonly number[],
  embeddingSize: number,
  imageTokenId: number
): Pick<KananaTvmInputData, 'imageEmbeddings' | 'imagePositions' | 'positionIds'> {
  const sequenceLength = ids.length;
  validateMultimodalPositions(positionValues, sequenceLength);
  const imagePositions = indexedValues(multimodal.imagePositions, 'image_positions', {
    maximumLength: sequenceLength
  });
  const imageEmbeddingShape = multimodal.imageEmbeddings?.shape;
  const imageEmbeddingValues = multimodal.imageEmbeddings?.values;
  validateImageEmbeddingShape(imageEmbeddingShape, imageEmbeddingValues, embeddingSize);
  if (imageEmbeddingShape[0] !== imagePositions.length)
    fail('invalid_input', 'image_embeddings and image_positions must contain the same number of image tokens.');
  validateImageTokenPositions(ids, imagePositions, imageTokenId);
  return {
    imageEmbeddings: copyFloat32Values(imageEmbeddingValues, 'image_embeddings'),
    imagePositions: Int32Array.from(imagePositions),
    positionIds: Int32Array.from(positionValues)
  };
}

function inputTensorData(
  inputIds: ArrayLike<number>,
  attentionMask: ArrayLike<number>,
  multimodal: KananaTvmMultimodalInput | undefined,
  expectsMultimodalInputs: boolean,
  embeddingSize: number,
  imageTokenId: number
): KananaTvmInputData {
  const ids = indexedValues(inputIds, 'input_ids', {
    maximumLength: KANANA_MODEL_MAX_LENGTH,
    maximumLengthCode: 'input_too_long'
  });
  validateTokenIds(ids);

  const mask = indexedValues(attentionMask, 'attention_mask', {expectedLength: ids.length});
  validateAttentionMask(mask, ids.length);
  const result: KananaTvmInputData = {ids: Int32Array.from(ids), mask: Int32Array.from(mask)};
  if (!expectsMultimodalInputs) {
    if (multimodal) fail('graph_mismatch', 'The loaded TVM graph does not accept multimodal inputs.');
    return result;
  }

  const sequenceLength = ids.length;
  const positionValues = multimodal
    ? indexedValues(multimodal.positionIds, 'position_ids', {expectedLength: 4 * sequenceLength})
    : [];
  if (!multimodal) {
    result.imageEmbeddings = new Float32Array(0);
    result.imagePositions = new Int32Array(0);
    result.positionIds = linearPositionIds(sequenceLength);
    return result;
  }

  Object.assign(result, multimodalInputData(ids, multimodal, positionValues, embeddingSize, imageTokenId));
  return result;
}

function visionInputData(
  pixelValues: ArrayLike<number>,
  gridValue: readonly [number, number, number],
  options: Pick<
    KananaTvmVisionInferenceSessionOptions,
    'patchValues' | 'numPositionEmbeddings' | 'spatialMergeSize' | 'maxPatchCount'
  >
) {
  if (
    !Number.isSafeInteger(options.patchValues) ||
    options.patchValues < 1 ||
    !Number.isSafeInteger(options.maxPatchCount) ||
    options.maxPatchCount < 1
  )
    fail('vision_config', 'Kanana TVM vision patch dimensions must be positive safe integers.');
  if (!pixelValues || !Number.isSafeInteger(pixelValues.length) || pixelValues.length < 1)
    fail('invalid_input', 'pixel_values must be a nonempty indexed sequence.');
  const patchCount = kananaVisionPatchCount(gridValue, options.spatialMergeSize);
  if (patchCount > options.maxPatchCount)
    fail('input_too_large', `Kanana vision graph supports at most ${options.maxPatchCount} patches.`);
  const inputs = prepareKananaVisionGraphInputs(gridValue, {
    numPositionEmbeddings: options.numPositionEmbeddings,
    spatialMergeSize: options.spatialMergeSize
  });
  if (pixelValues.length !== patchCount * options.patchValues)
    fail('invalid_input', `pixel_values must contain ${patchCount * options.patchValues} values for this vision grid.`);

  const pixels = copyFloat32Values(pixelValues, 'pixel_values');
  return {pixels, ...inputs};
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

function validateInferenceOptions(
  weightSource: KananaTvmWeightSource,
  embeddingSize: number,
  imageTokenId: number
): void {
  if (!weightSource || typeof weightSource !== 'object')
    fail('weight_manifest', 'A verified Kanana TVM weight source is required.');
  if (!Number.isSafeInteger(embeddingSize) || embeddingSize < 1)
    fail('invalid_config', 'Kanana TVM embeddingSize must be a positive safe integer.');
  if (!Number.isSafeInteger(imageTokenId) || imageTokenId < 0 || imageTokenId >= KANANA_VOCAB_SIZE)
    fail('invalid_config', 'Kanana TVM imageTokenId must be within the pinned vocabulary.');
}

async function loadOrderedWeights(
  source: KananaTvmWeightSource,
  ordered: readonly IndexedWeightRecord[],
  runtime: KananaTvmRuntimeLike,
  device: TvmDeviceLike & object
): Promise<TvmTensorLike[]> {
  const weights: TvmTensorLike[] = [];
  for (const entry of ordered) {
    const bytes = await readWeightBytes(source, entry.record, entry.sourceIndex);
    const tensor = runtime.empty([...entry.record.shape], entry.record.dtype, device).copyFromRawBytes(bytes);
    weights.push(tensor);
    if (weights.length % 4 === 0 || weights.length === ordered.length) await device.sync();
  }
  return weights;
}

function wrapInferenceError(stage: string, error: unknown): Error {
  const detail = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
  const failure = new Error(`Kanana TVM ${stage} failed: ${detail}`, {cause: error});
  const code = error && typeof error === 'object' && 'code' in error ? error.code : undefined;
  if (code !== undefined) Object.assign(failure, {code});
  return failure;
}

function throwInferenceErrors(
  inferenceFailed: boolean,
  inferenceError: unknown,
  scopeCleanupFailed: boolean,
  scopeCleanupError: unknown
): void {
  if (inferenceFailed && scopeCleanupFailed)
    throw new AggregateError(
      [inferenceError, scopeCleanupError],
      'Kanana TVM inference and scope cleanup both failed.',
      {cause: inferenceError}
    );
  if (inferenceFailed) throw inferenceError;
  if (scopeCleanupFailed) throw scopeCleanupError;
}

/**
 * Load caller-verified weights into an already scoped TVM setup and create a text-graph inference session.
 * The weight reader must return verified bytes; transport, caching, and asset verification stay with the caller.
 */
export async function createKananaTvmInferenceSession({
  setup,
  weightSource,
  embeddingSize = KANANA_EMBEDDING_SIZE,
  imageTokenId = KANANA_IMAGE_TOKEN_ID
}: KananaTvmInferenceSessionOptions): Promise<KananaTvmInferenceSession> {
  try {
    validateInferenceOptions(weightSource, embeddingSize, imageTokenId);
    const records = weightRecords(weightSource);
    const {main, parameterNames} = inspectMain(setup, records.length);
    const inputParameterCount = inputCount(parameterNames, records);
    const ordered = orderedWeightRecords(parameterNames, records, inputParameterCount);
    const runtime = setup.runtime as KananaTvmRuntimeLike;
    const device = runtime.webgpu();
    const weights = await loadOrderedWeights(weightSource, ordered, runtime, device);

    let disposed = false;
    let pendingInference = Promise.resolve();
    let disposal: Promise<void> | undefined;

    async function runInference(inputs: KananaTvmInputData): Promise<KananaTvmInferenceResult> {
      const {ids, mask} = inputs;
      runtime.beginScope();
      let inferenceFailed = false;
      let inferenceError: unknown;
      let scopeCleanupFailed = false;
      let scopeCleanupError: unknown;
      let result: KananaTvmInferenceResult | undefined;
      let stage = 'input tensor preparation';
      try {
        const idsTensor = runtime.empty([1, ids.length], 'int32', device).copyFrom(ids);
        const maskTensor = runtime.empty([1, mask.length], 'int32', device).copyFrom(mask);
        const modelInputs: TvmTensorLike[] = [idsTensor, maskTensor];
        if (inputParameterCount === 5) {
          const imageEmbeddings = inputs.imageEmbeddings ?? new Float32Array(0);
          const imagePositions = inputs.imagePositions ?? new Int32Array(0);
          const positionIds = inputs.positionIds;
          if (!positionIds) fail('invalid_input', 'Multimodal TVM inputs are incomplete.');
          modelInputs.push(
            runtime.empty([imagePositions.length, embeddingSize], 'float32', device).copyFrom(imageEmbeddings),
            runtime.empty([imagePositions.length], 'int32', device).copyFrom(imagePositions),
            runtime.empty([4, 1, ids.length], 'int32', device).copyFrom(positionIds)
          );
        }
        stage = 'graph execution';
        const output = (await main(...modelInputs, ...weights)) as TvmTensorLike;
        stage = 'WebGPU execution synchronization';
        await device.sync();

        stage = 'output validation';
        if (!output || output.dtype !== 'float32' || typeof output.toRawBytes !== 'function')
          fail('output_dtype', 'Kanana TVM main must return float32 hidden-state values.');
        const shape = outputShape(output.shape);
        stage = 'output copy to CPU';
        const host = runtime.empty(shape, 'float32', runtime.cpu()).copyFrom(output);
        stage = 'CPU output synchronization';
        await device.sync();
        stage = 'CPU output readback';
        const values = float32Values(host.toRawBytes(), shape);
        result = Object.freeze({shape: Object.freeze(shape), dtype: 'float32' as const, values});
      } catch (error) {
        inferenceFailed = true;
        inferenceError = wrapInferenceError(stage, error);
      }
      try {
        runtime.endScope();
      } catch (error) {
        scopeCleanupFailed = true;
        scopeCleanupError = error;
      }
      throwInferenceErrors(inferenceFailed, inferenceError, scopeCleanupFailed, scopeCleanupError);
      if (!result) fail('inference_result', 'Kanana TVM inference did not produce hidden-state values.');
      return result;
    }

    return Object.freeze({
      infer(inputIds: ArrayLike<number>, attentionMask: ArrayLike<number>, multimodal?: KananaTvmMultimodalInput) {
        if (disposed)
          return Promise.reject(
            new KananaTvmSessionError('session_disposed', 'Kanana TVM inference session has been disposed.')
          );
        let inputs: {ids: Int32Array; mask: Int32Array};
        try {
          inputs = inputTensorData(
            inputIds,
            attentionMask,
            multimodal,
            inputParameterCount === 5,
            embeddingSize,
            imageTokenId
          );
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

/** Load the caller-verified vision weights and run one preprocessed image through the pinned TVM vision graph. */
export async function createKananaTvmVisionInferenceSession({
  setup,
  weightSource,
  patchValues,
  numPositionEmbeddings,
  spatialMergeSize,
  maxPatchCount
}: KananaTvmVisionInferenceSessionOptions): Promise<KananaTvmVisionInferenceSession> {
  try {
    if (!weightSource || typeof weightSource !== 'object')
      fail('weight_manifest', 'A verified Kanana TVM vision weight source is required.');
    const records = weightRecords(weightSource);
    const {main, parameterNames} = inspectMain(setup, records.length);
    const inputNames = ['pixels', 'positions', 'indices', 'weights', 'merged_tokens'];
    if (
      parameterNames.length !== records.length + inputNames.length ||
      inputNames.some((name, index) => parameterNames[index] !== name)
    )
      fail(
        'parameter_mismatch',
        'Kanana TVM vision main must begin with pixels, positions, indices, weights, and merged_tokens.'
      );
    const ordered = orderedWeightRecords(parameterNames, records, 5);
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

    async function runInference(inputs: ReturnType<typeof visionInputData>): Promise<KananaTvmInferenceResult> {
      runtime.beginScope();
      let result: KananaTvmInferenceResult | undefined;
      let inferenceError: unknown;
      let scopeError: unknown;
      try {
        const patchCount = inputs.positionShape[0];
        const mergedCount = inputs.mergedTokens.length;
        const modelInputs = [
          runtime.empty([patchCount, patchValues], 'float32', device).copyFrom(inputs.pixels),
          runtime.empty([...inputs.positionShape], 'int32', device).copyFrom(inputs.positionIds),
          runtime.empty([...inputs.interpolationShape], 'int32', device).copyFrom(inputs.interpolationIndices),
          runtime.empty([...inputs.interpolationShape], 'float32', device).copyFrom(inputs.interpolationWeights),
          runtime.empty([mergedCount], 'int32', device).copyFrom(inputs.mergedTokens)
        ];
        const output = (await main(...modelInputs, ...weights)) as TvmTensorLike;
        await device.sync();
        if (!output || output.dtype !== 'float32' || typeof output.toRawBytes !== 'function')
          fail('output_dtype', 'Kanana TVM vision main must return float32 image embeddings.');
        const shape = outputShape(output.shape);
        if (shape.length !== 2 || shape[0] !== mergedCount || shape[1] !== KANANA_EMBEDDING_SIZE)
          fail('output_shape', `Kanana TVM vision output must have shape [${mergedCount}, ${KANANA_EMBEDDING_SIZE}].`);
        const host = runtime.empty(shape, 'float32', runtime.cpu()).copyFrom(output);
        await device.sync();
        result = Object.freeze({
          shape: Object.freeze(shape),
          dtype: 'float32' as const,
          values: float32Values(host.toRawBytes(), shape)
        });
      } catch (error) {
        inferenceError = error;
      }
      try {
        runtime.endScope();
      } catch (error) {
        scopeError = error;
      }
      if (inferenceError && scopeError)
        throw new AggregateError(
          [inferenceError, scopeError],
          'Kanana vision inference and scope cleanup both failed.',
          {
            cause: inferenceError
          }
        );
      if (inferenceError) throw inferenceError;
      if (scopeError) throw scopeError;
      if (!result) fail('inference_result', 'Kanana TVM vision inference returned no embeddings.');
      return result;
    }

    return Object.freeze({
      infer(pixelValues: ArrayLike<number>, visionGridThw: readonly [number, number, number]) {
        if (disposed)
          return Promise.reject(
            new KananaTvmSessionError('session_disposed', 'Kanana TVM vision session has been disposed.')
          );
        let inputs: ReturnType<typeof visionInputData>;
        try {
          inputs = visionInputData(pixelValues, visionGridThw, {
            patchValues,
            numPositionEmbeddings,
            spatialMergeSize,
            maxPatchCount
          });
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
