import type {Qwen35GraphInputRecord} from './qwen35-graph-input.mts';
import type {Qwen35PerImageVisionGeometry} from './qwen35-vision-geometry.mts';
import type {Instance, Tensor, VirtualMachine} from '../runtime/tvm-types.js';

const INPUTS = Object.freeze([
  {name: 'pixel_values', dtype: 'float32'},
  {name: 'grid_thw', dtype: 'int32'},
  {name: 'interp_indices', dtype: 'int32'},
  {name: 'interp_weights', dtype: 'float32'},
  {name: 'position_ids_by_token', dtype: 'int32'},
  {name: 'cu_seqlens', dtype: 'int32'}
] as const);
const PATCH_WIDTH = 1536;
const FEATURE_WIDTH = 2048;

export interface Qwen35VisionArtifactArgument {
  readonly name: string;
  readonly sourceKind: 'USER_INPUT' | 'PARAMETER';
  readonly dtype: string;
  readonly shape?: readonly number[];
  readonly relaxShape?: readonly string[];
}

export interface Qwen35VisionRopeInitializationEvidence {
  readonly restoredBuffers: readonly ['inv_freq', 'original_inv_freq'];
  readonly sourceModelParameterDtype: 'bfloat16';
  readonly bufferDtype: 'float32';
  readonly matchesPinnedInitialization: true;
  readonly bufferSha256: Readonly<{
    inv_freq: string;
    original_inv_freq: string;
  }>;
}

export type Qwen35VisionActivationPrecision = 'bfloat16' | 'float32';

export interface Qwen35VisionArtifact {
  readonly format: 'qwen35-dynamic-per-image-vision-tvmjs-fixture-v1';
  /** The original fixture omits this field and evaluates the source BF16 torso. */
  readonly activationPrecision?: Qwen35VisionActivationPrecision;
  readonly wasm?: Readonly<{bytes: number; sha256: string}>;
  readonly visionRopeInitialization: Qwen35VisionRopeInitializationEvidence;
  readonly shapeContract: Readonly<{
    oneImagePerInvocation: boolean;
    patchRows: Readonly<{formula: string; minimum: number; maximum: number}>;
  }>;
  readonly mainArgumentOrder: readonly Qwen35VisionArtifactArgument[];
}

export interface Qwen35VisionFeatureRunnerOptions {
  /** An instantiated TVM.js runtime whose WebGPU device has already been initialized. */
  readonly runtime: Pick<Instance, 'beginScope' | 'endScope' | 'createVirtualMachine' | 'empty' | 'webgpu' | 'cpu'>;
  /** The loaded, hash-checked compiler fixture.json for the injected artifact. */
  readonly artifact: Qwen35VisionArtifact;
  /** Verified external parameter tensors allocated by the same TVM runtime. */
  readonly parameterTensors: ReadonlyMap<string, Tensor>;
}

export interface Qwen35VisionFeatureRunner {
  /** Run each source-ordered image and return its features in float32 storage. */
  run(input: Qwen35GraphInputRecord, signal?: AbortSignal | null): Promise<readonly Float32Array[]>;
}

type RuntimeTensor = Tensor;

type VisionInputs = Readonly<
  Record<(typeof INPUTS)[number]['name'], Readonly<{shape: number[]; dtype: string; data: Int32Array | Float32Array}>>
>;

function invalid(message: string): never {
  throw new TypeError('Invalid Qwen3.5 vision runner: ' + message);
}

export function qwen35VisionActivationPrecision(artifact: Qwen35VisionArtifact): Qwen35VisionActivationPrecision {
  const precision = artifact.activationPrecision;
  if (precision === undefined || precision === 'bfloat16') return 'bfloat16';
  if (precision === 'float32') return precision;
  invalid('artifact declares an unsupported activation precision.');
}

export function isQwen35FiniteActivationArray(
  values: Float32Array,
  precision: Qwen35VisionActivationPrecision
): boolean {
  if (!(values instanceof Float32Array)) return false;
  if (precision === 'float32') return values.every(Number.isFinite);
  if (precision !== 'bfloat16') return false;
  const bits = new Uint32Array(values.buffer, values.byteOffset, values.length);
  return values.every((value, index) => Number.isFinite(value) && (bits[index]! & 0xffff) === 0);
}

export function isPinnedQwen35VisionRopeInitializationEvidence(
  value: unknown
): value is Qwen35VisionRopeInitializationEvidence {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const evidence = value as Partial<Qwen35VisionRopeInitializationEvidence>;
  if (
    evidence?.restoredBuffers?.[0] !== 'inv_freq' ||
    evidence.restoredBuffers[1] !== 'original_inv_freq' ||
    evidence.restoredBuffers.length !== 2 ||
    evidence.sourceModelParameterDtype !== 'bfloat16' ||
    evidence.bufferDtype !== 'float32' ||
    evidence.matchesPinnedInitialization !== true ||
    evidence.bufferSha256?.inv_freq !== '9cc63b858a37a16bf14e559754d6fee5f50ae74aabfce29fec9097b70902849c' ||
    evidence.bufferSha256.original_inv_freq !== '9cc63b858a37a16bf14e559754d6fee5f50ae74aabfce29fec9097b70902849c'
  )
    return false;
  return true;
}

function assertShape(actual: readonly number[], expected: readonly number[], name: string): void {
  if (actual.length !== expected.length || actual.some((value, index) => value !== expected[index]))
    invalid(name + ' does not match the dynamic vision graph shape.');
}

function scaledSymbol(expression: string | undefined, factor: number): string | undefined {
  if (factor === 1) return expression?.match(/^s\d+$/)?.[0];
  return expression?.match(new RegExp(`^T\\.int64\\(${factor}\\)\\s*\\*\\s*(s\\d+)$`))?.[1];
}

function assertArtifact(artifact: Qwen35VisionArtifact): readonly Qwen35VisionArtifactArgument[] {
  if (
    artifact?.format !== 'qwen35-dynamic-per-image-vision-tvmjs-fixture-v1' ||
    artifact.shapeContract?.oneImagePerInvocation !== true ||
    artifact.shapeContract.patchRows?.formula !== '4*S' ||
    artifact.shapeContract.patchRows.minimum !== 256 ||
    artifact.shapeContract.patchRows.maximum !== 65_536 ||
    !Array.isArray(artifact.mainArgumentOrder)
  )
    invalid('artifact must declare the pinned dynamic one-image P=4*S range [256,65536].');

  if (!isPinnedQwen35VisionRopeInitializationEvidence(artifact.visionRopeInitialization))
    invalid('artifact lacks verified pinned BF16 vision RoPE initialization evidence.');

  const userInputs = artifact.mainArgumentOrder.filter(({sourceKind}) => sourceKind === 'USER_INPUT');
  if (
    userInputs.length !== INPUTS.length ||
    userInputs.some(({name, dtype}, index) => name !== INPUTS[index]!.name || dtype !== INPUTS[index]!.dtype)
  )
    invalid('artifact user-input order or dtype differs from the six-input TVM ABI.');

  const [pixel, grid, indices, weights, positions, cuSeqlens] = userInputs;
  const imageRows = scaledSymbol(pixel?.relaxShape?.[0], 4);
  if (
    !imageRows ||
    pixel?.relaxShape?.[1] !== 'T.int64(1536)' ||
    grid?.relaxShape?.[0] !== 'T.int64(1)' ||
    grid.relaxShape[1] !== 'T.int64(3)' ||
    indices?.relaxShape?.[0] !== pixel.relaxShape[0] ||
    indices.relaxShape[1] !== 'T.int64(4)' ||
    weights?.relaxShape?.[0] !== pixel.relaxShape[0] ||
    weights.relaxShape[1] !== 'T.int64(4)' ||
    scaledSymbol(positions?.relaxShape?.[0], 1) !== imageRows ||
    positions?.relaxShape?.[1] !== 'T.int64(8)' ||
    cuSeqlens?.relaxShape?.[0] !== 'T.int64(2)'
  )
    invalid('artifact Relax shapes do not describe dynamic [P=4*S] single-image inputs.');

  const names = new Set<string>();
  for (const argument of artifact.mainArgumentOrder) {
    if (!argument.name || names.has(argument.name)) invalid('artifact argument names must be unique.');
    names.add(argument.name);
    if (argument.sourceKind !== 'USER_INPUT' && argument.sourceKind !== 'PARAMETER')
      invalid('artifact contains an unsupported main argument source.');
  }
  return artifact.mainArgumentOrder;
}

function assertParameterTensors(
  arguments_: readonly Qwen35VisionArtifactArgument[],
  parameterTensors: ReadonlyMap<string, Tensor>
): void {
  const parameters = arguments_.filter(({sourceKind}) => sourceKind === 'PARAMETER');
  if (
    !parameterTensors ||
    typeof parameterTensors.size !== 'number' ||
    typeof parameterTensors.get !== 'function' ||
    parameterTensors.size !== parameters.length
  )
    invalid('loaded parameter tensors do not match the injected artifact ABI.');

  for (const argument of parameters) {
    const tensor = parameterTensors.get(argument.name) as RuntimeTensor | undefined;
    const expectedShape = argument.shape;
    if (
      !tensor ||
      !Array.isArray(expectedShape) ||
      expectedShape.some((dimension) => !Number.isSafeInteger(dimension) || dimension < 1) ||
      !Array.isArray(tensor.shape) ||
      tensor.shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension < 1) ||
      tensor.dtype !== argument.dtype
    )
      invalid('loaded vision parameter ' + argument.name + ' has invalid shape or dtype metadata.');
    assertShape(tensor.shape, expectedShape, 'loaded vision parameter ' + argument.name);
  }
}

function assertGeometry(
  geometry: Qwen35PerImageVisionGeometry,
  patchRowsLimit: Readonly<{minimum: number; maximum: number}>,
  imageIndex: number
): void {
  const patchRows = geometry.patchRows;
  const [frames, height, width] = geometry.gridThw;
  if (
    frames !== 1 ||
    !Number.isSafeInteger(patchRows) ||
    patchRows < patchRowsLimit.minimum ||
    patchRows > patchRowsLimit.maximum ||
    patchRows !== height * width ||
    patchRows % 4 !== 0 ||
    geometry.featureRows !== patchRows / 4
  )
    invalid('image ' + imageIndex + ' lies outside the artifact patch-row contract.');

  assertShape(geometry.gridThwInputShape, [1, 3], 'grid_thw');
  assertShape(geometry.interpIndicesShape, [patchRows, 4], 'interp_indices');
  assertShape(geometry.interpWeightsShape, [patchRows, 4], 'interp_weights');
  assertShape(geometry.positionIdsShape, [geometry.featureRows, 8], 'position_ids_by_token');
  assertShape(geometry.cuSeqlensShape, [2], 'cu_seqlens');
  if (
    geometry.gridThwInput.length !== 3 ||
    geometry.gridThwInput[0] !== frames ||
    geometry.gridThwInput[1] !== height ||
    geometry.gridThwInput[2] !== width ||
    geometry.interpIndices.length !== patchRows * 4 ||
    geometry.interpWeights.length !== patchRows * 4 ||
    geometry.positionIdsByToken.length !== geometry.featureRows * 8 ||
    geometry.cuSeqlens[0] !== 0 ||
    geometry.cuSeqlens[1] !== patchRows
  )
    invalid('image ' + imageIndex + ' geometry arrays differ from their declared shapes.');
}

function imageInputs(input: Qwen35GraphInputRecord, imageIndex: number, rowOffset: number): VisionInputs {
  const image = input.imageInput;
  const geometry = input.visionGeometry[imageIndex]!;
  const patchRows = geometry.patchRows;
  const imageGrid = image?.imageGridThw[imageIndex];
  if (!image || !imageGrid || imageGrid.some((value, index) => value !== geometry.gridThw[index]))
    invalid('image processor grid and vision geometry differ.');

  return {
    pixel_values: {
      shape: [patchRows, PATCH_WIDTH],
      dtype: 'float32',
      data: image.pixelValues.subarray(rowOffset * PATCH_WIDTH, (rowOffset + patchRows) * PATCH_WIDTH)
    },
    grid_thw: {shape: [1, 3], dtype: 'int32', data: geometry.gridThwInput},
    interp_indices: {shape: [patchRows, 4], dtype: 'int32', data: geometry.interpIndices},
    interp_weights: {shape: [patchRows, 4], dtype: 'float32', data: geometry.interpWeights},
    position_ids_by_token: {
      shape: [geometry.featureRows, 8],
      dtype: 'int32',
      data: geometry.positionIdsByToken
    },
    cu_seqlens: {shape: [2], dtype: 'int32', data: geometry.cuSeqlens}
  };
}

function dataBytes(data: Int32Array | Float32Array): Uint8Array {
  return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
}

function outputTensor(value: unknown): RuntimeTensor {
  if (!value || typeof value !== 'object') invalid('TVM main returned no feature tensor.');
  const candidate = value as Partial<RuntimeTensor>;
  if (
    !Array.isArray(candidate.shape) ||
    typeof candidate.dtype !== 'string' ||
    typeof candidate.copyFrom !== 'function' ||
    typeof candidate.toRawBytes !== 'function'
  )
    invalid('TVM main returned an invalid feature tensor.');
  return candidate as RuntimeTensor;
}

function unwrapOutput(value: unknown): RuntimeTensor {
  if (Array.isArray(value)) {
    if (value.length !== 1) invalid('TVM main must return exactly one feature tensor.');
    return outputTensor(value[0]);
  }
  if (value && typeof value === 'object') {
    const array = value as {size?: unknown; get?: unknown};
    if (typeof array.size === 'function' && typeof array.get === 'function') {
      if (Reflect.apply(array.size, value, []) !== 1) invalid('TVM main must return exactly one feature tensor.');
      return outputTensor(Reflect.apply(array.get, value, [0]));
    }
  }
  return outputTensor(value);
}

function float32FromBytes(bytes: Uint8Array): Float32Array {
  if (bytes.byteLength % 4 !== 0) invalid('TVM float32 feature output has an invalid byte length.');
  const values = new Float32Array(bytes.byteLength / 4);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < values.length; index += 1) values[index] = view.getFloat32(index * 4, true);
  return values;
}

function widenBf16(bytes: Uint8Array): Float32Array {
  if (bytes.byteLength % 2 !== 0) invalid('TVM BF16 feature output has an invalid byte length.');
  const values = new Float32Array(bytes.byteLength / 2);
  const output = new DataView(values.buffer);
  const input = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < values.length; index += 1)
    output.setUint32(index * 4, input.getUint16(index * 2, true) << 16, true);
  return values;
}

async function runImage(
  runtime: Qwen35VisionFeatureRunnerOptions['runtime'],
  main: ReturnType<VirtualMachine['getFunction']>,
  argumentOrder: readonly Qwen35VisionArtifactArgument[],
  parameterTensors: ReadonlyMap<string, Tensor>,
  inputs: VisionInputs,
  featureRows: number,
  precision: Qwen35VisionActivationPrecision,
  signal?: AbortSignal | null
): Promise<Float32Array> {
  signal?.throwIfAborted();
  runtime.beginScope();
  let failure: unknown;
  let failed = false;
  let result: Float32Array | undefined;
  try {
    const webgpu = runtime.webgpu();
    const inputTensors = new Map<string, Tensor>();
    for (const input of INPUTS) {
      signal?.throwIfAborted();
      const value = inputs[input.name];
      const tensor = runtime.empty(value.shape, value.dtype, webgpu);
      tensor.copyFromRawBytes(dataBytes(value.data));
      inputTensors.set(input.name, tensor);
    }
    const args = argumentOrder.map((argument) => {
      if (argument.sourceKind === 'USER_INPUT') {
        const input = inputTensors.get(argument.name);
        if (!input) invalid('missing TVM user input: ' + argument.name + '.');
        return input;
      }
      const parameter = parameterTensors.get(argument.name);
      if (!parameter) invalid('missing loaded TVM vision parameter: ' + argument.name + '.');
      return parameter;
    });
    const output = unwrapOutput(main(...args));
    signal?.throwIfAborted();
    assertShape(output.shape, [featureRows, FEATURE_WIDTH], 'TVM visual feature output');
    if (output.dtype !== 'float32' && !(precision === 'bfloat16' && output.dtype === 'bfloat16'))
      invalid('TVM feature output dtype differs from the source precision.');

    const host = runtime.empty([...output.shape], output.dtype, runtime.cpu());
    host.copyFrom(output);
    await webgpu.sync();
    signal?.throwIfAborted();
    const raw = host.toRawBytes();
    const values = output.dtype === 'bfloat16' ? widenBf16(raw) : float32FromBytes(raw);
    if (!isQwen35FiniteActivationArray(values, precision))
      invalid(
        precision === 'float32'
          ? 'feature output must contain finite float32 values.'
          : 'feature output must preserve source BF16 values widened exactly to float32.'
      );
    result = values;
  } catch (cause) {
    failure = cause;
    failed = true;
  } finally {
    try {
      runtime.endScope();
    } catch (cause) {
      if (!failed) failure = cause;
      failed = true;
    }
  }
  if (failed) throw failure;
  return result!;
}

async function runImages(
  runtime: Qwen35VisionFeatureRunnerOptions['runtime'],
  main: ReturnType<VirtualMachine['getFunction']>,
  argumentOrder: readonly Qwen35VisionArtifactArgument[],
  parameterTensors: ReadonlyMap<string, Tensor>,
  input: Qwen35GraphInputRecord,
  patchRowsLimit: Readonly<{minimum: number; maximum: number}>,
  precision: Qwen35VisionActivationPrecision,
  signal?: AbortSignal | null
): Promise<readonly Float32Array[]> {
  signal?.throwIfAborted();
  if (
    !input ||
    !Number.isSafeInteger(input.imageCount) ||
    input.imageCount < 1 ||
    !input.imageInput ||
    input.visionGeometry.length !== input.imageCount ||
    input.imageInput.imageGridThw.length !== input.imageCount ||
    input.imageGridThw.length !== input.imageCount * 3
  )
    invalid('prepared input with source-ordered images is required.');

  const image = input.imageInput;
  if (
    image.pixelShape[1] !== PATCH_WIDTH ||
    image.pixelValues.length !== image.pixelShape[0] * PATCH_WIDTH ||
    !Number.isSafeInteger(image.pixelShape[0])
  )
    invalid('processor pixel_values do not match their declared shape.');

  let totalPatchRows = 0;
  for (let index = 0; index < input.imageCount; index += 1) {
    const geometry = input.visionGeometry[index]!;
    assertGeometry(geometry, patchRowsLimit, index);
    totalPatchRows += geometry.patchRows;
    if (!Number.isSafeInteger(totalPatchRows)) invalid('total image patch rows exceed the safe integer range.');
    for (let field = 0; field < 3; field += 1)
      if (input.imageGridThw[index * 3 + field] !== geometry.gridThw[field])
        invalid('packed image grid order differs from source geometry.');
  }
  if (image.pixelShape[0] !== totalPatchRows) invalid('pixel_values rows do not cover every source image.');

  const outputs: Float32Array[] = [];
  let rowOffset = 0;
  for (let index = 0; index < input.imageCount; index += 1) {
    signal?.throwIfAborted();
    const geometry = input.visionGeometry[index]!;
    outputs.push(
      await runImage(
        runtime,
        main,
        argumentOrder,
        parameterTensors,
        imageInputs(input, index, rowOffset),
        geometry.featureRows,
        precision,
        signal
      )
    );
    signal?.throwIfAborted();
    rowOffset += geometry.patchRows;
  }
  return Object.freeze(outputs);
}

export function createQwen35VisionFeatureRunner(options: Qwen35VisionFeatureRunnerOptions): Qwen35VisionFeatureRunner {
  const argumentOrder = assertArtifact(options.artifact);
  const precision = qwen35VisionActivationPrecision(options.artifact);
  assertParameterTensors(argumentOrder, options.parameterTensors);

  const runtime = options.runtime;
  const virtualMachine = runtime.createVirtualMachine(runtime.webgpu());
  const main = virtualMachine.getFunction('main');
  let queued: Promise<unknown> = Promise.resolve();
  return Object.freeze({
    run(input: Qwen35GraphInputRecord, signal?: AbortSignal | null) {
      const task = queued.then(() =>
        runImages(
          runtime,
          main,
          argumentOrder,
          options.parameterTensors,
          input,
          options.artifact.shapeContract.patchRows,
          precision,
          signal
        )
      );
      queued = task.catch(() => {});
      return task;
    }
  });
}
