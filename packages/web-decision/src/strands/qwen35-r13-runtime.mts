import {sha256} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import type {Qwen35GraphInputChunk, Qwen35GraphInputRecord} from './qwen35-graph-input.mts';
import {
  prepareQwen35R13GraphInputTuple,
  QWEN35_R13_RUNTIME_INPUT_NAMES,
  type Qwen35PreEmbedSources,
  type Qwen35R13GraphInputTuple
} from './qwen35-r13-pre-embed-bridge.mts';
import {readQwen35GraphProbabilityRows} from './qwen35-graph-output.mts';
import {QWEN35_R19_TEXT} from './qwen35-r19-text-contract.mts';
import {QWEN35_GRAPH_MAX_BATCH} from './qwen35-protocol.mts';
import type {Qwen35GraphRunner} from './qwen35-session-adapter.mts';
import type {Qwen35WordGraphBinding} from './qwen35-word-graph-input.mts';
import type {Qwen35WordTensor} from './qwen35-assets.mjs';
import {orderQwen35SourceTensorReads, type Qwen35PinnedSource} from './qwen35-source.mts';
import type {DLDevice, Instance, Tensor, TVMImportObject, TVMLibraryProvider} from '../runtime/tvm-types.js';

const R13_WEIGHT_COUNT = 697;
const R13_WEIGHT_BINDINGS_SHA256 = '0107466fdfe1eaa8ab35caa3f22f0a382838c58c9ebb852270a5001a5fd987dd';
const R13_EXTERNAL_ARGUMENT_NAMES_SHA256 = 'f2a0e40d162bcd064e9f924c7ee91f50dd91604ed362b0a2e9cab9187d4e313f';
const R13_FIXTURE_SHA256 = 'bde2174935a064392cadea8331c03752980fc99ef3f0de0807d4499b994b105b';
const R13_WASM_SHA256 = '8d86864fc8235d300c7b8a13f5f5a3c735c52a9a8a564855542039125f1642b5';
const TEXT_GRAPH_FULL_PARAMETER_NAMES_SHA256 = '33002edb71b198a7a3e336a6daf7db952b5d04f97ae015174a705b1ae3943f36';
const R13_WASM_BYTES = 6455483;
const R13_FIXTURE_FORMAT = 'qwen35-full-pointer-tvmjs-fixture-v1';
const R18_FIXTURE_SHA256 = '26cdde995544757d6c395bfe1a427e0008900248be267a0eecc9590dd3491954';
const R18_WASM_SHA256 = '7510129f71d1513c4ffe08c3630abba4d44bf14ae0f0ea8352b32f07a2e59f9d';
const R18_WASM_BYTES = 6_460_801;
const R18_SOURCE_REPORT_SHA256 = '4d159b0397305b0e45173125ef52c1fa05aa4af341889c7b313e783edb345961';
const R18_SAVED_RELAX_SHA256 = '1c4ba13625ff9b3cb152c7b97e505406d6f8cbacbaf210f378e2337dec00c491';
const R18_STRANDS_SOURCE_REVISION = '13c05149265dc404c09d2469636f71b1320e1713';
const R19_SAVED_RELAX_SHA256 = '624cc890ac1ffb92caac65fd27b9c924f2d0b71bb009325bf16aa77fe8f6830a';
const R19_STRANDS_SOURCE_REVISION = '13c05149265dc404c09d2469636f71b1320e1713';
const R13_FIRST_PARAMETER_NAMES = Object.freeze([
  'inputs_embeds',
  'attention_mask',
  'position_ids',
  'option_indices',
  'temperatures',
  'n_slots'
]);

export interface Qwen35R13OrderedWeightBinding extends Qwen35WordGraphBinding {
  readonly index: number;
}

export type Qwen35R13Tensor = Tensor;
export type Qwen35R13Device = DLDevice;
export type Qwen35R13Runtime = Pick<
  Instance,
  'initWebGPU' | 'beginScope' | 'endScope' | 'dispose' | 'webgpu' | 'cpu' | 'empty' | 'createVirtualMachine'
> & {makeShapeTuple?(dimensions: number[]): unknown};

export interface Qwen35R13TvmJs {
  instantiate(
    wasm: ArrayBuffer,
    imports: TVMImportObject,
    logger: (message: string) => void
  ): PromiseLike<Qwen35R13Runtime>;
  createPolyfillWASI(): TVMLibraryProvider;
}

export interface Qwen35R13RuntimeHandle {
  readonly runtime: Qwen35R13Runtime;
  readonly device: Qwen35R13Device;
  readonly main: (...args: unknown[]) => unknown;
  readonly hasSourcePrefix?: boolean;
  dispose(): Promise<void>;
}

export interface Qwen35R13RuntimeOptions {
  readonly tvmjs: Qwen35R13TvmJs;
  readonly fixtureBytes: Uint8Array;
  readonly wasmBytes: Uint8Array;
  readonly gpuDevice: GPUDevice;
  readonly logger?: ((message: string) => void) | undefined;
}

export type Qwen35R18OrderedWeightBinding = Qwen35R13OrderedWeightBinding;
export type Qwen35R18Tensor = Qwen35R13Tensor;
export type Qwen35R18Device = Qwen35R13Device;
export type Qwen35R18Runtime = Qwen35R13Runtime;
export type Qwen35R18TvmJs = Qwen35R13TvmJs;
export type Qwen35R18RuntimeHandle = Qwen35R13RuntimeHandle;
export type Qwen35R18RuntimeOptions = Qwen35R13RuntimeOptions;
export type Qwen35R18RuntimeWeight = Qwen35R13RuntimeWeight;
export type Qwen35R18RuntimeWeights = Qwen35R13RuntimeWeights;
export type Qwen35R18GraphRunnerOptions = Qwen35R13GraphRunnerOptions;
export type Qwen35R18GraphRunner = Qwen35R13GraphRunner;
export type Qwen35R18WeightLoadOptions = Qwen35R13WeightLoadOptions;
export type Qwen35R18WeightLoader = Qwen35R13WeightLoader;

export interface Qwen35R13RuntimeWeight {
  readonly binding: Qwen35GraphWeightBinding;
  readonly tensor: Qwen35R13Tensor;
}

export interface Qwen35R13RuntimeWeights {
  readonly bindingsSha256: string;
  readonly argumentNamesSha256: string;
  readonly values: readonly Qwen35R13RuntimeWeight[];
}

export interface Qwen35R13GraphRunnerOptions {
  readonly handle: Qwen35R13RuntimeHandle;
  readonly source: Qwen35PinnedSource;
  readonly bindings: readonly Qwen35GraphWeightBinding[];
  readonly expectedBindingHashes?:
    | Readonly<{
        readonly bindingsSha256: string;
        readonly argumentNamesSha256: string;
      }>
    | undefined;
  readonly resolveInputSources: (
    chunk: Qwen35GraphInputChunk,
    input: Qwen35GraphInputRecord,
    signal?: AbortSignal | null
  ) => Qwen35PreEmbedSources | PromiseLike<Qwen35PreEmbedSources>;
  readonly signal?: AbortSignal | null | undefined;
  readonly onProgress?: ((bytes: number) => void) | undefined;
  readonly loadWeights?: Qwen35R13WeightLoader | undefined;
}

export interface Qwen35R13GraphRunner extends Qwen35GraphRunner {
  runInputs(
    input: Qwen35R13GraphInputTuple,
    signal?: AbortSignal | null,
    sourcePrefixLength?: number
  ): Promise<readonly Float32Array[]>;
  dispose(): Promise<void>;
}

export interface Qwen35R13WeightLoadOptions {
  readonly runtime: Pick<Qwen35R13Runtime, 'empty'>;
  readonly device: Qwen35R13Device;
  readonly source: Qwen35PinnedSource;
  readonly bindings: readonly Qwen35GraphWeightBinding[];
  readonly expectedBindingHashes?:
    | Readonly<{
        readonly bindingsSha256: string;
        readonly argumentNamesSha256: string;
      }>
    | undefined;
  readonly signal?: AbortSignal | null | undefined;
  readonly onProgress?: ((bytes: number) => void) | undefined;
}

export type Qwen35R13WeightLoader = (options: Qwen35R13WeightLoadOptions) => Promise<Qwen35R13RuntimeWeights>;

export interface Qwen35PackedGraphWeightBinding extends Omit<Qwen35R13OrderedWeightBinding, 'relaxDtype'> {
  readonly relaxDtype: 'float32' | 'int32';
  readonly relaxShape: readonly number[];
}

export interface Qwen35PackedGraphWeightPayload {
  readonly bytes: Uint8Array;
  readonly dtype: 'float32' | 'int32';
  readonly shape: readonly number[];
}

export type Qwen35GraphWeightBinding = Qwen35R13OrderedWeightBinding | Qwen35PackedGraphWeightBinding;

function isPackedGraphBinding(binding: Qwen35GraphWeightBinding): binding is Qwen35PackedGraphWeightBinding {
  return 'relaxShape' in binding;
}

function invalid(message: string): WebDecisionError {
  return new WebDecisionError('INVALID_ASSET', message);
}

const verifiedRuntimeHandles = new WeakSet<object>();
const claimedRuntimeHandles = new WeakSet<object>();
const disposedRuntimeHandles = new WeakSet<object>();

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

interface TextArtifactPin {
  readonly generation: 'R13' | 'R18' | 'R19';
  readonly fixtureSha256: string;
  readonly wasmSha256: string;
  readonly wasmBytes: number;
  readonly sourceGraphGeneration?: 'R18';
  readonly sourceReportSha256?: string;
  readonly savedRelaxSha256?: string;
  readonly strandsSourceRevision?: string;
  readonly runtimeInputNames?: readonly string[];
  readonly parameterNamesSha256?: string;
}

async function verifyTextArtifact(
  fixtureBytes: Uint8Array,
  wasmBytes: Uint8Array,
  pin: TextArtifactPin
): Promise<void> {
  if (
    !(fixtureBytes instanceof Uint8Array) ||
    !(wasmBytes instanceof Uint8Array) ||
    (await sha256(fixtureBytes)) !== pin.fixtureSha256 ||
    (await sha256(wasmBytes)) !== pin.wasmSha256 ||
    wasmBytes.byteLength !== pin.wasmBytes
  )
    throw invalid(`Qwen3.5 ${pin.generation} TVM.js artifact bytes differ from the pinned fixture.`);

  let fixtureValue: unknown;
  try {
    fixtureValue = JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(fixtureBytes)) as unknown;
  } catch (cause) {
    throw new WebDecisionError('INVALID_ASSET', `Could not parse the Qwen3.5 ${pin.generation} TVM.js fixture.`, {
      cause
    });
  }
  if (!isRecord(fixtureValue)) throw invalid(`Invalid Qwen3.5 ${pin.generation} TVM.js fixture.`);
  const wasm = fixtureValue.wasm;
  const names = fixtureValue.parameterOrder;
  const generationMismatch =
    pin.sourceGraphGeneration !== undefined && fixtureValue.sourceGraphGeneration !== pin.sourceGraphGeneration;
  const sourceEvidence = fixtureValue.sourceEvidence;
  const runtimeInputNames = pin.runtimeInputNames ?? R13_FIRST_PARAMETER_NAMES;
  const parameterCount = R13_WEIGHT_COUNT + runtimeInputNames.length;
  const sourceEvidenceMismatch =
    pin.sourceReportSha256 !== undefined &&
    (!isRecord(sourceEvidence) ||
      sourceEvidence.sourceReportSha256 !== pin.sourceReportSha256 ||
      sourceEvidence.savedRelaxSha256 !== pin.savedRelaxSha256 ||
      sourceEvidence.strandsDeciderSourceRevision !== pin.strandsSourceRevision);
  if (
    fixtureValue.format !== R13_FIXTURE_FORMAT ||
    generationMismatch ||
    sourceEvidenceMismatch ||
    fixtureValue.mainParameterCount !== parameterCount ||
    fixtureValue.externalWeightCount !== R13_WEIGHT_COUNT ||
    !Array.isArray(fixtureValue.runtimeInputNames) ||
    JSON.stringify(fixtureValue.runtimeInputNames) !== JSON.stringify(runtimeInputNames) ||
    !isRecord(wasm) ||
    wasm.file !== 'qwen35-text-pointer.webgpu.wasm' ||
    wasm.bytes !== pin.wasmBytes ||
    wasm.sha256 !== pin.wasmSha256 ||
    !Array.isArray(names) ||
    names.length !== parameterCount
  )
    throw invalid(`Qwen3.5 ${pin.generation} TVM.js fixture does not describe the pinned graph ABI.`);

  const parameterNames: string[] = [];
  for (let index = 0; index < names.length; index += 1) {
    const value = names[index];
    if (
      !isRecord(value) ||
      value.index !== index ||
      typeof value.name !== 'string' ||
      typeof value.external !== 'boolean'
    )
      throw invalid(`Qwen3.5 ${pin.generation} graph parameter order is malformed.`);
    if (value.external !== index >= runtimeInputNames.length)
      throw invalid(`Qwen3.5 ${pin.generation} graph runtime and external parameter slots are out of order.`);
    parameterNames.push(value.name);
  }
  if (
    JSON.stringify(parameterNames.slice(0, runtimeInputNames.length)) !== JSON.stringify(runtimeInputNames) ||
    (await sha256(new TextEncoder().encode(parameterNames.join('\n')))) !==
      (pin.parameterNamesSha256 ?? TEXT_GRAPH_FULL_PARAMETER_NAMES_SHA256)
  )
    throw invalid(`Qwen3.5 ${pin.generation} graph parameter order differs from its pinned fixture.`);
}

async function verifyR13Artifact(fixtureBytes: Uint8Array, wasmBytes: Uint8Array): Promise<void> {
  await verifyTextArtifact(fixtureBytes, wasmBytes, {
    generation: 'R13',
    fixtureSha256: R13_FIXTURE_SHA256,
    wasmSha256: R13_WASM_SHA256,
    wasmBytes: R13_WASM_BYTES
  });
}

async function verifyR18Artifact(fixtureBytes: Uint8Array, wasmBytes: Uint8Array): Promise<void> {
  await verifyTextArtifact(fixtureBytes, wasmBytes, {
    generation: 'R18',
    fixtureSha256: R18_FIXTURE_SHA256,
    wasmSha256: R18_WASM_SHA256,
    wasmBytes: R18_WASM_BYTES,
    sourceGraphGeneration: 'R18',
    sourceReportSha256: R18_SOURCE_REPORT_SHA256,
    savedRelaxSha256: R18_SAVED_RELAX_SHA256,
    strandsSourceRevision: R18_STRANDS_SOURCE_REVISION
  });
}

async function verifyR19Artifact(fixtureBytes: Uint8Array, wasmBytes: Uint8Array): Promise<void> {
  await verifyTextArtifact(fixtureBytes, wasmBytes, {
    generation: 'R19',
    fixtureSha256: QWEN35_R19_TEXT.fixture.sha256,
    wasmSha256: QWEN35_R19_TEXT.wasm.sha256,
    wasmBytes: QWEN35_R19_TEXT.wasm.bytes,
    runtimeInputNames: QWEN35_R19_TEXT.inputNames,
    parameterNamesSha256: QWEN35_R19_TEXT.parameterNamesSha256
  });
  let fixture: unknown;
  try {
    fixture = JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(fixtureBytes)) as unknown;
  } catch (cause) {
    throw new WebDecisionError('INVALID_ASSET', 'Could not parse the Qwen3.5 R19 TVM.js fixture.', {cause});
  }
  const sourceEvidence = isRecord(fixture) ? fixture.sourceEvidence : undefined;
  const source = isRecord(sourceEvidence) ? sourceEvidence.source : undefined;
  if (
    !isRecord(fixture) ||
    fixture.savedRelaxSha256 !== R19_SAVED_RELAX_SHA256 ||
    !isRecord(sourceEvidence) ||
    sourceEvidence.relaxScriptSha256 !== R19_SAVED_RELAX_SHA256 ||
    !isRecord(source) ||
    source.strandsDeciderSourceRevision !== R19_STRANDS_SOURCE_REVISION
  )
    throw invalid('Qwen3.5 R19 fixture differs from its pinned source graph.');
}

function verifyShapeTupleSupport(runtime: Qwen35R13Runtime, hasSourcePrefix: boolean): void {
  if (hasSourcePrefix && typeof runtime.makeShapeTuple !== 'function')
    throw invalid('The Qwen3.5 runtime must support shape tuples for the source prefix.');
}

async function instantiateVerifiedTextRuntime(
  {tvmjs, wasmBytes, gpuDevice, logger = () => {}}: Qwen35R13RuntimeOptions,
  hasSourcePrefix = false
): Promise<Qwen35R13RuntimeHandle> {
  let runtime: Qwen35R13Runtime | undefined;
  let scopeOpen = false;
  try {
    const wasm = new ArrayBuffer(wasmBytes.byteLength);
    new Uint8Array(wasm).set(wasmBytes);
    runtime = await tvmjs.instantiate(wasm, tvmjs.createPolyfillWASI(), logger);
    runtime.initWebGPU(gpuDevice);
    runtime.beginScope();
    scopeOpen = true;
    const device = runtime.webgpu();
    const main = runtime.createVirtualMachine(device).getFunction('main');
    verifyShapeTupleSupport(runtime, hasSourcePrefix);
    let disposed = false;
    const handle: Qwen35R13RuntimeHandle = Object.freeze({
      runtime,
      device,
      main,
      hasSourcePrefix,
      async dispose() {
        if (disposed) return;
        disposed = true;
        disposedRuntimeHandles.add(handle);
        let failure: unknown;
        try {
          await device.sync();
        } catch (cause) {
          failure = cause;
        }
        if (scopeOpen) {
          scopeOpen = false;
          try {
            runtime!.endScope();
          } catch (cause) {
            if (failure === undefined) failure = cause;
          }
        }
        try {
          runtime!.dispose();
        } catch (cause) {
          if (failure === undefined) failure = cause;
        }
        try {
          gpuDevice.destroy();
        } catch (cause) {
          if (failure === undefined) failure = cause;
        }
        if (failure !== undefined) throw failure;
      }
    });
    verifiedRuntimeHandles.add(handle);
    return handle;
  } catch (cause) {
    if (runtime) {
      if (scopeOpen) {
        try {
          runtime.endScope();
        } catch {}
      }
      try {
        runtime.dispose();
      } catch {}
    }
    try {
      gpuDevice.destroy();
    } catch {}
    throw cause;
  }
}

/** Instantiate only the exact compiled R13 artifact described by its pinned fixture. */
export async function instantiatePinnedQwen35R13Runtime(
  options: Qwen35R13RuntimeOptions
): Promise<Qwen35R13RuntimeHandle> {
  try {
    await verifyR13Artifact(options.fixtureBytes, options.wasmBytes);
  } catch (cause) {
    try {
      options.gpuDevice.destroy();
    } catch {}
    throw cause;
  }
  return instantiateVerifiedTextRuntime(options);
}

/** Instantiate the explicit R18 text graph under its own fixture and WASM pins. */
export async function instantiatePinnedQwen35R18Runtime(
  options: Qwen35R18RuntimeOptions
): Promise<Qwen35R18RuntimeHandle> {
  try {
    await verifyR18Artifact(options.fixtureBytes, options.wasmBytes);
  } catch (cause) {
    try {
      options.gpuDevice.destroy();
    } catch {}
    throw cause;
  }
  return instantiateVerifiedTextRuntime(options);
}

/** Instantiate the exact packed R19 graph under its own fixture and WASM pins. */
export async function instantiatePinnedQwen35R19Runtime(
  options: Qwen35R18RuntimeOptions
): Promise<Qwen35R18RuntimeHandle> {
  try {
    await verifyR19Artifact(options.fixtureBytes, options.wasmBytes);
  } catch (cause) {
    try {
      options.gpuDevice.destroy();
    } catch {}
    throw cause;
  }
  return instantiateVerifiedTextRuntime(options, QWEN35_R19_TEXT.sourcePrefix);
}

function canonicalBindings(bindings: readonly Qwen35GraphWeightBinding[]): string {
  return JSON.stringify(
    bindings.map((binding) => ({
      index: binding.index,
      asset: binding.asset,
      sourceName: binding.sourceName,
      targetName: binding.targetName,
      relaxArgumentName: binding.relaxArgumentName,
      shape: binding.shape,
      sourceDtype: binding.sourceDtype,
      loadedDtype: binding.loadedDtype,
      relaxDtype: binding.relaxDtype,
      loadedPayloadSha256: binding.loadedPayloadSha256 ?? null,
      ...('relaxShape' in binding ? {relaxShape: binding.relaxShape} : {})
    }))
  );
}

function snapshotBindingSet(bindings: readonly Qwen35GraphWeightBinding[]): readonly Qwen35GraphWeightBinding[] {
  if (!Array.isArray(bindings) || bindings.length !== R13_WEIGHT_COUNT)
    throw invalid(`The pinned Qwen3.5 R13 graph requires ${R13_WEIGHT_COUNT} external weights.`);
  if (
    bindings.some(
      (binding, index) =>
        !binding ||
        binding.index !== index ||
        typeof binding.sourceName !== 'string' ||
        typeof binding.targetName !== 'string' ||
        typeof binding.relaxArgumentName !== 'string' ||
        !Array.isArray(binding.shape) ||
        binding.shape.some((dimension: unknown) => !Number.isSafeInteger(dimension) || (dimension as number) < 1) ||
        !['float32', 'int32'].includes(binding.relaxDtype) ||
        (!isPackedGraphBinding(binding) && binding.relaxDtype !== 'float32') ||
        (isPackedGraphBinding(binding) &&
          (!Array.isArray(binding.relaxShape) ||
            binding.relaxShape.length === 0 ||
            binding.relaxShape.some(
              (dimension: unknown) => !Number.isSafeInteger(dimension) || (dimension as number) < 1
            )))
    )
  )
    throw invalid('Qwen3.5 R13 external weights are not in pinned graph order.');
  return Object.freeze(
    bindings.map((binding) =>
      Object.freeze({
        ...binding,
        shape: Object.freeze([...binding.shape]),
        ...('relaxShape' in binding ? {relaxShape: Object.freeze([...binding.relaxShape])} : {})
      })
    )
  );
}

async function bindingHashes(bindings: readonly Qwen35GraphWeightBinding[]): Promise<{
  bindingsSha256: string;
  argumentNamesSha256: string;
}> {
  const bindingsSha256 = await sha256(new TextEncoder().encode(canonicalBindings(bindings)));
  const argumentNamesSha256 = await sha256(
    new TextEncoder().encode(bindings.map(({relaxArgumentName}) => relaxArgumentName).join('\n'))
  );
  return {bindingsSha256, argumentNamesSha256};
}

function dimensionsMatch(actual: readonly number[], expected: readonly number[]): boolean {
  return actual.length === expected.length && actual.every((dimension, index) => dimension === expected[index]);
}

function isR13Tensor(value: unknown): value is Qwen35R13Tensor {
  if (typeof value !== 'object' || value === null) return false;
  const shape = Reflect.get(value, 'shape');
  const device = Reflect.get(value, 'device');
  return (
    Array.isArray(shape) &&
    shape.every(Number.isSafeInteger) &&
    typeof Reflect.get(value, 'dtype') === 'string' &&
    typeof Reflect.get(value, 'copyFrom') === 'function' &&
    typeof Reflect.get(value, 'copyFromRawBytes') === 'function' &&
    typeof Reflect.get(value, 'toArray') === 'function' &&
    typeof device === 'object' &&
    device !== null &&
    Number.isSafeInteger(Reflect.get(device, 'deviceType')) &&
    Number.isSafeInteger(Reflect.get(device, 'deviceId')) &&
    typeof Reflect.get(device, 'sync') === 'function'
  );
}

function validateRuntimeTensor(tensor: Qwen35R13Tensor, shape: readonly number[], dtype: string, label: string): void {
  if (!tensor || tensor.dtype !== dtype || !Array.isArray(tensor.shape) || !dimensionsMatch(tensor.shape, shape))
    throw invalid(`Qwen3.5 R13 ${label} tensor does not match its pinned shape and dtype.`);
}

function findSourceTensor(
  source: Qwen35PinnedSource,
  binding: Pick<Qwen35R13OrderedWeightBinding, 'index' | 'asset' | 'sourceName' | 'shape' | 'sourceDtype'>
): Qwen35WordTensor {
  const tensor = source.manifest.assets[binding.asset]?.tensors.find(({name}) => name === binding.sourceName);
  const sourceDtype = tensor?.dtype === 'BF16' ? 'bfloat16' : 'float32';
  if (
    !tensor ||
    !dimensionsMatch(tensor.shape, binding.shape) ||
    sourceDtype !== binding.sourceDtype ||
    tensor.elementCount !== binding.shape.reduce((count, dimension) => count * dimension, 1)
  )
    throw invalid(`Qwen3.5 R13 source tensor differs from ABI slot ${binding.index}.`);
  return tensor;
}

function widenBf16(raw: Uint8Array, elementCount: number): Uint8Array {
  if (raw.byteLength !== elementCount * 2) throw invalid('Qwen3.5 source BF16 tensor has an invalid byte count.');
  const source = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const output = new Uint8Array(elementCount * 4);
  const result = new DataView(output.buffer);
  for (let index = 0; index < elementCount; index += 1)
    result.setUint32(index * 4, source.getUint16(index * 2, true) << 16, true);
  return output;
}

function f32ToBf16Rne(raw: Uint8Array, elementCount: number): Uint8Array {
  if (raw.byteLength !== elementCount * 4) throw invalid('Qwen3.5 source F32 tensor has an invalid byte count.');
  const source = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const output = new Uint8Array(elementCount * 2);
  const result = new DataView(output.buffer);
  for (let index = 0; index < elementCount; index += 1) {
    const bits = source.getUint32(index * 4, true);
    const upper = bits >>> 16;
    const exponent = bits & 0x7f800000;
    const mantissa = bits & 0x007fffff;
    const rounded = exponent === 0x7f800000 && mantissa !== 0 ? upper | 0x0040 : (bits + 0x7fff + (upper & 1)) >>> 16;
    result.setUint16(index * 2, rounded, true);
  }
  return output;
}

async function castProvenSourceBf16(
  raw: Uint8Array,
  elementCount: number,
  binding: Pick<Qwen35R13OrderedWeightBinding, 'sourceName' | 'loadedPayloadSha256'>
): Promise<Uint8Array> {
  if (!binding.loadedPayloadSha256 || !/^[0-9a-f]{64}$/.test(binding.loadedPayloadSha256))
    throw invalid(`Qwen3.5 base F32→BF16 source-cast proof is missing: ${binding.sourceName}.`);
  const rounded = f32ToBf16Rne(raw, elementCount);
  if ((await sha256(rounded)) !== binding.loadedPayloadSha256)
    throw invalid(`Qwen3.5 base F32→BF16 values differ from source-cast proof: ${binding.sourceName}.`);
  return rounded;
}

/** Materialize one original pinned Safetensors tensor with the loader's proven dtype transition. */
export async function materializeQwen35R13PinnedSourceWeight(
  source: Qwen35PinnedSource,
  binding: Qwen35R13OrderedWeightBinding,
  signal?: AbortSignal | null
): Promise<Uint8Array> {
  signal?.throwIfAborted();
  const tensor = findSourceTensor(source, binding);
  const raw = await source.readTensor(binding.asset, binding.sourceName, {signal: signal ?? undefined});
  signal?.throwIfAborted();
  if (raw.byteLength !== tensor.sourceByteCount)
    throw invalid(`Qwen3.5 pinned source tensor has an invalid size: ${binding.sourceName}.`);

  if (binding.asset !== 'base' && binding.sourceDtype === 'float32' && binding.loadedDtype === 'float32') return raw;
  if (binding.asset === 'base' && binding.sourceDtype === 'bfloat16' && binding.loadedDtype === 'bfloat16')
    return widenBf16(raw, tensor.elementCount);
  if (binding.asset === 'base' && binding.sourceDtype === 'float32' && binding.loadedDtype === 'bfloat16') {
    const rounded = await castProvenSourceBf16(raw, tensor.elementCount, binding);
    return widenBf16(rounded, tensor.elementCount);
  }
  throw invalid(`Qwen3.5 source dtype path differs from the pinned loader: ${binding.sourceName}.`);
}

function validatePackedWeightShape(binding: Qwen35PackedGraphWeightBinding, elementCount: number): number {
  const graphElementCount = binding.relaxShape.reduce((count, dimension) => count * dimension, 1);
  if (
    !Number.isSafeInteger(elementCount) ||
    !Number.isSafeInteger(graphElementCount) ||
    binding.relaxShape.length === 0 ||
    binding.relaxShape.some((dimension) => !Number.isSafeInteger(dimension) || dimension < 1)
  )
    throw invalid(`Qwen3.5 packed graph shape is invalid: ${binding.sourceName}.`);
  if (binding.relaxDtype === 'int32') {
    if (binding.loadedDtype !== 'bfloat16' || elementCount % 2 !== 0 || graphElementCount * 4 !== elementCount * 2)
      throw invalid(`Qwen3.5 packed int32 ABI does not preserve BF16 bytes: ${binding.sourceName}.`);
    return graphElementCount;
  }
  if (
    graphElementCount !== elementCount ||
    binding.relaxShape.some((dimension, index) => dimension !== binding.shape[index])
  )
    throw invalid(`Qwen3.5 float32 graph shape differs from its source: ${binding.sourceName}.`);
  return graphElementCount;
}

/** Materialize one weight for the packed BF16 Relax ABI without widening packed parameters. */
export async function materializeQwen35PackedGraphSourceWeight(
  source: Qwen35PinnedSource,
  binding: Qwen35PackedGraphWeightBinding,
  signal?: AbortSignal | null
): Promise<Qwen35PackedGraphWeightPayload> {
  signal?.throwIfAborted();
  const tensor = findSourceTensor(source, binding);
  const elementCount = binding.shape.reduce((count, dimension) => count * dimension, 1);
  const graphElementCount = validatePackedWeightShape(binding, elementCount);

  const raw = await source.readTensor(binding.asset, binding.sourceName, {signal: signal ?? undefined});
  signal?.throwIfAborted();
  if (raw.byteLength !== tensor.sourceByteCount)
    throw invalid(`Qwen3.5 pinned source tensor has an invalid size: ${binding.sourceName}.`);

  let loaded = raw;
  if (binding.asset === 'base' && binding.loadedDtype === 'bfloat16') {
    if (binding.sourceDtype === 'float32') loaded = await castProvenSourceBf16(raw, elementCount, binding);
    else if (binding.sourceDtype !== 'bfloat16')
      throw invalid(`Qwen3.5 source dtype path differs from the packed loader: ${binding.sourceName}.`);
  } else if (
    (binding.asset !== 'lora' && binding.asset !== 'head') ||
    binding.sourceDtype !== 'float32' ||
    binding.loadedDtype !== 'float32'
  ) {
    throw invalid(`Qwen3.5 source dtype path differs from the packed loader: ${binding.sourceName}.`);
  }

  const widensBf16 = binding.relaxDtype !== 'int32' && binding.loadedDtype === 'bfloat16';
  const bytes = widensBf16 ? widenBf16(loaded, elementCount) : loaded;
  if (bytes.byteLength !== graphElementCount * 4)
    throw invalid(`Qwen3.5 packed graph byte count differs from its ABI: ${binding.sourceName}.`);
  return Object.freeze({bytes, dtype: binding.relaxDtype, shape: Object.freeze([...binding.relaxShape])});
}

/**
 * Upload the exact 697 verified source tensors required by the pinned R13 ABI.
 * Call inside the loader's owning TVM scope so the returned tensors remain live.
 */
export async function loadQwen35R13RuntimeWeights({
  runtime,
  device,
  source,
  bindings,
  expectedBindingHashes,
  signal,
  onProgress
}: Qwen35R13WeightLoadOptions): Promise<Qwen35R13RuntimeWeights> {
  const bindingSnapshot = snapshotBindingSet(bindings);
  const {bindingsSha256, argumentNamesSha256} = await bindingHashes(bindingSnapshot);
  const expectedHashes =
    expectedBindingHashes ??
    (bindingSnapshot.some(isPackedGraphBinding)
      ? undefined
      : {bindingsSha256: R13_WEIGHT_BINDINGS_SHA256, argumentNamesSha256: R13_EXTERNAL_ARGUMENT_NAMES_SHA256});
  signal?.throwIfAborted();
  if (
    !expectedHashes ||
    bindingsSha256 !== expectedHashes.bindingsSha256 ||
    argumentNamesSha256 !== expectedHashes.argumentNamesSha256
  )
    throw invalid('Qwen3.5 runtime weight bindings differ from the pinned graph ABI.');

  const sourceOrder = orderQwen35SourceTensorReads(source.manifest, bindingSnapshot);
  const valuesByIndex = new Map<number, Qwen35R13RuntimeWeight>();
  for (const binding of sourceOrder) {
    signal?.throwIfAborted();
    const payload = isPackedGraphBinding(binding)
      ? await materializeQwen35PackedGraphSourceWeight(source, binding, signal)
      : {
          bytes: await materializeQwen35R13PinnedSourceWeight(source, binding, signal),
          dtype: 'float32' as const,
          shape: binding.shape
        };
    const tensor = runtime.empty([...payload.shape], payload.dtype, device);
    validateRuntimeTensor(tensor, payload.shape, payload.dtype, binding.relaxArgumentName);
    tensor.copyFromRawBytes(payload.bytes);
    await device.sync();
    signal?.throwIfAborted();
    valuesByIndex.set(binding.index, Object.freeze({binding, tensor}));
    onProgress?.(payload.bytes.byteLength);
  }
  const values = bindingSnapshot.map(({index}) => {
    const value = valuesByIndex.get(index);
    if (!value) throw invalid(`Qwen3.5 R13 weight was not loaded: ${index}.`);
    return value;
  });
  return Object.freeze({
    bindingsSha256,
    argumentNamesSha256,
    values: Object.freeze(values)
  });
}

async function snapshotAndValidateRuntimeWeights(
  weights: Qwen35R13RuntimeWeights,
  expectedBindingHashes?: Readonly<{bindingsSha256: string; argumentNamesSha256: string}>
): Promise<Qwen35R13RuntimeWeights> {
  if (!weights || !Array.isArray(weights.values) || weights.values.length !== R13_WEIGHT_COUNT)
    throw invalid('Qwen3.5 R13 runtime weights are incomplete.');
  if (weights.values.some((value) => !value || !value.binding || !value.tensor))
    throw invalid('Qwen3.5 R13 runtime weights contain an empty tensor slot.');
  const bindings = snapshotBindingSet(weights.values.map(({binding}) => binding));
  const actualHashes = await bindingHashes(bindings);
  const expectedHashes =
    expectedBindingHashes ??
    (bindings.some(isPackedGraphBinding)
      ? undefined
      : {bindingsSha256: R13_WEIGHT_BINDINGS_SHA256, argumentNamesSha256: R13_EXTERNAL_ARGUMENT_NAMES_SHA256});
  if (
    weights.bindingsSha256 !== actualHashes.bindingsSha256 ||
    weights.argumentNamesSha256 !== actualHashes.argumentNamesSha256 ||
    !expectedHashes ||
    actualHashes.bindingsSha256 !== expectedHashes.bindingsSha256 ||
    actualHashes.argumentNamesSha256 !== expectedHashes.argumentNamesSha256
  )
    throw invalid('Qwen3.5 R13 runtime weights do not match the pinned ABI.');
  const values = weights.values.map(({tensor}, index) => {
    const binding = bindings[index]!;
    validateRuntimeTensor(
      tensor,
      isPackedGraphBinding(binding) ? binding.relaxShape : binding.shape,
      binding.relaxDtype,
      binding.relaxArgumentName
    );
    return Object.freeze({binding, tensor});
  });
  return Object.freeze({...actualHashes, values: Object.freeze(values)});
}

function validateInputTuple(tuple: Qwen35R13GraphInputTuple): readonly [number[], number[], number[]] {
  if (!Array.isArray(tuple) || tuple.length !== QWEN35_R13_RUNTIME_INPUT_NAMES.length)
    throw new TypeError('Qwen3.5 R13 requires its six ordered runtime inputs.');
  if (tuple.some(({name}, index) => name !== QWEN35_R13_RUNTIME_INPUT_NAMES[index]))
    throw new TypeError('Qwen3.5 R13 runtime inputs are not in pinned order.');
  const [embeds, mask, positions, options, temperatures, nSlots] = tuple;
  const [batch, sequence, hiddenSize] = embeds.shape;
  const optionWidth = options.shape[1];
  if (
    embeds.dtype !== 'float32' ||
    !(embeds.values instanceof Float32Array) ||
    embeds.shape.length !== 3 ||
    !Number.isSafeInteger(batch) ||
    batch! < 1 ||
    batch! > QWEN35_GRAPH_MAX_BATCH ||
    !Number.isSafeInteger(sequence) ||
    sequence! < 2 ||
    hiddenSize !== 2048 ||
    embeds.values.length !== batch! * sequence! * hiddenSize ||
    mask.dtype !== 'int32' ||
    !(mask.values instanceof Int32Array) ||
    !dimensionsMatch(mask.shape, [batch!, sequence!]) ||
    mask.values.length !== batch! * sequence! ||
    positions.dtype !== 'int32' ||
    !(positions.values instanceof Int32Array) ||
    !dimensionsMatch(positions.shape, [4, batch!, sequence!]) ||
    positions.values.length !== 4 * batch! * sequence! ||
    options.dtype !== 'int32' ||
    !(options.values instanceof Int32Array) ||
    options.shape.length !== 2 ||
    options.shape[0] !== batch ||
    !Number.isSafeInteger(optionWidth) ||
    optionWidth! < 2 ||
    optionWidth! > 255 ||
    options.values.length !== batch! * optionWidth! ||
    temperatures.dtype !== 'float32' ||
    !(temperatures.values instanceof Float32Array) ||
    !dimensionsMatch(temperatures.shape, [batch!]) ||
    temperatures.values.length !== batch! ||
    !temperatures.values.every((value) => Number.isFinite(value) && value > 0) ||
    nSlots.dtype !== 'int32' ||
    !(nSlots.values instanceof Int32Array) ||
    !dimensionsMatch(nSlots.shape, [batch!]) ||
    nSlots.values.length !== batch! ||
    !nSlots.values.every((value) => Number.isSafeInteger(value) && value >= 2 && value <= optionWidth!)
  )
    throw new TypeError('Qwen3.5 R13 runtime input shapes or dtypes do not match the pinned six-input ABI.');
  return [
    [batch!, sequence!, hiddenSize],
    [batch!, sequence!],
    [batch!, optionWidth!]
  ];
}

function bytes(values: Float32Array | Int32Array): Uint8Array {
  return new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
}

function tupleValues(value: unknown): readonly unknown[] {
  if (Array.isArray(value)) return value;
  if (
    !value ||
    typeof value !== 'object' ||
    typeof Reflect.get(value, 'size') !== 'function' ||
    typeof Reflect.get(value, 'get') !== 'function'
  )
    throw new TypeError('Qwen3.5 R13 TVM main must return the pinned two-tensor tuple.');
  const size = Reflect.apply(Reflect.get(value, 'size') as (...args: unknown[]) => unknown, value, []);
  if (size !== 2) throw new TypeError('Qwen3.5 R13 TVM main must return two outputs.');
  const get = Reflect.get(value, 'get') as (index: number) => unknown;
  return [get.call(value, 0), get.call(value, 1)];
}

function validateGraphOutputs(
  output: readonly unknown[],
  expectedShape: readonly number[]
): readonly [Qwen35R13Tensor, Qwen35R13Tensor] {
  if (output.length !== 2) throw new TypeError('Qwen3.5 R13 TVM main returned an invalid output tuple.');
  const [logits, probabilities] = output;
  if (
    !isR13Tensor(logits) ||
    !isR13Tensor(probabilities) ||
    logits.dtype !== 'float32' ||
    probabilities.dtype !== 'float32'
  )
    throw new TypeError('Qwen3.5 R13 TVM outputs must be float32 tensors.');
  if (!dimensionsMatch(logits.shape, expectedShape) || !dimensionsMatch(probabilities.shape, expectedShape))
    throw new TypeError('Qwen3.5 R13 TVM outputs do not match the batch and option dimensions.');
  return [logits, probabilities];
}

/**
 * Lower-level Qwen graph runner. It owns the R13 weight scope and verified runtime
 * handle; its parent loader owns any outer setup. It does not claim the
 * Qwen35OwnedRunner cached-memory-release contract.
 */
export async function createQwen35R13GraphRunner({
  handle,
  source,
  bindings,
  expectedBindingHashes,
  resolveInputSources,
  signal,
  onProgress,
  loadWeights = loadQwen35R13RuntimeWeights
}: Qwen35R13GraphRunnerOptions): Promise<Qwen35R13GraphRunner> {
  if (typeof resolveInputSources !== 'function') throw new TypeError('Qwen3.5 R13 input sources are required.');
  if (!verifiedRuntimeHandles.has(handle) || claimedRuntimeHandles.has(handle) || disposedRuntimeHandles.has(handle))
    throw invalid('Qwen3.5 R13 runtime handle must be verified, live, and unclaimed.');
  const bindingSnapshot = snapshotBindingSet(bindings);
  claimedRuntimeHandles.add(handle);
  const {runtime, device, main} = handle;
  let ownerScopeOpen = false;
  try {
    runtime.beginScope();
    ownerScopeOpen = true;
    const weights = await snapshotAndValidateRuntimeWeights(
      await loadWeights({
        runtime,
        device,
        source,
        bindings: bindingSnapshot,
        expectedBindingHashes,
        signal,
        onProgress
      }),
      expectedBindingHashes
    );
    let disposed = false;

    async function runInputs(
      inputTuple: Qwen35R13GraphInputTuple,
      signal?: AbortSignal | null,
      sourcePrefixLength?: number
    ): Promise<readonly Float32Array[]> {
      if (disposed) throw new WebDecisionError('SESSION_CLOSED', 'The Strands Qwen3.5 runner has been disposed.');
      signal?.throwIfAborted();
      const [[, sequence], , optionShape] = validateInputTuple(inputTuple);
      const nSlots = inputTuple[5].values;
      if (
        handle.hasSourcePrefix &&
        (sourcePrefixLength === undefined ||
          !Number.isSafeInteger(sourcePrefixLength) ||
          sourcePrefixLength < 0 ||
          sourcePrefixLength > sequence!)
      )
        throw new TypeError('Qwen3.5 requires a source prefix length within the input sequence.');
      const args: unknown[] = [];
      let failure: unknown;
      let failed = false;
      let result: readonly Float32Array[] | undefined;
      let scopeOpen = false;
      try {
        runtime.beginScope();
        scopeOpen = true;
        for (const item of inputTuple) {
          const tensor = runtime.empty([...item.shape], item.dtype, device);
          tensor.copyFromRawBytes(bytes(item.values));
          args.push(tensor);
        }
        if (handle.hasSourcePrefix) args.push(runtime.makeShapeTuple!([sourcePrefixLength!]));
        args.push(...weights.values.map(({tensor}) => tensor));
        const output = tupleValues(await main(...args));
        signal?.throwIfAborted();
        const expectedOutputShape = [optionShape[0]!, optionShape[1]!];
        const [logits, probabilities] = validateGraphOutputs(output, expectedOutputShape);
        const hostLogits = runtime.empty(expectedOutputShape, 'float32', runtime.cpu());
        const hostProbabilities = runtime.empty(expectedOutputShape, 'float32', runtime.cpu());
        hostLogits.copyFrom(logits);
        hostProbabilities.copyFrom(probabilities);
        await device.sync();
        signal?.throwIfAborted();
        result = readQwen35GraphProbabilityRows([hostLogits, hostProbabilities], Array.from(nSlots));
      } catch (cause) {
        failure = cause;
        failed = true;
      } finally {
        if (scopeOpen) {
          try {
            runtime.endScope();
          } catch (cause) {
            if (!failed) failure = cause;
            failed = true;
          }
        }
      }
      if (failed) throw failure;
      return result!;
    }

    const graphRunner: Qwen35R13GraphRunner = {
      async run(chunk, input, signal) {
        if (disposed) throw new WebDecisionError('SESSION_CLOSED', 'The Strands Qwen3.5 runner has been disposed.');
        signal?.throwIfAborted();
        const sources = await resolveInputSources(chunk, input, signal);
        signal?.throwIfAborted();
        return runInputs(
          await prepareQwen35R13GraphInputTuple(chunk, input, sources),
          signal,
          chunk.prefixInputIds.length
        );
      },
      runInputs,
      async dispose() {
        if (disposed) return;
        disposed = true;
        let failure: unknown;
        if (ownerScopeOpen) {
          ownerScopeOpen = false;
          try {
            runtime.endScope();
          } catch (cause) {
            failure = cause;
          }
        }
        try {
          await handle.dispose();
        } catch (cause) {
          if (failure === undefined) failure = cause;
        }
        if (failure !== undefined) throw failure;
      }
    };
    return Object.freeze(graphRunner);
  } catch (cause) {
    if (ownerScopeOpen) {
      ownerScopeOpen = false;
      try {
        runtime.endScope();
      } catch {}
    }
    try {
      await handle.dispose();
    } catch {}
    throw cause;
  }
}

// R18 uses the same verified 6+697 input/weight ABI. Keep the generation-specific
// fixture verifier above separate while sharing the proven source-weight runner.
export const materializeQwen35R18PinnedSourceWeight = materializeQwen35R13PinnedSourceWeight;
export const loadQwen35R18RuntimeWeights = loadQwen35R13RuntimeWeights;
export const createQwen35R18GraphRunner = createQwen35R13GraphRunner;
