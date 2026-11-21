import {sha256} from '../assets.js';
import type {DLDevice, Instance, Tensor} from '../runtime/tvm-types.js';
import {WebDecisionError} from '../kev.js';
import {QWEN35_VISION_ARGUMENT_NAMES_SHA256, QWEN35_VISION_BINDINGS_SHA256} from './qwen35-vision-contract.mts';
import {orderQwen35SourceTensorReads, type Qwen35PinnedSource} from './qwen35-source.mts';
import type {Qwen35WordTensor} from './qwen35-assets.mjs';
import type {Qwen35VisionArtifact} from './qwen35-vision-runner.mts';

export interface Qwen35VisionWeightBinding {
  readonly index: number;
  readonly targetName: string;
  readonly sourceName: string;
  readonly shape: readonly number[];
  readonly elementCount: number;
  readonly sourceByteCount: number;
  readonly sourcePayloadSha256: string;
  readonly sourceDtype: 'BF16';
  readonly targetDtype: 'float32';
  readonly conversion: 'exact-bf16-to-float32';
}

type VisionWeightLoaderOptions = Readonly<{
  readonly runtime: Pick<Instance, 'empty'>;
  readonly device: DLDevice;
  readonly artifact: Qwen35VisionArtifact;
  readonly bindings: readonly Qwen35VisionWeightBinding[];
  readonly signal?: AbortSignal | null | undefined;
  readonly onProgress?: ((bytes: number) => void) | undefined;
}>;

type ReadVisionWeight = (binding: Qwen35VisionWeightBinding, signal?: AbortSignal | null) => Promise<Uint8Array>;

export type Qwen35VisionWeightLoadOptions = VisionWeightLoaderOptions & Readonly<{source: Qwen35PinnedSource}>;

type RuntimeTensor = Tensor & Readonly<{shape: readonly number[]; dtype: string}>;

function invalid(message: string): WebDecisionError {
  return new WebDecisionError('INVALID_ASSET', message);
}

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((dimension, index) => dimension === right[index]);
}

function canonicalBindings(bindings: readonly Qwen35VisionWeightBinding[]): string {
  return JSON.stringify(
    bindings.map(
      ({
        index,
        targetName,
        sourceName,
        shape,
        elementCount,
        sourceByteCount,
        sourcePayloadSha256,
        sourceDtype,
        targetDtype,
        conversion
      }) => ({
        index,
        targetName,
        sourceName,
        shape,
        elementCount,
        sourceByteCount,
        sourcePayloadSha256,
        sourceDtype,
        targetDtype,
        conversion
      })
    )
  );
}

async function verifyBindings(
  artifact: Qwen35VisionArtifact,
  bindings: readonly Qwen35VisionWeightBinding[]
): Promise<void> {
  const parameters = artifact.mainArgumentOrder.filter(({sourceKind}) => sourceKind === 'PARAMETER');
  if (!Array.isArray(bindings) || bindings.length !== 297 || parameters.length !== bindings.length)
    throw invalid('Qwen3.5 vision source bindings do not match the pinned parameter count.');
  const targets = new Set<string>();
  const sources = new Set<string>();
  for (let index = 0; index < bindings.length; index += 1) {
    const binding = bindings[index]!;
    const argument = parameters[index]!;
    if (
      binding.index !== index ||
      typeof binding.targetName !== 'string' ||
      binding.targetName !== argument.name ||
      !argument.shape ||
      !Array.isArray(binding.shape) ||
      binding.shape.some((dimension: number) => !Number.isSafeInteger(dimension) || dimension < 1) ||
      !sameShape(binding.shape, argument.shape) ||
      typeof binding.sourceName !== 'string' ||
      binding.sourceName.length === 0 ||
      targets.has(binding.targetName) ||
      sources.has(binding.sourceName) ||
      !Number.isSafeInteger(binding.elementCount) ||
      binding.elementCount !== binding.shape.reduce((count: number, dimension: number) => count * dimension, 1) ||
      binding.sourceByteCount !== binding.elementCount * 2 ||
      binding.sourceDtype !== 'BF16' ||
      binding.targetDtype !== 'float32' ||
      binding.conversion !== 'exact-bf16-to-float32' ||
      !/^[0-9a-f]{64}$/.test(binding.sourcePayloadSha256)
    )
      throw invalid('Qwen3.5 vision source binding differs at parameter ' + index + '.');
    targets.add(binding.targetName);
    sources.add(binding.sourceName);
  }
  if (
    (await sha256(new TextEncoder().encode(canonicalBindings(bindings)))) !== QWEN35_VISION_BINDINGS_SHA256 ||
    (await sha256(new TextEncoder().encode(bindings.map(({targetName}) => targetName).join('\n')))) !==
      QWEN35_VISION_ARGUMENT_NAMES_SHA256
  )
    throw invalid('Qwen3.5 vision source binding digest does not match the pinned graph.');
}

function sourceTensor(source: Qwen35PinnedSource, binding: Qwen35VisionWeightBinding): Qwen35WordTensor {
  const tensor = source.manifest.assets.base?.tensors.find(({name}) => name === binding.sourceName);
  if (
    !tensor ||
    tensor.dtype !== 'BF16' ||
    !sameShape(tensor.shape, binding.shape) ||
    tensor.elementCount !== binding.elementCount ||
    tensor.sourceByteCount !== binding.sourceByteCount ||
    tensor.sourcePayloadSha256 !== binding.sourcePayloadSha256
  )
    throw invalid('Qwen3.5 pinned vision tensor metadata differs: ' + binding.sourceName + '.');
  return tensor;
}

/** Widen one original checkpoint tensor after verifying its source payload hash. */
export async function materializeQwen35VisionSourceWeight(
  source: Qwen35PinnedSource,
  binding: Qwen35VisionWeightBinding,
  signal?: AbortSignal | null
): Promise<Uint8Array> {
  signal?.throwIfAborted();
  const tensor = sourceTensor(source, binding);
  const raw = await source.readTensor('base', tensor.name, {signal: signal ?? undefined});
  return widenVerifiedQwen35VisionWeight(raw, binding, signal);
}

async function widenVerifiedQwen35VisionWeight(
  raw: Uint8Array,
  binding: Qwen35VisionWeightBinding,
  signal?: AbortSignal | null
): Promise<Uint8Array> {
  signal?.throwIfAborted();
  if (raw.byteLength !== binding.sourceByteCount || (await sha256(raw)) !== binding.sourcePayloadSha256)
    throw invalid('Qwen3.5 pinned vision tensor payload differs: ' + binding.sourceName + '.');

  const input = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const output = new Uint8Array(binding.elementCount * 4);
  const outputView = new DataView(output.buffer);
  for (let index = 0; index < binding.elementCount; index += 1)
    outputView.setUint32(index * 4, input.getUint16(index * 2, true) << 16, true);
  return output;
}

/** Upload the original, verified Qwen3.5 checkpoint vision parameters. */
export function loadQwen35VisionParameterTensors({
  source,
  ...options
}: Qwen35VisionWeightLoadOptions): Promise<ReadonlyMap<string, Tensor>> {
  return loadQwen35VisionParameterTensorsFromSource({
    ...options,
    orderBindings(bindings) {
      if (!source || typeof source.readTensor !== 'function' || !source.manifest)
        throw invalid('Qwen3.5 vision weight loading requires the pinned Safetensors source.');
      return orderQwen35SourceTensorReads(
        source.manifest,
        bindings.map((binding) => ({...binding, asset: 'base' as const}))
      );
    },
    async readWeight(binding, signal) {
      const tensor = sourceTensor(source, binding);
      return source.readTensor('base', tensor.name, {signal: signal ?? undefined});
    }
  });
}

/** Verify the graph bindings and each native BF16 payload before uploading it. */
export async function loadQwen35VisionParameterTensorsFromSource({
  runtime,
  device,
  artifact,
  bindings,
  signal,
  onProgress,
  orderBindings = (sourceBindings) => sourceBindings,
  readWeight
}: VisionWeightLoaderOptions &
  Readonly<{
    orderBindings?: (bindings: readonly Qwen35VisionWeightBinding[]) => readonly Qwen35VisionWeightBinding[];
    readWeight: ReadVisionWeight;
  }>): Promise<ReadonlyMap<string, Tensor>> {
  signal?.throwIfAborted();
  await verifyBindings(artifact, bindings);
  signal?.throwIfAborted();

  const loaded = new Map<string, Tensor>();
  for (const binding of orderBindings(bindings)) {
    signal?.throwIfAborted();
    const raw = await readWeight(binding, signal);
    const bytes = await widenVerifiedQwen35VisionWeight(raw, binding, signal);
    const tensor = runtime.empty([...binding.shape], 'float32', device) as RuntimeTensor;
    if (tensor.dtype !== 'float32' || !Array.isArray(tensor.shape) || !sameShape(tensor.shape, binding.shape))
      throw invalid('Qwen3.5 TVM vision parameter has an unexpected shape or dtype: ' + binding.targetName + '.');
    tensor.copyFromRawBytes(bytes);
    await device.sync();
    signal?.throwIfAborted();
    loaded.set(binding.targetName, tensor);
    onProgress?.(bytes.byteLength);
  }
  const parameters = new Map<string, Tensor>();
  for (const {targetName} of bindings) {
    const tensor = loaded.get(targetName);
    if (!tensor) throw invalid('Qwen3.5 vision tensor load missed a pinned ABI parameter.');
    parameters.set(targetName, tensor);
  }
  return parameters;
}
