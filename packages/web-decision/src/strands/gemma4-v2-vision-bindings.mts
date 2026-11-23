import bindingInventory from '../../assets/strands-gemma4-v2/vision-bindings.json' with {type: 'json'};
import sourceInventory from '../../assets/strands-gemma4-v2/source-inventory.json' with {type: 'json'};
import {sha256} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import type {DLDevice, Instance, Tensor} from '../runtime/tvm-types.js';
import {GEMMA4_V2_REPOSITORY, GEMMA4_V2_REVISION} from './gemma4-v2-source.mts';
import type {Gemma4V2VisionWeights} from './gemma4-v2-vision.mts';

const BINDING_SHA256 = '23255ad23212dc2d286ed58d2a09deed429b0aac2895536699a4ddb41be2616f';
type Parameter = Readonly<{name: string; shape: readonly number[]; dtype: string; float32Sha256: string}>;
type Layer = Readonly<{layer: number; parameters: readonly Parameter[]}>;
type Bindings = Readonly<{
  format: string;
  repository: string;
  revision: string;
  sourceInventorySha256: string;
  sourceGraphs: Readonly<Record<string, {bytes: number; sha256: string}>>;
  patch: readonly Parameter[];
  layers: readonly Layer[];
  projection: readonly Parameter[];
}>;
type SourceInventory = Readonly<{
  tensors: Readonly<Record<string, {shape: readonly number[]; dtype: string; length: number}>>;
}>;
type SourceReader = Readonly<{readTensor(name: string, signal?: AbortSignal | null): Promise<Uint8Array>}>;
const invalid = (message: string) => new WebDecisionError('INVALID_ASSET', message);
const digest = (value: unknown) => sha256(new TextEncoder().encode(JSON.stringify(value) + '\n'));

function validateParameters(parameters: readonly Parameter[], source: SourceInventory, prefix: string): void {
  const seen = new Set<string>();
  for (const parameter of parameters) {
    const expected = source.tensors[parameter.name];
    if (
      !parameter.name.startsWith(prefix) ||
      seen.has(parameter.name) ||
      !expected ||
      parameter.dtype !== 'BF16' ||
      expected.dtype !== 'BF16' ||
      !/^[0-9a-f]{64}$/u.test(parameter.float32Sha256) ||
      expected.shape.length !== parameter.shape.length ||
      parameter.shape.some((size, index) => size !== expected.shape[index]) ||
      expected.length !== parameter.shape.reduce((size, dimension) => size * dimension, 1) * 2
    )
      throw invalid('Vision parameter does not match the pinned checkpoint.');
    seen.add(parameter.name);
  }
}

/** Check the binding list, ordered as the corresponding compiled graph expects it. */
export function assertGemma4V2VisionGraphBindings(
  expected: readonly Parameter[],
  actual: readonly Pick<Parameter, 'name' | 'shape'>[]
): void {
  if (
    actual.length !== expected.length ||
    actual.some(
      (item, index) =>
        item.name !== expected[index]!.name ||
        item.shape.length !== expected[index]!.shape.length ||
        item.shape.some((size, dimension) => size !== expected[index]!.shape[dimension])
    )
  )
    throw invalid('Vision graph arguments do not match the pinned bindings.');
}

/** Internal test seam. Production loading verifies both embedded manifest digests. */
export async function loadGemma4V2VisionBindingsFromManifests(
  value: Bindings,
  expectedHash: string,
  source: SourceInventory
) {
  if ((await digest(value)) !== expectedHash || (await digest(source)) !== value.sourceInventorySha256)
    throw invalid('Vision binding or source inventory digest differs.');
  const manifest = JSON.parse(JSON.stringify(value)) as Bindings;
  if (
    manifest.format !== 'web-decision-gemma4-v2-vision-bindings-v1' ||
    manifest.repository !== GEMMA4_V2_REPOSITORY ||
    manifest.revision !== GEMMA4_V2_REVISION ||
    !Array.isArray(manifest.layers) ||
    manifest.layers.length !== 16 ||
    !Array.isArray(manifest.patch) ||
    manifest.patch.length !== 2 ||
    !Array.isArray(manifest.projection) ||
    manifest.projection.length !== 1
  )
    throw invalid('Invalid Gemma 4 vision binding manifest.');
  for (const name of ['patch', 'layer', 'projection']) {
    const graph = manifest.sourceGraphs[name];
    if (!graph || !Number.isSafeInteger(graph.bytes) || graph.bytes < 1 || !/^[0-9a-f]{64}$/u.test(graph.sha256))
      throw invalid('Invalid Gemma 4 vision source graph evidence.');
  }
  validateParameters(manifest.patch, source, 'model.vision_tower.patch_embedder.');
  validateParameters(manifest.projection, source, 'model.embed_vision.');
  for (const [index, layer] of manifest.layers.entries()) {
    if (layer.layer !== index || !Array.isArray(layer.parameters) || layer.parameters.length !== 41)
      throw invalid('Invalid Gemma 4 vision layer binding.');
    validateParameters(layer.parameters, source, `model.vision_tower.encoder.layers.${index}.`);
  }
  return manifest;
}

export function loadPinnedGemma4V2VisionBindings() {
  return loadGemma4V2VisionBindingsFromManifests(
    bindingInventory as unknown as Bindings,
    BINDING_SHA256,
    sourceInventory as unknown as SourceInventory
  );
}

/** Convert exact checkpoint BF16 bits to the float32 graph input representation. */
export function gemma4V2VisionFloat32(bytes: Uint8Array): Float32Array {
  if (bytes.length % 2) throw invalid('Invalid BF16 vision tensor length.');
  const values = new Float32Array(bytes.length / 2);
  const output = new DataView(values.buffer);
  const input = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < values.length; index++)
    output.setUint32(index * 4, input.getUint16(index * 2, true) << 16, true);
  return values;
}

/** Upload the verified graph arguments in graph order. Caller owns the enclosing TVM scope. */
export async function loadGemma4V2VisionWeights(
  runtime: Instance,
  device: DLDevice,
  source: SourceReader,
  signal?: AbortSignal | null
): Promise<Gemma4V2VisionWeights> {
  const bindings = await loadPinnedGemma4V2VisionBindings();
  async function upload(parameters: readonly Parameter[]): Promise<Tensor[]> {
    const tensors: Tensor[] = [];
    for (const parameter of parameters) {
      signal?.throwIfAborted();
      const values = gemma4V2VisionFloat32(await source.readTensor(parameter.name, signal));
      if (
        values.length !== parameter.shape.reduce((size, dimension) => size * dimension, 1) ||
        (await sha256(new Uint8Array(values.buffer))) !== parameter.float32Sha256
      )
        throw invalid(`Vision parameter digest differs: ${parameter.name}`);
      signal?.throwIfAborted();
      tensors.push(runtime.empty([...parameter.shape], 'float32', device).copyFrom(values));
    }
    return tensors;
  }
  const patch = await upload(bindings.patch);
  const layers: Tensor[][] = [];
  for (const layer of bindings.layers) layers.push(await upload(layer.parameters));
  return {patch, layers, projection: await upload(bindings.projection)};
}
