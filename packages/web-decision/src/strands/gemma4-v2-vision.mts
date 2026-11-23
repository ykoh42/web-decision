import type {DLDevice, Instance, Tensor} from '../runtime/tvm-types.js';
import {withGemma4V2Cleanup} from './gemma4-v2-cleanup.mjs';
import {GEMMA4_V2_COMPILED_CONTEXT_LENGTH} from './gemma4-v2-context.mjs';

const PATCHES = 2520;
const WIDTH = 768;
const ROTARY_WIDTH = 64;
const LAYERS = 16;
const OUTPUT_WIDTH = 1536;

export interface Gemma4V2VisionInput {
  pixels: Float32Array;
  positions: Int32Array;
  tokens: number;
}

const scratch = new Float32Array(1);
const bits = new Uint32Array(scratch.buffer);
function bfloat16(value: number): number {
  scratch[0] = value;
  bits[0] = (bits[0]! + 0x7fff + ((bits[0]! >>> 16) & 1)) & 0xffff0000;
  return scratch[0]!;
}

function measureImageGeometry(input: Gemma4V2VisionInput) {
  if (
    input.pixels.length !== PATCHES * WIDTH ||
    input.positions.length !== PATCHES * 2 ||
    !Number.isInteger(input.tokens) ||
    input.tokens < 1 ||
    input.tokens > PATCHES / 9
  )
    throw new RangeError('Invalid Gemma 4 image tensor shape.');
  let maxX = -1;
  let maxY = -1;
  let valid = 0;
  for (let i = 0; i < PATCHES; i++) {
    const x = input.positions[i * 2]!;
    const y = input.positions[i * 2 + 1]!;
    if (x === -1 && y === -1) continue;
    if (x < 0 || y < 0 || x >= PATCHES / 3 || y >= PATCHES / 3) throw new RangeError('Invalid image patch position.');
    valid++;
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }
  const width = maxX + 1;
  const height = maxY + 1;
  if (width % 3 !== 0 || height % 3 !== 0 || width * height !== valid || valid !== input.tokens * 9)
    throw new RangeError('Invalid image patch geometry.');
  return {width, valid};
}

function fillRotary(cos: Float32Array, sin: Float32Array, index: number, x: number, y: number) {
  for (let d = 0; d < ROTARY_WIDTH; d++) {
    const coordinate = d < 32 ? x : y;
    const inverse = Math.fround(1 / Math.fround(100 ** ((d % 16) / 16)));
    const angle = Math.fround(coordinate * inverse);
    cos[index * ROTARY_WIDTH + d] = bfloat16(Math.cos(angle));
    sin[index * ROTARY_WIDTH + d] = bfloat16(Math.sin(angle));
  }
}

/** Construct the rotary embedding, attention mask, and pooling indices used by Gemma 4. */
export function gemma4V2VisionGeometry(input: Gemma4V2VisionInput) {
  const {width, valid} = measureImageGeometry(input);
  const cos = new Float32Array(PATCHES * ROTARY_WIDTH);
  const sin = new Float32Array(PATCHES * ROTARY_WIDTH);
  const mask = new Float32Array(PATCHES);
  const indices = new Int32Array(input.tokens * 9);
  const counts = new Uint8Array(input.tokens);
  const seen = new Uint8Array(valid);
  const pooledWidth = width / 3;
  for (let i = 0; i < PATCHES; i++) {
    const x = input.positions[i * 2]!;
    const y = input.positions[i * 2 + 1]!;
    if (x === -1) mask[i] = -1e30;
    else {
      const location = y * width + x;
      if (seen[location]) throw new RangeError('Duplicate image patch position.');
      seen[location] = 1;
      const group = Math.floor(x / 3) + pooledWidth * Math.floor(y / 3);
      const count = counts[group]!;
      if (count >= 9) throw new RangeError('Invalid image pooling group.');
      indices[group * 9 + count] = i;
      counts[group] = count + 1;
    }
    fillRotary(cos, sin, i, x, y);
  }
  return {cos, sin, mask, indices};
}

type VisionStage = (...inputs: Tensor[]) => Tensor;
export interface Gemma4V2VisionGraph {
  patch: VisionStage;
  layer: VisionStage;
  pool: VisionStage;
  projection: VisionStage;
}

/** All weights and graph functions must belong to the supplied TVM runtime. */
export interface Gemma4V2VisionWeights {
  patch: readonly Tensor[];
  layers: readonly (readonly Tensor[])[];
  projection: readonly Tensor[];
}

/** Caller owns the enclosing TVM scope and must keep it alive until GPU work completes. */
export async function runGemma4V2VisionTensor(
  runtime: Instance,
  device: DLDevice,
  graph: Gemma4V2VisionGraph,
  weights: Gemma4V2VisionWeights,
  input: Gemma4V2VisionInput,
  signal?: AbortSignal | null
): Promise<Tensor> {
  signal?.throwIfAborted();
  if (weights.layers.length !== LAYERS) throw new RangeError('Gemma 4 requires 16 vision encoder layers.');
  const geometry = gemma4V2VisionGeometry(input);
  signal?.throwIfAborted();
  const tensor = (shape: number[], dtype: string, values: Float32Array | Int32Array) =>
    runtime.empty(shape, dtype, device).copyFrom(values);
  const pixels = tensor([1, PATCHES, WIDTH], 'float32', input.pixels);
  const positions = tensor([1, PATCHES, 2], 'int32', input.positions);
  const cos = tensor([1, PATCHES, ROTARY_WIDTH], 'float32', geometry.cos);
  const sin = tensor([1, PATCHES, ROTARY_WIDTH], 'float32', geometry.sin);
  const mask = tensor([1, 1, 1, PATCHES], 'float32', geometry.mask);
  const indices = tensor([input.tokens, 9], 'int32', geometry.indices);
  let hidden = graph.patch(pixels, positions, ...weights.patch);
  for (const [index, layer] of weights.layers.entries()) {
    signal?.throwIfAborted();
    hidden = graph.layer(hidden, cos, sin, positions, mask, ...layer);
    // Bound the amount of queued vision work on devices with constrained GPU resources.
    if ((index + 1) % 2 === 0) await device.sync();
  }
  signal?.throwIfAborted();
  const pooled = graph.pool(hidden, indices);
  const output = graph.projection(pooled, ...weights.projection);
  if (
    output.dtype !== 'float32' ||
    output.shape.length !== 2 ||
    output.shape[0] !== input.tokens ||
    output.shape[1] !== OUTPUT_WIDTH
  )
    throw new Error('Unexpected Gemma 4 vision output shape or dtype.');
  return output;
}

/** Execute the model-local image graph and copy its embeddings to the CPU. */
export async function runGemma4V2Vision(
  runtime: Instance,
  device: DLDevice,
  graph: Gemma4V2VisionGraph,
  weights: Gemma4V2VisionWeights,
  input: Gemma4V2VisionInput,
  signal?: AbortSignal | null
): Promise<Float32Array> {
  signal?.throwIfAborted();
  runtime.beginScope();
  return withGemma4V2Cleanup(
    async () => {
      const output = await runGemma4V2VisionTensor(runtime, device, graph, weights, input, signal);
      const host = runtime.empty([input.tokens, OUTPUT_WIDTH], 'float32', runtime.cpu()).copyFrom(output);
      await device.sync();
      signal?.throwIfAborted();
      return (host.toArray() as Float32Array).slice();
    },
    () => runtime.endScope()
  );
}

/** Insert image features on-device. Call once per image in prompt order. */
export function insertGemma4V2VisionEmbedding(
  runtime: Instance,
  device: DLDevice,
  insert: VisionStage,
  hidden: Tensor,
  image: Tensor,
  start: number
): Tensor {
  const length = hidden.shape[1];
  const count = image.shape[0];
  if (
    hidden.dtype !== 'float32' ||
    hidden.shape.length !== 3 ||
    hidden.shape[0] !== 1 ||
    hidden.shape[2] !== OUTPUT_WIDTH ||
    !Number.isSafeInteger(length) ||
    length! < 1 ||
    length! > GEMMA4_V2_COMPILED_CONTEXT_LENGTH ||
    image.dtype !== 'float32' ||
    image.shape.length !== 2 ||
    image.shape[1] !== OUTPUT_WIDTH ||
    !Number.isSafeInteger(count) ||
    count! < 1 ||
    count! > length! ||
    !Number.isSafeInteger(start) ||
    start < 0 ||
    start > length! - count!
  )
    throw new RangeError('Image features do not fit the Gemma 4 text embedding.');
  const offset = runtime.empty([1], 'int32', device).copyFrom(Int32Array.of(start));
  const result = insert(hidden, image, offset);
  if (
    result.dtype !== 'float32' ||
    result.shape.length !== 3 ||
    result.shape[0] !== 1 ||
    result.shape[1] !== length ||
    result.shape[2] !== OUTPUT_WIDTH
  )
    throw new Error('Unexpected Gemma 4 image embedding output shape or dtype.');
  return result;
}
