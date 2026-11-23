import type {DLDevice, Instance, Tensor, VirtualMachine} from '../runtime/tvm-types.js';
import {withGemma4V2Cleanup} from './gemma4-v2-cleanup.mjs';
import type {Gemma4V2RunnerInput} from './gemma4-v2-session.mts';
import type {openPinnedGemma4V2Source} from './gemma4-v2-source.mts';
import {loadGemma4V2VisionWeights} from './gemma4-v2-vision-bindings.mts';
import {
  insertGemma4V2VisionEmbedding,
  runGemma4V2VisionTensor,
  type Gemma4V2VisionInput,
  type Gemma4V2VisionWeights
} from './gemma4-v2-vision.mts';

type Source = Awaited<ReturnType<typeof openPinnedGemma4V2Source>>;
type WeightLoader = typeof loadGemma4V2VisionWeights;
type Inserter = (word: Tensor, input: Gemma4V2RunnerInput, signal?: AbortSignal | null) => Promise<Tensor>;
export type Gemma4V2ImageInserter = Inserter & {releaseCachedMemory(): void};

function visionInput(value: unknown): Gemma4V2VisionInput {
  if (!value || typeof value !== 'object') throw new TypeError('Invalid prepared Gemma 4 image.');
  const image = value as Partial<Gemma4V2VisionInput>;
  if (
    !(image.pixels instanceof Float32Array) ||
    !(image.positions instanceof Int32Array) ||
    !Number.isInteger(image.tokens)
  )
    throw new TypeError('Invalid prepared Gemma 4 image.');
  return image as Gemma4V2VisionInput;
}

/** Create an image inserter inside a TVM scope that outlives all its calls. */
export async function createGemma4V2ImageInserter(
  runtime: Instance,
  device: DLDevice,
  vm: VirtualMachine,
  source: Source,
  signal?: AbortSignal | null,
  loadWeights: WeightLoader = loadGemma4V2VisionWeights
) {
  signal?.throwIfAborted();
  const scoped = runtime as Instance & {
    moveToParentScope<T extends Tensor>(value: T): T;
    detachFromCurrentScope<T extends Tensor>(value: T): T;
  };
  let inserter: Inserter | undefined;
  let cachedWeights: Tensor[] = [];
  let busy = false;
  const insert: Inserter = async (word, input, runSignal) => {
    if (busy) throw new Error('Gemma 4 image inserter is already in use.');
    runSignal?.throwIfAborted();
    if (input.images.length !== input.imageSpans.length)
      throw new RangeError('Gemma 4 image spans do not match images.');
    if (input.images.length === 0) return word;
    busy = true;
    try {
      if (!inserter) {
        const weights = await loadWeights(runtime, device, source, runSignal);
        runSignal?.throwIfAborted();
        const tensors = [...weights.patch, ...weights.layers.flat(), ...weights.projection];
        for (const tensor of tensors) scoped.moveToParentScope(tensor);
        cachedWeights = tensors;
        inserter = createGemma4V2ImageInserterWithWeights(runtime, device, vm, weights);
      }
      return await inserter(word, input, runSignal);
    } finally {
      busy = false;
    }
  };
  return Object.assign(insert, {
    releaseCachedMemory() {
      if (busy) throw new Error('Cannot release Gemma 4 image weights while in use.');
      const weights = cachedWeights;
      cachedWeights = [];
      inserter = undefined;
      let failure: unknown;
      for (const tensor of weights) {
        try {
          scoped.detachFromCurrentScope(tensor);
          (tensor as Tensor & {dispose(): void}).dispose();
        } catch (error) {
          failure ??= error;
        }
      }
      if (failure !== undefined) throw failure;
    }
  });
}

/** Internal seam for sharing verified weights across calls. */
export function createGemma4V2ImageInserterWithWeights(
  runtime: Instance,
  device: DLDevice,
  vm: VirtualMachine,
  weights: Gemma4V2VisionWeights
) {
  const scoped = runtime as Instance & {moveToParentScope<T extends Tensor>(value: T): T};
  const stage =
    (name: string) =>
    (...args: Tensor[]) =>
      vm.getFunction(name)(...args) as Tensor;
  const graph = {
    patch: stage('vision_patch_main'),
    layer: stage('vision_encoder_main'),
    pool: stage('vision_pool_main'),
    projection: stage('vision_projection_main')
  };
  const insert = stage('image_insert_main');
  return async (word: Tensor, input: Gemma4V2RunnerInput, runSignal?: AbortSignal | null): Promise<Tensor> => {
    if (input.images.length !== input.imageSpans.length)
      throw new RangeError('Gemma 4 image spans do not match images.');
    let output = word;
    let previousEnd = 0;
    for (let i = 0; i < input.images.length; i++) {
      runSignal?.throwIfAborted();
      const image = visionInput(input.images[i]);
      const span = input.imageSpans[i]!;
      if (
        !Number.isSafeInteger(span.start) ||
        !Number.isSafeInteger(span.end) ||
        span.start < previousEnd ||
        span.end > input.stateIds.length ||
        span.end - span.start !== image.tokens ||
        input.stateIds.slice(span.start, span.end).some((id) => id !== 258880)
      )
        throw new RangeError('Invalid Gemma 4 image span.');
      runtime.beginScope();
      await withGemma4V2Cleanup(
        async () => {
          const features = await runGemma4V2VisionTensor(runtime, device, graph, weights, image, runSignal);
          const next = insertGemma4V2VisionEmbedding(runtime, device, insert, output, features, span.start);
          await device.sync();
          runSignal?.throwIfAborted();
          const previous = output;
          output = scoped.moveToParentScope(next);
          if (previous !== word) (previous as Tensor & {dispose(): void}).dispose();
        },
        () => runtime.endScope()
      );
      previousEnd = span.end;
    }
    return output;
  };
}
