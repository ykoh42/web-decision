import type {Qwen35GraphInputChunk, Qwen35GraphInputRecord} from './qwen35-graph-input.mts';
import type {Qwen35PreEmbedSources} from './qwen35-r13-pre-embed-bridge.mts';
import type {Qwen35SourceEmbeddingLookup} from './qwen35-source-embedding-lookup.mts';
import type {Qwen35VisionActivationPrecision, Qwen35VisionFeatureRunner} from './qwen35-vision-runner.mts';

export interface Qwen35PreEmbedSourceResolverOptions {
  readonly embeddings: Pick<Qwen35SourceEmbeddingLookup, 'lookupTokenEmbeddings'>;
  readonly vision: Qwen35VisionFeatureRunner | null;
  readonly visionActivationPrecision?: Qwen35VisionActivationPrecision;
}

export type Qwen35PreEmbedSourceResolver = (
  chunk: Qwen35GraphInputChunk,
  input: Qwen35GraphInputRecord,
  signal?: AbortSignal | null
) => Promise<Qwen35PreEmbedSources>;

/** Connect pinned source embedding rows and the per-image TVM runner to the R13 bridge. */
export function createQwen35PreEmbedSourceResolver({
  embeddings,
  vision,
  visionActivationPrecision = 'bfloat16'
}: Qwen35PreEmbedSourceResolverOptions): Qwen35PreEmbedSourceResolver {
  if (!embeddings || typeof embeddings.lookupTokenEmbeddings !== 'function')
    throw new TypeError('Qwen3.5 needs the pinned source embedding lookup.');

  // R13 dispatches one prepared request as multiple graph chunks. Keep one
  // vision result per prepared input, while allowing completed requests to be
  // collected and ensuring different requests never share image features.
  const imageFeaturesByInput = new WeakMap<Qwen35GraphInputRecord, Promise<readonly Float32Array[]>>();

  return async function resolve(_chunk, input, signal) {
    signal?.throwIfAborted();
    if (!input || !Number.isSafeInteger(input.imageCount) || input.imageCount < 0)
      throw new TypeError('Qwen3.5 needs a prepared graph input with a valid image count.');
    if (input.imageCount > 0 && (!vision || typeof vision.run !== 'function'))
      throw new TypeError('Qwen3.5 image input needs its pinned TVM vision runner.');

    let imageFeatures: readonly Float32Array[] = [];
    if (input.imageCount > 0) {
      let pending = imageFeaturesByInput.get(input);
      if (!pending) {
        pending = Promise.resolve().then(() => vision!.run(input, signal));
        imageFeaturesByInput.set(input, pending);
      }
      try {
        imageFeatures = await pending;
      } catch (cause) {
        if (imageFeaturesByInput.get(input) === pending) imageFeaturesByInput.delete(input);
        throw cause;
      }
    }
    signal?.throwIfAborted();
    return {
      imageFeatures,
      visionActivationPrecision,
      lookupTokenEmbeddings(tokenIds, options) {
        const activeSignal = options?.signal ?? signal;
        activeSignal?.throwIfAborted();
        return embeddings.lookupTokenEmbeddings(tokenIds, {signal: activeSignal});
      }
    };
  };
}
