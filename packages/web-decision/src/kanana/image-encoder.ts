import {KANANA_EMBEDDING_SIZE, KANANA_MODEL_MAX_LENGTH} from './contract.mjs';
import {decodeKananaImageSource} from './browser-image.js';
import {preprocessKananaImage} from './image-processing.js';

type ImageGrid = readonly [temporal: number, height: number, width: number];

type DecodedImage = Readonly<{
  rgb: Uint8Array;
  width: number;
  height: number;
}>;

type PreprocessedImage = Readonly<{
  imageTokenThw: ImageGrid;
  pixelValues: Float32Array;
  visionGridThw: ImageGrid;
}>;

type VisionInferenceSession = Readonly<{
  infer(
    pixelValues: ArrayLike<number>,
    visionGridThw: ImageGrid
  ): PromiseLike<{
    shape: readonly number[];
    dtype: 'float32';
    values: Float32Array;
  }>;
}>;

export type KananaImageEmbeddingEncoderOptions = Readonly<{
  vision: VisionInferenceSession;
  decode?: (source: unknown, signal?: AbortSignal) => DecodedImage | PromiseLike<DecodedImage>;
  preprocess?: (rgb: Uint8Array, width: number, height: number) => PreprocessedImage;
}>;

function snapshotGrid(value: unknown, index: number): ImageGrid {
  if (
    !Array.isArray(value) ||
    value.length !== 3 ||
    !value.every((dimension) => Number.isSafeInteger(dimension) && dimension > 0)
  )
    throw new TypeError(`imageTokenThw for image ${index} must contain three positive safe integers.`);
  return Object.freeze([...value]) as unknown as ImageGrid;
}

function sameGrid(left: ImageGrid, right: ImageGrid): boolean {
  return left.every((dimension, index) => dimension === right[index]);
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  signal?.throwIfAborted();
}

/** Decode, preprocess, and encode each original image once in prompt order. */
export function createKananaTvmImageEmbeddingEncoder({
  vision,
  decode = (source, signal) => decodeKananaImageSource(source, signal ? {signal} : {}),
  preprocess = preprocessKananaImage
}: KananaImageEmbeddingEncoderOptions) {
  if (!vision || typeof vision.infer !== 'function') throw new TypeError('A Kanana TVM vision session is required.');
  if (typeof decode !== 'function') throw new TypeError('Kanana image decode must be a function.');
  if (typeof preprocess !== 'function') throw new TypeError('Kanana image preprocess must be a function.');

  return async function prepareImageEmbeddings(
    sources: readonly unknown[],
    gridValues: readonly ImageGrid[],
    signal?: AbortSignal
  ) {
    if (!Array.isArray(sources) || !Array.isArray(gridValues) || sources.length !== gridValues.length)
      throw new RangeError('Kanana image sources and imageTokenThw grids must have the same length.');
    const grids = gridValues.map(snapshotGrid);
    const tokenCount = grids.reduce((total, [temporal, height, width]) => total + temporal * height * width, 0);
    if (!Number.isSafeInteger(tokenCount) || tokenCount > KANANA_MODEL_MAX_LENGTH)
      throw new RangeError(`Kanana image tokens exceed the model sequence limit of ${KANANA_MODEL_MAX_LENGTH}.`);
    const values = new Float32Array(tokenCount * KANANA_EMBEDDING_SIZE);
    let valueOffset = 0;

    for (let index = 0; index < sources.length; index++) {
      throwIfAborted(signal);
      const decoded = await decode(sources[index], signal);
      throwIfAborted(signal);
      const prepared = preprocess(decoded.rgb, decoded.width, decoded.height);
      const grid = grids[index]!;
      if (!sameGrid(prepared.imageTokenThw, grid))
        throw new RangeError(`Kanana image ${index} preprocessor grid does not match the prompt token grid.`);
      const encoded = await vision.infer(prepared.pixelValues, prepared.visionGridThw);
      throwIfAborted(signal);
      const expectedTokens = grid[0] * grid[1] * grid[2];
      if (
        encoded.dtype !== 'float32' ||
        encoded.shape.length !== 2 ||
        encoded.shape[0] !== expectedTokens ||
        encoded.shape[1] !== KANANA_EMBEDDING_SIZE ||
        encoded.values.length !== expectedTokens * KANANA_EMBEDDING_SIZE
      )
        throw new RangeError(`Kanana vision output ${index} does not match its merged image-token grid.`);
      values.set(encoded.values, valueOffset);
      valueOffset += encoded.values.length;
    }

    return Object.freeze({
      shape: Object.freeze([tokenCount, KANANA_EMBEDDING_SIZE]) as readonly [number, number],
      values
    });
  };
}
