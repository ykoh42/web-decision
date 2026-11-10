import {KANANA_IMAGE_TOKEN_ID} from './contract.mjs';

type ImageTokenGrid = readonly [temporal: number, height: number, width: number];

export type KananaPositionInputs = Readonly<{
  /** Four axis-major rows: text positions, temporal, height, and width. */
  positionIds: Int32Array;
  positionShape: readonly [axes: 4, batch: 1, sequenceLength: number];
  imagePositions: Int32Array;
}>;

function validateGrid(value: unknown, index: number): ImageTokenGrid {
  if (
    !Array.isArray(value) ||
    value.length !== 3 ||
    !value.every((dimension) => Number.isSafeInteger(dimension) && dimension > 0)
  )
    throw new TypeError(`imageTokenThw for image ${index} must contain three positive safe integers.`);
  const [temporal, height, width] = value as [number, number, number];
  if (!Number.isSafeInteger(temporal * height * width))
    throw new RangeError(`imageTokenThw for image ${index} exceeds the safe integer range.`);
  return [temporal, height, width];
}

function snapshotInputIds(inputIds: ArrayLike<number>): number[] {
  if (!inputIds || !Number.isSafeInteger(inputIds.length) || inputIds.length < 1)
    throw new TypeError('inputIds must be a nonempty indexed sequence.');
  const ids: number[] = [];
  for (let index = 0; index < inputIds.length; index++) {
    const id = inputIds[index];
    if (!Number.isSafeInteger(id) || id! < 0)
      throw new TypeError(`inputIds[${index}] must be a nonnegative safe integer.`);
    ids.push(id!);
  }
  return ids;
}

function findRunEnd(ids: readonly number[], start: number, isImage: boolean, imageTokenId: number): number {
  let end = start + 1;
  while (end < ids.length && (ids[end] === imageTokenId) === isImage) end++;
  return end;
}

function writeTextRun(
  positions: Int32Array,
  sequenceLength: number,
  start: number,
  end: number,
  currentPosition: number
): number {
  for (let index = start; index < end; index++) {
    const position = currentPosition + index - start;
    positions[index] = index;
    positions[sequenceLength + index] = position;
    positions[2 * sequenceLength + index] = position;
    positions[3 * sequenceLength + index] = position;
  }
  return currentPosition + end - start;
}

function writeImageRun(
  positions: Int32Array,
  imagePositions: number[],
  sequenceLength: number,
  start: number,
  end: number,
  currentPosition: number,
  grid: ImageTokenGrid | undefined,
  imageIndex: number
): number {
  if (!grid) throw new RangeError('Kanana input has more image token runs than image grids.');
  const [temporal, height, width] = grid;
  const tokenCount = temporal * height * width;
  if (end - start !== tokenCount)
    throw new RangeError(`Kanana image ${imageIndex} has ${end - start} tokens; expected ${tokenCount}.`);

  let offset = start;
  for (let time = 0; time < temporal; time++) {
    for (let row = 0; row < height; row++) {
      for (let column = 0; column < width; column++) {
        positions[offset] = offset;
        positions[sequenceLength + offset] = currentPosition + time;
        positions[2 * sequenceLength + offset] = currentPosition + row;
        positions[3 * sequenceLength + offset] = currentPosition + column;
        imagePositions.push(offset);
        offset++;
      }
    }
  }
  return currentPosition + Math.max(height, width);
}

/** Build the pinned Qwen3-VL M-RoPE positions used by Kanana, including image-token order. */
export function prepareKananaPositionInputs(
  inputIds: ArrayLike<number>,
  imageTokenThw: readonly unknown[],
  imageTokenId = KANANA_IMAGE_TOKEN_ID
): KananaPositionInputs {
  const ids = snapshotInputIds(inputIds);
  if (!Array.isArray(imageTokenThw)) throw new TypeError('imageTokenThw must be an array.');
  if (!Number.isSafeInteger(imageTokenId) || imageTokenId < 0)
    throw new RangeError('imageTokenId must be a nonnegative safe integer.');

  const grids = imageTokenThw.map(validateGrid);
  const sequenceLength = ids.length;
  const positions = new Int32Array(4 * sequenceLength);
  const imagePositions: number[] = [];
  let imageIndex = 0;
  let currentPosition = 0;
  let tokenIndex = 0;

  while (tokenIndex < sequenceLength) {
    const isImage = ids[tokenIndex] === imageTokenId;
    const end = findRunEnd(ids, tokenIndex, isImage, imageTokenId);
    currentPosition = isImage
      ? writeImageRun(
          positions,
          imagePositions,
          sequenceLength,
          tokenIndex,
          end,
          currentPosition,
          grids[imageIndex],
          imageIndex
        )
      : writeTextRun(positions, sequenceLength, tokenIndex, end, currentPosition);
    if (isImage) imageIndex++;
    tokenIndex = end;
  }

  if (imageIndex !== grids.length) throw new RangeError('Kanana image grid count does not match image token runs.');
  return Object.freeze({
    positionIds: positions,
    positionShape: Object.freeze([4, 1, sequenceLength]) as readonly [4, 1, number],
    imagePositions: Int32Array.from(imagePositions)
  });
}
