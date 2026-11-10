import {KANANA_SPATIAL_MERGE_SIZE} from './contract.mjs';

type VisionGrid = readonly [temporal: number, height: number, width: number];

export type KananaVisionGraphInputs = Readonly<{
  positionIds: Int32Array;
  positionShape: readonly [patches: number, axes: 2];
  interpolationIndices: Int32Array;
  interpolationWeights: Float32Array;
  interpolationShape: readonly [patches: number, taps: 4];
  mergedTokens: Int32Array;
}>;

type AxisTaps = readonly [indices: readonly [number, number], weights: readonly [number, number]];

function validateGrid(value: unknown): VisionGrid {
  if (
    !Array.isArray(value) ||
    value.length !== 3 ||
    !value.every((dimension) => Number.isSafeInteger(dimension) && dimension > 0)
  )
    throw new TypeError('visionGridThw must contain three positive safe integers.');
  const [temporal, height, width] = value as [number, number, number];
  if (!Number.isSafeInteger(temporal * height * width))
    throw new RangeError('visionGridThw exceeds the safe integer range.');
  return [temporal, height, width];
}

function axisTaps(index: number, size: number, side: number): AxisTaps {
  const source = (index * (side - 1)) / Math.max(size - 1, 1);
  const floor = Math.floor(source);
  const first = source - floor;
  const second = 1 - first;
  return [
    [Math.min(floor, side - 1), Math.min(floor + 1, side - 1)],
    [second, first]
  ];
}

function axisTable(size: number, side: number): AxisTaps[] {
  return Array.from({length: size}, (_, index) => axisTaps(index, size, side));
}

type VisionBuffers = Readonly<{
  positionIds: Int32Array;
  interpolationIndices: Int32Array;
  interpolationWeights: Float32Array;
  heightTaps: readonly AxisTaps[];
  widthTaps: readonly AxisTaps[];
  grid: VisionGrid;
  side: number;
  spatialMergeSize: number;
}>;

function writeVisionPatch(buffers: VisionBuffers, patch: number, row: number, column: number): void {
  const {positionIds, interpolationIndices, interpolationWeights, heightTaps, widthTaps, side} = buffers;
  positionIds[patch * 2] = row;
  positionIds[patch * 2 + 1] = column;

  const [rows, rowWeights] = heightTaps[row]!;
  const [columns, columnWeights] = widthTaps[column]!;
  const offset = patch * 4;
  interpolationIndices[offset] = rows[0] * side + columns[0];
  interpolationIndices[offset + 1] = rows[0] * side + columns[1];
  interpolationIndices[offset + 2] = rows[1] * side + columns[0];
  interpolationIndices[offset + 3] = rows[1] * side + columns[1];
  interpolationWeights[offset] = rowWeights[0] * columnWeights[0];
  interpolationWeights[offset + 1] = rowWeights[0] * columnWeights[1];
  interpolationWeights[offset + 2] = rowWeights[1] * columnWeights[0];
  interpolationWeights[offset + 3] = rowWeights[1] * columnWeights[1];
}

function writeMergedBlock(
  buffers: VisionBuffers,
  time: number,
  blockY: number,
  blockX: number,
  framePatch: number
): number {
  const [, height, width] = buffers.grid;
  const {spatialMergeSize} = buffers;
  for (let innerY = 0; innerY < spatialMergeSize; innerY++) {
    for (let innerX = 0; innerX < spatialMergeSize; innerX++) {
      const patch = time * height * width + framePatch;
      const row = blockY * spatialMergeSize + innerY;
      const column = blockX * spatialMergeSize + innerX;
      writeVisionPatch(buffers, patch, row, column);
      framePatch++;
    }
  }
  return framePatch;
}

function writeVisionFrame(buffers: VisionBuffers, time: number): void {
  const [, height, width] = buffers.grid;
  const {spatialMergeSize} = buffers;
  let framePatch = 0;
  for (let blockY = 0; blockY < height / spatialMergeSize; blockY++) {
    for (let blockX = 0; blockX < width / spatialMergeSize; blockX++) {
      framePatch = writeMergedBlock(buffers, time, blockY, blockX, framePatch);
    }
  }
}

/** Reproduce the pinned Transformers Qwen3-VL rotary positions and bilinear position-table lookup. */
export function prepareKananaVisionGraphInputs(
  gridValue: unknown,
  {
    numPositionEmbeddings,
    spatialMergeSize = KANANA_SPATIAL_MERGE_SIZE
  }: {
    numPositionEmbeddings: number;
    spatialMergeSize?: number;
  }
): KananaVisionGraphInputs {
  const [temporal, height, width] = validateGrid(gridValue);
  if (
    !Number.isSafeInteger(spatialMergeSize) ||
    spatialMergeSize < 1 ||
    height % spatialMergeSize !== 0 ||
    width % spatialMergeSize !== 0
  )
    throw new RangeError('visionGridThw height and width must be divisible by spatialMergeSize.');
  if (!Number.isSafeInteger(numPositionEmbeddings) || numPositionEmbeddings < 1)
    throw new RangeError('numPositionEmbeddings must be a positive safe integer.');

  const side = Math.sqrt(numPositionEmbeddings);
  if (!Number.isSafeInteger(side)) throw new RangeError('numPositionEmbeddings must form a square table.');
  const patchCount = temporal * height * width;
  const mergedTokenCount = patchCount / spatialMergeSize ** 2;
  if (!Number.isSafeInteger(mergedTokenCount))
    throw new RangeError('visionGridThw must produce a whole number of merged tokens.');

  const positionIds = new Int32Array(patchCount * 2);
  const interpolationIndices = new Int32Array(patchCount * 4);
  const interpolationWeights = new Float32Array(patchCount * 4);
  const heightTaps = axisTable(height, side);
  const widthTaps = axisTable(width, side);
  const buffers: VisionBuffers = {
    positionIds,
    interpolationIndices,
    interpolationWeights,
    heightTaps,
    widthTaps,
    grid: [temporal, height, width],
    side,
    spatialMergeSize
  };
  for (let time = 0; time < temporal; time++) writeVisionFrame(buffers, time);

  return Object.freeze({
    positionIds,
    positionShape: Object.freeze([patchCount, 2]) as readonly [number, 2],
    interpolationIndices,
    interpolationWeights,
    interpolationShape: Object.freeze([patchCount, 4]) as readonly [number, 4],
    mergedTokens: new Int32Array(mergedTokenCount)
  });
}
