const DEFAULT_MAX_IMAGES = 4;
const MIN_PATCH_ROWS = 256;
// Pinned Qwen2VLImageProcessorPil min/max pixels divided by patch_size² (16²).
const MAX_PATCH_ROWS = 65_536;
const MAX_INT32 = 0x7fffffff;

export type Qwen35VisionGeometryConfig = Readonly<{
  numPositionEmbeddings: number;
  spatialMergeSize: number;
  maxImages?: number;
}>;

export type Qwen35PerImageVisionGeometry = Readonly<{
  gridThw: readonly [frames: 1, gridHeight: number, gridWidth: number];
  gridThwInputShape: readonly [images: 1, fields: 3];
  gridThwInput: Int32Array;
  patchRows: number;
  featureRows: number;
  interpIndicesShape: readonly [patchRows: number, taps: 4];
  interpWeightsShape: readonly [patchRows: number, taps: 4];
  interpIndices: Int32Array;
  interpWeights: Float32Array;
  positionIdsShape: readonly [featureRows: number, mergeCoordinates: 8];
  positionIdsByToken: Int32Array;
  cuSeqlensShape: readonly [boundaries: 2];
  cuSeqlens: Int32Array;
}>;

function assertConfig(config: Qwen35VisionGeometryConfig): number {
  if (!Number.isSafeInteger(config.numPositionEmbeddings) || config.numPositionEmbeddings < 4)
    throw new RangeError('Qwen3.5 numPositionEmbeddings must be a safe integer of at least 4.');
  if (!Number.isSafeInteger(config.spatialMergeSize) || config.spatialMergeSize < 1)
    throw new RangeError('Qwen3.5 spatialMergeSize must be a positive safe integer.');
  if (config.spatialMergeSize !== 2) throw new RangeError('This Qwen3.5 per-image graph requires spatialMergeSize=2.');
  const side = Math.floor(Math.sqrt(config.numPositionEmbeddings));
  if (side < 2) throw new RangeError('Qwen3.5 position embedding side must be at least 2.');
  return side;
}

function assertGrid(grid: readonly [number, number, number], merge: number): number {
  const [frames, height, width] = grid;
  if (![frames, height, width].every(Number.isSafeInteger) || frames !== 1 || height < 1 || width < 1)
    throw new RangeError('Qwen3.5 per-image geometry requires one positive integer [T,H,W] grid with T=1.');
  if (height % merge !== 0 || width % merge !== 0)
    throw new RangeError(`Qwen3.5 vision grid height and width must be multiples of ${merge}.`);
  const patchRows = height * width;
  if (!Number.isSafeInteger(patchRows) || patchRows < MIN_PATCH_ROWS || patchRows > MAX_PATCH_ROWS)
    throw new RangeError(
      `Qwen3.5 one-image graph requires ${MIN_PATCH_ROWS} <= P <= ${MAX_PATCH_ROWS}; got ${patchRows}.`
    );
  return patchRows;
}

export function getQwen35MaximumPackedPatchRows(maxImages = DEFAULT_MAX_IMAGES): number {
  if (!Number.isSafeInteger(maxImages) || maxImages < 1)
    throw new RangeError('Qwen3.5 maxImages must be a positive safe integer.');
  const processorBound = MAX_PATCH_ROWS * maxImages;
  if (!Number.isSafeInteger(processorBound))
    throw new RangeError('Qwen3.5 packed patch rows exceed the safe integer range.');
  return Math.min(processorBound, MAX_INT32);
}

function f32(value: number): number {
  return Math.fround(value);
}

function subtractF32(left: number, right: number): number {
  return f32(left - right);
}

function multiplyF32(left: number, right: number): number {
  return f32(left * right);
}

function divideF32(left: number, right: number): number {
  return f32(left / right);
}

type BilinearAxis = Readonly<{
  indices: readonly [number, number];
  weights: readonly [number, number];
}>;

function bilinearAxis(index: number, size: number, side: number): BilinearAxis {
  // Transformers 5.22.2 Qwen3.5 uses bilinear interpolation with align_corners=True.
  const source = divideF32(multiplyF32(f32(index), side - 1), Math.max(size - 1, 1));
  const floor = Math.floor(source);
  const fromFloor = subtractF32(source, floor);
  const indices: [number, number] = [0, 0];
  const weights: [number, number] = [0, 0];
  for (let offset = 0; offset < 2; offset += 1) {
    indices[offset] = Math.max(0, Math.min(side - 1, floor + offset));
    const distance = Math.abs(subtractF32(fromFloor, offset));
    weights[offset] = Math.max(subtractF32(1, distance), 0);
  }
  return {indices, weights};
}

function writeBilinearTaps(
  height: BilinearAxis,
  width: BilinearAxis,
  side: number,
  patch: number,
  indices: Int32Array,
  weights: Float32Array
): void {
  let target = patch * 4;
  for (let heightTap = 0; heightTap < 2; heightTap += 1) {
    for (let widthTap = 0; widthTap < 2; widthTap += 1) {
      indices[target] = height.indices[heightTap]! * side + width.indices[widthTap]!;
      weights[target] = multiplyF32(height.weights[heightTap]!, width.weights[widthTap]!);
      target += 1;
    }
  }
}

function buildImageGeometry(
  gridThw: readonly [1, number, number],
  config: Qwen35VisionGeometryConfig,
  side: number
): Qwen35PerImageVisionGeometry {
  const merge = config.spatialMergeSize;
  const patchRows = assertGrid(gridThw, merge);
  const featureRows = patchRows / (merge * merge);
  if (!Number.isSafeInteger(featureRows)) throw new RangeError('Qwen3.5 patch rows do not align to the spatial merge.');

  const interpIndices = new Int32Array(patchRows * 4);
  const interpWeights = new Float32Array(patchRows * 4);
  const positionIdsByToken = new Int32Array(patchRows * 2);
  let patch = 0;
  for (let blockRow = 0; blockRow < gridThw[1] / merge; blockRow += 1) {
    for (let blockColumn = 0; blockColumn < gridThw[2] / merge; blockColumn += 1) {
      for (let innerRow = 0; innerRow < merge; innerRow += 1) {
        for (let innerColumn = 0; innerColumn < merge; innerColumn += 1) {
          const row = blockRow * merge + innerRow;
          const column = blockColumn * merge + innerColumn;
          const height = bilinearAxis(row, gridThw[1], side);
          const width = bilinearAxis(column, gridThw[2], side);
          writeBilinearTaps(height, width, side, patch, interpIndices, interpWeights);
          positionIdsByToken[patch * 2] = row;
          positionIdsByToken[patch * 2 + 1] = column;
          patch += 1;
        }
      }
    }
  }

  return {
    gridThw,
    gridThwInputShape: [1, 3],
    gridThwInput: new Int32Array(gridThw),
    patchRows,
    featureRows,
    interpIndicesShape: [patchRows, 4],
    interpWeightsShape: [patchRows, 4],
    interpIndices,
    interpWeights,
    positionIdsShape: [featureRows, 8],
    positionIdsByToken,
    cuSeqlensShape: [2],
    cuSeqlens: new Int32Array([0, patchRows])
  };
}

export function buildQwen35PerImageVisionGeometry(
  imageGridThw: readonly (readonly [1, number, number])[],
  config: Qwen35VisionGeometryConfig
): readonly Qwen35PerImageVisionGeometry[] {
  const maxImages = config.maxImages ?? DEFAULT_MAX_IMAGES;
  if (!Number.isSafeInteger(maxImages) || maxImages < 0)
    throw new RangeError('Qwen3.5 maxImages must be a non-negative safe integer.');
  if (imageGridThw.length < 1 || imageGridThw.length > maxImages)
    throw new RangeError(`Qwen3.5 vision geometry requires 1 to ${maxImages} images.`);
  const side = assertConfig(config);
  return imageGridThw.map((gridThw) => buildImageGeometry(gridThw, config, side));
}

export function buildQwen35PackedVisionCuSeqlens(
  imageGeometry: readonly Qwen35PerImageVisionGeometry[],
  maxImages = DEFAULT_MAX_IMAGES
): Int32Array {
  const maximumPackedPatchRows = getQwen35MaximumPackedPatchRows(maxImages);
  if (imageGeometry.length < 1 || imageGeometry.length > maxImages)
    throw new RangeError(`Qwen3.5 packed vision requires 1 to ${maxImages} images.`);

  const boundaries = new Int32Array(imageGeometry.length + 1);
  let totalRows = 0;
  for (let index = 0; index < imageGeometry.length; index += 1) {
    const image = imageGeometry[index]!;
    if (image.cuSeqlens.length !== 2 || image.cuSeqlens[0] !== 0 || image.cuSeqlens[1] !== image.patchRows)
      throw new RangeError(`Qwen3.5 image ${index} must carry the source [0,P] segment.`);
    totalRows += image.patchRows;
    if (!Number.isSafeInteger(totalRows) || totalRows > maximumPackedPatchRows)
      throw new RangeError('Qwen3.5 packed patch rows exceed the int32 cu_seqlens range.');
    boundaries[index + 1] = totalRows;
  }
  return boundaries;
}
