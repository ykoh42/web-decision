// Internal still-image pixel core. Input is already-decoded RGB uint8 data.
const PIXEL_FACTOR = 32;
const LOAD_MIN_PIXELS = 4 * PIXEL_FACTOR ** 2;
const LOAD_MAX_PIXELS = 1_638_400;
const IMAGE_MIN_PIXELS = 65_536;
const IMAGE_MAX_PIXELS = 16_777_216;
const PATCH_SIZE = 16;
const MERGE_SIZE = 2;
const TEMPORAL_PATCH_SIZE = 2;
const PILLOW_PRECISION = 1 << 22;
const PATCH_VALUES = 3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE ** 2;

type ResizeWeight = [index: number, weight: number];
type ResizeWeights = ResizeWeight[][];
type ImageGrid = [number, number, number];
type PreprocessedKananaImage = {
  imageTokenThw: ImageGrid;
  pixelValues: Float32Array;
  visionGridThw: ImageGrid;
  width: number;
  height: number;
};

function assertImage(rgb: Uint8Array, width: number, height: number): void {
  if (!(rgb instanceof Uint8Array)) throw new TypeError('RGB pixels must be a Uint8Array');
  if (!Number.isSafeInteger(width) || width < 1 || !Number.isSafeInteger(height) || height < 1)
    throw new RangeError('image dimensions must be positive safe integers');
  const length = width * height * 3;
  if (!Number.isSafeInteger(length) || rgb.length !== length)
    throw new RangeError('RGB pixel length must match the image dimensions');
}

function roundToEvenMultiple(value: number, factor: number): number {
  const lower = Math.floor(value / factor);
  const remainder = value - lower * factor;
  const doubled = remainder * 2;
  if (doubled < factor) return lower * factor;
  if (doubled > factor) return (lower + 1) * factor;
  return (lower % 2 === 0 ? lower : lower + 1) * factor;
}

export function smartResize(
  height: number,
  width: number,
  minPixels: number,
  maxPixels: number,
  factor = PIXEL_FACTOR
): [number, number] {
  if (
    ![height, width, minPixels, maxPixels, factor].every(Number.isSafeInteger) ||
    height < 1 ||
    width < 1 ||
    minPixels < 1 ||
    maxPixels < minPixels ||
    factor < 1
  )
    throw new RangeError('resize dimensions and pixel limits must be positive safe integers');
  if (height < factor || width < factor)
    throw new RangeError(`height:${height} or width:${width} must be larger than factor:${factor}`);
  const aspectRatio = Math.max(height, width) / Math.min(height, width);
  if (aspectRatio > 200) {
    const formattedAspectRatio = Number.isInteger(aspectRatio) ? `${aspectRatio}.0` : String(aspectRatio);
    throw new RangeError(`absolute aspect ratio must be smaller than 200, got ${formattedAspectRatio}`);
  }

  let resizedHeight = Math.max(factor, roundToEvenMultiple(height, factor));
  let resizedWidth = Math.max(factor, roundToEvenMultiple(width, factor));

  if (resizedHeight * resizedWidth > maxPixels) {
    const scale = Math.sqrt((height * width) / maxPixels);
    resizedHeight = Math.floor(height / scale / factor) * factor;
    resizedWidth = Math.floor(width / scale / factor) * factor;
  } else if (resizedHeight * resizedWidth < minPixels) {
    const scale = Math.sqrt(minPixels / (height * width));
    resizedHeight = Math.ceil((height * scale) / factor) * factor;
    resizedWidth = Math.ceil((width * scale) / factor) * factor;
  }

  return [resizedHeight, resizedWidth];
}

function cubicPillow(value: number): number {
  const distance = Math.abs(value);
  if (distance < 1) return (1.5 * distance - 2.5) * distance * distance + 1;
  if (distance < 2) return ((-0.5 * distance + 2.5) * distance - 4) * distance + 2;
  return 0;
}

function pillowWeights(inputSize: number, outputSize: number): ResizeWeights {
  const scale = inputSize / outputSize;
  const filterScale = Math.max(scale, 1);
  const support = 2 * filterScale;
  const rows: ResizeWeights = [];
  rows.length = outputSize;

  for (let output = 0; output < outputSize; output++) {
    const center = (output + 0.5) * scale;
    const first = Math.max(0, Math.floor(center - support + 0.5));
    const end = Math.min(inputSize, Math.floor(center + support + 0.5));
    const row: ResizeWeight[] = [];
    let total = 0;

    for (let input = first; input < end; input++) {
      const weight = cubicPillow((input - center + 0.5) / filterScale);
      row.push([input, weight]);
      total += weight;
    }

    rows[output] = row.map<ResizeWeight>(([input, weight]) => [input, Math.round((weight / total) * PILLOW_PRECISION)]);
  }

  return rows;
}

function clampByte(value: number): number {
  return Math.max(0, Math.min(255, value));
}

function sumPillowHorizontal(
  input: Uint8Array,
  inputWidth: number,
  line: number,
  channel: number,
  weights: ResizeWeight[]
): number {
  let sum = 0;
  for (const [source, weight] of weights) sum += input[(line * inputWidth + source) * 3 + channel]! * weight;
  return sum;
}

function sumPillowVertical(
  input: Uint8Array,
  inputWidth: number,
  line: number,
  channel: number,
  weights: ResizeWeight[]
): number {
  let sum = 0;
  for (const [source, weight] of weights) sum += input[(source * inputWidth + line) * 3 + channel]! * weight;
  return sum;
}

function storePillowByte(
  output: Uint8Array,
  outputWidth: number,
  x: number,
  y: number,
  channel: number,
  sum: number
): void {
  output[(y * outputWidth + x) * 3 + channel] = clampByte(Math.floor((sum + PILLOW_PRECISION / 2) / PILLOW_PRECISION));
}

function resizePillowHorizontalLine(
  input: Uint8Array,
  output: Uint8Array,
  rows: ResizeWeights,
  width: number,
  outputWidth: number,
  line: number
): void {
  for (let x = 0; x < outputWidth; x++) {
    for (let channel = 0; channel < 3; channel++) {
      const sum = sumPillowHorizontal(input, width, line, channel, rows[x]!);
      storePillowByte(output, outputWidth, x, line, channel, sum);
    }
  }
}

function resizePillowVerticalLine(
  input: Uint8Array,
  output: Uint8Array,
  rows: ResizeWeights,
  width: number,
  outputWidth: number,
  outputHeight: number,
  line: number
): void {
  for (let y = 0; y < outputHeight; y++) {
    for (let channel = 0; channel < 3; channel++) {
      const sum = sumPillowVertical(input, width, line, channel, rows[y]!);
      storePillowByte(output, outputWidth, line, y, channel, sum);
    }
  }
}

function resizePillowAxis(
  input: Uint8Array,
  inputWidth: number,
  inputHeight: number,
  outputWidth: number,
  outputHeight: number,
  horizontal: boolean
): Uint8Array {
  const inputSize = horizontal ? inputWidth : inputHeight;
  const outputSize = horizontal ? outputWidth : outputHeight;
  const rows = pillowWeights(inputSize, outputSize);
  const output = new Uint8Array(outputWidth * outputHeight * 3);

  if (horizontal) {
    for (let line = 0; line < inputHeight; line++)
      resizePillowHorizontalLine(input, output, rows, inputWidth, outputWidth, line);
  } else {
    for (let line = 0; line < outputWidth; line++)
      resizePillowVerticalLine(input, output, rows, inputWidth, outputWidth, outputHeight, line);
  }

  return output;
}

export function resizePillowBicubic(
  rgb: Uint8Array,
  width: number,
  height: number,
  outputWidth: number,
  outputHeight: number
): Uint8Array {
  assertImage(rgb, width, height);
  if (!Number.isSafeInteger(outputWidth) || outputWidth < 1 || !Number.isSafeInteger(outputHeight) || outputHeight < 1)
    throw new RangeError('resize target dimensions must be positive safe integers');
  if (width === outputWidth && height === outputHeight) return rgb;

  const horizontal = resizePillowAxis(rgb, width, height, outputWidth, height, true);
  return resizePillowAxis(horizontal, outputWidth, height, outputWidth, outputHeight, false);
}

function cubicTorchvision(value: number): number {
  const distance = Math.abs(value);
  const coefficient = -0.5;
  if (distance < 1) return ((coefficient + 2) * distance - (coefficient + 3)) * distance * distance + 1;
  if (distance < 2)
    return ((coefficient * distance - 5 * coefficient) * distance + 8 * coefficient) * distance - 4 * coefficient;
  return 0;
}

function torchvisionUpscaleWeights(inputSize: number, outputSize: number): ResizeWeights {
  const scale = inputSize / outputSize;
  const rows: ResizeWeights = [];
  rows.length = outputSize;

  for (let output = 0; output < outputSize; output++) {
    const center = scale * (output + 0.5);
    const first = Math.max(Math.trunc(center - 2 + 0.5), 0);
    const end = Math.min(Math.trunc(center + 2 + 0.5), inputSize);
    const row: ResizeWeight[] = [];
    let total = 0;

    for (let input = first; input < end; input++) {
      const weight = cubicTorchvision(input - center + 0.5);
      row.push([input, weight]);
      total += weight;
    }

    rows[output] = row.map<ResizeWeight>(([input, weight]) => [input, weight / total]);
  }

  return rows;
}

function quantizeWeights(rows: ResizeWeights): {precision: number; rows: ResizeWeights} {
  let maximum = 0;
  for (const row of rows) {
    for (const [, weight] of row) maximum = Math.max(maximum, weight);
  }

  let precision = 0;
  while (precision < 21 && maximum * 2 ** (precision + 1) + 0.5 < 32_768) precision++;
  const scale = 2 ** precision;
  return {
    precision,
    rows: rows.map((row) =>
      row.map<ResizeWeight>(([index, weight]) => [index, Math.trunc(weight * scale + Math.sign(weight * scale) * 0.5)])
    )
  };
}

function roundedShift(value: number, precision: number): number {
  return Math.floor((value + 2 ** (precision - 1)) / 2 ** precision);
}

function sumTorchvisionHorizontal(
  input: Uint8Array,
  inputWidth: number,
  line: number,
  channel: number,
  weights: ResizeWeight[]
): number {
  let sum = 0;
  for (const [source, weight] of weights) sum += input[(line * inputWidth + source) * 3 + channel]! * weight;
  return sum;
}

function sumTorchvisionVertical(
  input: Uint8Array,
  inputWidth: number,
  line: number,
  channel: number,
  weights: ResizeWeight[]
): number {
  let sum = 0;
  for (const [source, weight] of weights) sum += input[(source * inputWidth + line) * 3 + channel]! * weight;
  return sum;
}

function storeTorchvisionByte(
  output: Uint8Array,
  outputWidth: number,
  x: number,
  y: number,
  channel: number,
  sum: number,
  precision: number
): void {
  output[(y * outputWidth + x) * 3 + channel] = clampByte(roundedShift(sum, precision));
}

function resizeTorchvisionHorizontalLine(
  input: Uint8Array,
  output: Uint8Array,
  rows: ResizeWeights,
  precision: number,
  width: number,
  outputWidth: number,
  line: number
): void {
  for (let x = 0; x < outputWidth; x++) {
    for (let channel = 0; channel < 3; channel++) {
      const sum = sumTorchvisionHorizontal(input, width, line, channel, rows[x]!);
      storeTorchvisionByte(output, outputWidth, x, line, channel, sum, precision);
    }
  }
}

function resizeTorchvisionVerticalLine(
  input: Uint8Array,
  output: Uint8Array,
  rows: ResizeWeights,
  precision: number,
  width: number,
  outputWidth: number,
  outputHeight: number,
  line: number
): void {
  for (let y = 0; y < outputHeight; y++) {
    for (let channel = 0; channel < 3; channel++) {
      const sum = sumTorchvisionVertical(input, width, line, channel, rows[y]!);
      storeTorchvisionByte(output, outputWidth, line, y, channel, sum, precision);
    }
  }
}

function resizeTorchvisionAxis(
  input: Uint8Array,
  inputWidth: number,
  inputHeight: number,
  outputWidth: number,
  outputHeight: number,
  horizontal: boolean
): Uint8Array {
  const inputSize = horizontal ? inputWidth : inputHeight;
  const outputSize = horizontal ? outputWidth : outputHeight;
  const weights = quantizeWeights(torchvisionUpscaleWeights(inputSize, outputSize));
  const output = new Uint8Array(outputWidth * outputHeight * 3);

  if (horizontal) {
    for (let line = 0; line < inputHeight; line++)
      resizeTorchvisionHorizontalLine(input, output, weights.rows, weights.precision, inputWidth, outputWidth, line);
  } else {
    for (let line = 0; line < outputWidth; line++)
      resizeTorchvisionVerticalLine(
        input,
        output,
        weights.rows,
        weights.precision,
        inputWidth,
        outputWidth,
        outputHeight,
        line
      );
  }

  return output;
}

export function resizeTorchvisionBicubicAntialias(
  rgb: Uint8Array,
  width: number,
  height: number,
  outputWidth: number,
  outputHeight: number
): Uint8Array {
  assertImage(rgb, width, height);
  if (
    !Number.isSafeInteger(outputWidth) ||
    outputWidth < width ||
    !Number.isSafeInteger(outputHeight) ||
    outputHeight < height
  )
    throw new RangeError('Kanana antialiased resize pass only upscales both dimensions');
  if (width === outputWidth && height === outputHeight) return rgb;

  const horizontal = resizeTorchvisionAxis(rgb, width, height, outputWidth, height, true);
  return resizeTorchvisionAxis(horizontal, outputWidth, height, outputWidth, outputHeight, false);
}

function normalizeByte(value: number): number {
  const rescaled = Math.fround(value * (1 / 255));
  const centered = Math.fround(rescaled - 0.5);
  return Math.fround(centered / 0.5);
}

const NORMALIZED_BYTE = Float32Array.from({length: 256}, (_, value) => normalizeByte(value));

function copyPatchPlane(
  pixelValues: Float32Array,
  rgb: Uint8Array,
  width: number,
  patchX: number,
  patchY: number,
  channel: number,
  offset: number
): number {
  for (let y = 0; y < PATCH_SIZE; y++) {
    const rowOffset = ((patchY + y) * width + patchX) * 3 + channel;
    for (let x = 0; x < PATCH_SIZE; x++) pixelValues[offset++] = NORMALIZED_BYTE[rgb[rowOffset + x * 3]!]!;
  }
  return offset;
}

function writePatch(
  pixelValues: Float32Array,
  rgb: Uint8Array,
  width: number,
  patchX: number,
  patchY: number,
  patch: number
): void {
  let offset = patch * PATCH_VALUES;
  for (let channel = 0; channel < 3; channel++) {
    for (let temporal = 0; temporal < TEMPORAL_PATCH_SIZE; temporal++)
      offset = copyPatchPlane(pixelValues, rgb, width, patchX, patchY, channel, offset);
  }
}

export function packKananaImagePatches(rgb: Uint8Array, width: number, height: number): Float32Array {
  assertImage(rgb, width, height);
  if (height % PIXEL_FACTOR !== 0 || width % PIXEL_FACTOR !== 0)
    throw new RangeError('Kanana image dimensions must be multiples of 32');

  const gridHeight = height / PATCH_SIZE;
  const gridWidth = width / PATCH_SIZE;
  const mergedHeight = gridHeight / MERGE_SIZE;
  const mergedWidth = gridWidth / MERGE_SIZE;
  const patchCount = gridHeight * gridWidth;
  const pixelValues = new Float32Array(patchCount * PATCH_VALUES);
  let patch = 0;

  for (let mergedY = 0; mergedY < mergedHeight; mergedY++) {
    for (let mergedX = 0; mergedX < mergedWidth; mergedX++) {
      for (let subY = 0; subY < MERGE_SIZE; subY++) {
        for (let subX = 0; subX < MERGE_SIZE; subX++) {
          const patchY = (mergedY * MERGE_SIZE + subY) * PATCH_SIZE;
          const patchX = (mergedX * MERGE_SIZE + subX) * PATCH_SIZE;
          writePatch(pixelValues, rgb, width, patchX, patchY, patch);
          patch++;
        }
      }
    }
  }

  return pixelValues;
}

export function preprocessKananaImage(rgb: Uint8Array, width: number, height: number): PreprocessedKananaImage {
  assertImage(rgb, width, height);

  const [loadHeight, loadWidth] = smartResize(height, width, LOAD_MIN_PIXELS, LOAD_MAX_PIXELS);
  let resized = resizePillowBicubic(rgb, width, height, loadWidth, loadHeight);
  const [imageHeight, imageWidth] = smartResize(loadHeight, loadWidth, IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS);

  if (imageWidth !== loadWidth || imageHeight !== loadHeight)
    resized = resizeTorchvisionBicubicAntialias(resized, loadWidth, loadHeight, imageWidth, imageHeight);

  const gridHeight = imageHeight / PATCH_SIZE;
  const gridWidth = imageWidth / PATCH_SIZE;
  const visionGridThw: ImageGrid = [1, gridHeight, gridWidth];
  const imageTokenThw: ImageGrid = [1, gridHeight / MERGE_SIZE, gridWidth / MERGE_SIZE];

  return {
    imageTokenThw,
    pixelValues: packKananaImagePatches(resized, imageWidth, imageHeight),
    visionGridThw,
    width: imageWidth,
    height: imageHeight
  };
}
