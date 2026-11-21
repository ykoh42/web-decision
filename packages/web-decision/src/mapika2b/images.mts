import {decode as decodePng} from 'fast-png';
import {
  decodeKananaEncodedImageSource,
  preflightKananaImageHeader,
  type KananaImageDecodeOptions
} from '../kanana/encoded-image.ts';
import {MAPIKA2B_VISION, type Mapika2bImageAsset} from './contract.mts';

const {patchSize, mergeSize, maxPixels} = MAPIKA2B_VISION;
const SIZE_FACTOR = patchSize * mergeSize;
const MIN_PIXELS = 65_536;
const CHANNELS = 3;
const TEMPORAL_PATCH_SIZE = 2;
const PATCH_VALUES = CHANNELS * TEMPORAL_PATCH_SIZE * patchSize ** 2;
const NORMALIZED_BYTES = Float32Array.from({length: 256}, (_, value) => (value - 127.5) / 127.5);

type ImageSize = {readonly width: number; readonly height: number};
type PixelImage = ImageSize & {readonly rgb: Uint8Array};
type Weights<T extends Float64Array | Int16Array> = {readonly start: number; readonly values: T};

export type Mapika2bImageOptions = Pick<KananaImageDecodeOptions, 'signal' | 'magickWasm'> &
  Readonly<{
    prepareJpeg?: () => Promise<void>;
    loadMagick?: () => Promise<Uint8Array>;
  }>;

export interface Mapika2bImageInput {
  readonly imageGridThw: [1, number, number];
  readonly pixelShape: [number, number];
  readonly pixelValues: Float32Array;
  readonly decodedSize: ImageSize;
  readonly processedSize: ImageSize;
}

function decodeRgbaPng(bytes: Uint8Array): PixelImage {
  const {width, height, depth, channels, data} = decodePng(bytes, {checkCrc: true});
  const pixels = width * height;
  if (channels !== 4 || (depth !== 8 && depth !== 16) || data.length !== pixels * channels)
    throw new TypeError('Mapika PNG returned invalid RGBA pixels.');

  const shift = depth - 8;
  const rgb = new Uint8Array(pixels * CHANNELS);
  for (let source = 0, target = 0; target < rgb.length; source += channels, target += CHANNELS) {
    rgb[target] = data[source]! >>> shift;
    rgb[target + 1] = data[source + 1]! >>> shift;
    rgb[target + 2] = data[source + 2]! >>> shift;
  }
  return {width, height, rgb};
}

async function decodePackedAdam7Png(bytes: Uint8Array, options: Mapika2bImageOptions): Promise<PixelImage> {
  preflightKananaImageHeader(bytes);
  const {magickRuntime} = await import('../kanana/magick-image.ts');
  const {ImageMagick} = await magickRuntime(options.magickWasm);
  options.signal?.throwIfAborted();
  return ImageMagick.read(bytes, (image) => {
    const {width, height} = image;
    const rgb = image.getPixels((pixels) => pixels.toByteArray(0, 0, width, height, 'RGB'));
    if (!rgb || rgb.length !== width * height * CHANNELS) {
      throw new TypeError('Mapika PNG returned incomplete RGB pixels.');
    }
    return {width, height, rgb};
  });
}

async function decodeImage(bytes: Uint8Array, options: Mapika2bImageOptions): Promise<PixelImage> {
  const header = preflightKananaImageHeader(bytes);
  const isPackedAdam7 = header?.format === 'png' && bytes[24]! < 8 && bytes[28] === 1;
  // fast-png's Adam7 decoder uses byte strides and cannot unpack depths below eight bits.
  if (isPackedAdam7) {
    const magickWasm = await options.loadMagick?.();
    return decodePackedAdam7Png(bytes, {...options, magickWasm: magickWasm ?? options.magickWasm});
  }
  // Pillow's RGBA to RGB conversion discards alpha without compositing.
  if (header?.format === 'png' && bytes[25] === 6) return decodeRgbaPng(bytes);

  let magickWasm = options.magickWasm;
  if (header?.format === 'jpeg') {
    if (header.components === 4) magickWasm = (await options.loadMagick?.()) ?? magickWasm;
    else await options.prepareJpeg?.();
  }
  return decodeKananaEncodedImageSource(new Blob([new Uint8Array(bytes)]), {...options, magickWasm});
}

function roundToFactor(size: number): number {
  const units = size / SIZE_FACTOR;
  const lower = Math.floor(units);
  if (units - lower === 0.5) return (lower + (lower % 2)) * SIZE_FACTOR;
  return Math.round(units) * SIZE_FACTOR;
}

function getProcessedSize({width, height}: ImageSize): ImageSize {
  const aspectRatio = Math.max(width, height) / Math.min(width, height);
  if (aspectRatio > 200) throw new RangeError(`Mapika image aspect ratio must be at most 200, got ${aspectRatio}.`);

  const roundedWidth = roundToFactor(width);
  const roundedHeight = roundToFactor(height);
  const roundedArea = roundedWidth * roundedHeight;
  if (roundedArea > maxPixels) {
    const scale = Math.sqrt((width * height) / maxPixels);
    return {
      width: Math.max(SIZE_FACTOR, Math.floor(width / scale / SIZE_FACTOR) * SIZE_FACTOR),
      height: Math.max(SIZE_FACTOR, Math.floor(height / scale / SIZE_FACTOR) * SIZE_FACTOR)
    };
  }
  if (roundedArea < MIN_PIXELS) {
    const scale = Math.sqrt(MIN_PIXELS / (width * height));
    return {
      width: Math.ceil((width * scale) / SIZE_FACTOR) * SIZE_FACTOR,
      height: Math.ceil((height * scale) / SIZE_FACTOR) * SIZE_FACTOR
    };
  }
  return {width: roundedWidth, height: roundedHeight};
}

function cubicWeight(distance: number): number {
  const x = Math.abs(distance);
  if (x < 1) return (1.5 * x - 2.5) * x * x + 1;
  if (x < 2) return ((-0.5 * x + 2.5) * x - 4) * x + 2;
  return 0;
}

function filterRow(inputSize: number, scale: number, output: number): Weights<Float64Array> {
  const center = (output + 0.5) * scale;
  const spread = Math.max(1, scale);
  const support = 2 * spread;
  const inverseSpread = 1 / spread;
  const start = Math.max(0, Math.trunc(center - support + 0.5));
  const end = Math.min(inputSize, Math.trunc(center + support + 0.5));
  const values = new Float64Array(end - start);
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    const weight = cubicWeight((start + i - center + 0.5) * inverseSpread);
    values[i] = weight;
    sum += weight;
  }
  for (let i = 0; i < values.length; i++) values[i] = values[i]! / sum;
  return {start, values};
}

function getFilter(inputSize: number, outputSize: number) {
  const scale = inputSize / outputSize;
  const rows = Array.from({length: outputSize}, (_, index) => filterRow(inputSize, scale, index));
  let maxWeight = 0;
  for (const {values} of rows) {
    for (const weight of values) maxWeight = Math.max(maxWeight, weight);
  }
  let precision = 0;
  while (precision < 22 && maxWeight * 2 ** (precision + 1) + 0.5 < 32_768) precision++;
  const unit = 2 ** precision;
  return {
    unit,
    rows: rows.map(({start, values}): Weights<Int16Array> => ({
      start,
      values: Int16Array.from(values, (weight) => Math.trunc(weight * unit + Math.sign(weight) * 0.5))
    }))
  };
}

function sample(rgb: Uint8Array, offset: number, stride: number, weights: Int16Array, unit: number): number {
  let sum = unit / 2;
  for (const weight of weights) {
    sum += rgb[offset]! * weight;
    offset += stride;
  }
  return Math.min(255, Math.max(0, Math.floor(sum / unit)));
}

function resizeAxis(image: PixelImage, outputSize: number, horizontal: boolean): PixelImage {
  const inputSize = horizontal ? image.width : image.height;
  if (inputSize === outputSize) return image;

  const width = horizontal ? outputSize : image.width;
  const height = horizontal ? image.height : outputSize;
  const rgb = new Uint8Array(width * height * CHANNELS);
  const {unit, rows} = getFilter(inputSize, outputSize);
  const lineCount = horizontal ? height : width;
  const sourceStride = (horizontal ? 1 : image.width) * CHANNELS;
  const sourceLineStride = (horizontal ? image.width : 1) * CHANNELS;
  const targetStride = (horizontal ? 1 : width) * CHANNELS;
  const targetLineStride = (horizontal ? width : 1) * CHANNELS;
  for (let line = 0; line < lineCount; line++) {
    let target = line * targetLineStride;
    for (const {start, values} of rows) {
      const source = line * sourceLineStride + start * sourceStride;
      for (let channel = 0; channel < CHANNELS; channel++) {
        rgb[target + channel] = sample(image.rgb, source + channel, sourceStride, values, unit);
      }
      target += targetStride;
    }
  }
  return {width, height, rgb};
}

function writePatch(image: PixelImage, pixels: Float32Array, patchX: number, patchY: number, offset: number): void {
  const stride = image.width * CHANNELS;
  const start = patchY * stride + patchX * CHANNELS;
  for (let channel = 0; channel < CHANNELS; channel++) {
    const planeStart = offset;
    for (let y = 0; y < patchSize; y++) {
      let source = start + y * stride + channel;
      for (let x = 0; x < patchSize; x++, source += CHANNELS) {
        pixels[offset++] = NORMALIZED_BYTES[image.rgb[source]!]!;
      }
    }
    pixels.copyWithin(offset, planeStart, offset);
    offset += patchSize ** 2;
  }
}

function packPatches(image: PixelImage): Float32Array {
  const pixels = new Float32Array((image.width / patchSize) * (image.height / patchSize) * PATCH_VALUES);
  let offset = 0;
  for (let y = 0; y < image.height; y += SIZE_FACTOR) {
    for (let x = 0; x < image.width; x += SIZE_FACTOR) {
      for (let subY = 0; subY < SIZE_FACTOR; subY += patchSize) {
        for (let subX = 0; subX < SIZE_FACTOR; subX += patchSize) {
          writePatch(image, pixels, x + subX, y + subY, offset);
          offset += PATCH_VALUES;
        }
      }
    }
  }
  return pixels;
}

/** Decode and pack the pinned Mapika Qwen2VL vision input for a single image occurrence. */
export async function preprocessMapika2bImage(
  asset: Mapika2bImageAsset,
  options: Mapika2bImageOptions = {}
): Promise<Mapika2bImageInput> {
  options.signal?.throwIfAborted();
  const decoded = await decodeImage(asset.data, options);
  options.signal?.throwIfAborted();
  const processedSize = getProcessedSize(decoded);
  // Torchvision's CPU uint8 BICUBIC path rounds and clips each pass, width first.
  const resized = resizeAxis(resizeAxis(decoded, processedSize.width, true), processedSize.height, false);
  const gridHeight = processedSize.height / patchSize;
  const gridWidth = processedSize.width / patchSize;
  return {
    imageGridThw: [1, gridHeight, gridWidth],
    pixelShape: [gridHeight * gridWidth, PATCH_VALUES],
    pixelValues: packPatches(resized),
    decodedSize: {width: decoded.width, height: decoded.height},
    processedSize
  };
}
