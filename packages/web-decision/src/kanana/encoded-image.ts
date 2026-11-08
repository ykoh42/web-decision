import {decode as decodePng} from 'fast-png';
import type {DecodedPng, PngDataArray} from 'fast-png';
import decodeJpeg from '@jsquash/jpeg/decode.js';
import {decodeKananaPnmImage} from './pnm-image.js';

export type KananaImageDecodeOptions = {
  signal?: AbortSignal | undefined;
  fetch?: typeof fetch | undefined;
  magickWasm?: URL | Int8Array | Uint8Array | Uint8ClampedArray | WebAssembly.Module | undefined;
};

type DecodedImage = {rgb: Uint8Array; width: number; height: number};
type ImageHeader =
  | {format: 'png'; width: number; height: number}
  | {format: 'jpeg'; width: number; height: number; components: number};
type JpegFrame = {width: number; height: number; components: number};

const PNG_SIGNATURE = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
const TIFF_SIGNATURES = [new Uint8Array([73, 73, 42, 0]), new Uint8Array([77, 77, 0, 42])];
const RGB_CHANNELS = 3;
// Pillow 12.1.1 uses int(1024 * 1024 * 1024 // 4 // 3) in Image.py.
const PILLOW_MAX_IMAGE_PIXELS = 89_478_485;
const PILLOW_ERROR_IMAGE_PIXELS = PILLOW_MAX_IMAGE_PIXELS * 2;

function isBlob(value: unknown): value is Blob {
  return typeof globalThis.Blob === 'function' && value instanceof globalThis.Blob;
}

function imageUrl(value: unknown): string | null {
  if (typeof value === 'string') return value;
  if (typeof globalThis.URL === 'function' && value instanceof globalThis.URL) return value.href;
  if (typeof globalThis.HTMLImageElement === 'function' && value instanceof globalThis.HTMLImageElement)
    return value.currentSrc || value.src;
  return null;
}

function parseImageUrl(value: string): string {
  let url;
  try {
    url = new URL(value, globalThis.location?.href);
  } catch (cause) {
    throw new TypeError('Kanana image input must be a Blob, File, URL, or image data URL.', {cause});
  }

  if (url.protocol === 'data:') {
    const comma = url.href.indexOf(',');
    const mediaType = comma < 0 ? '' : (url.href.slice(5, comma).split(';', 1)[0] ?? '');
    if (!mediaType.toLowerCase().startsWith('image/'))
      throw new TypeError('Kanana data URLs must declare an image media type.');
    return url.href;
  }

  if (!['http:', 'https:', 'blob:'].includes(url.protocol))
    throw new TypeError(`Kanana image URLs do not support the ${url.protocol} scheme.`);
  return url.href;
}

async function readImageBytes(
  source: unknown,
  options: KananaImageDecodeOptions,
  signal: AbortSignal | undefined
): Promise<Uint8Array> {
  if (isBlob(source)) return new Uint8Array(await source.arrayBuffer());

  const address = imageUrl(source);
  if (address === null) throw new TypeError('Kanana image input must be a Blob, File, URL, or image data URL.');
  const fetchImage = options.fetch ?? globalThis.fetch;
  if (typeof fetchImage !== 'function') throw new TypeError('Kanana URL images require fetch.');
  const url = parseImageUrl(address);

  let response;
  try {
    response = await fetchImage(url, signal ? {signal} : undefined);
  } catch (cause) {
    if (signal?.aborted) throw signal.reason;
    if ((cause as {name?: unknown} | null)?.name === 'AbortError') throw cause;
    throw new TypeError('Kanana image URL must allow a readable CORS response.', {cause});
  }
  signal?.throwIfAborted();
  if (response?.type === 'opaque') throw new TypeError('Kanana image URL must allow a readable CORS response.');
  if (!response?.ok)
    throw new TypeError(`Kanana image URL request failed with status ${response?.status ?? 'unknown'}.`);

  const bytes = new Uint8Array(await response.arrayBuffer());
  signal?.throwIfAborted();
  return bytes;
}

function isPng(bytes: Uint8Array): boolean {
  return PNG_SIGNATURE.every((byte, index) => bytes[index] === byte);
}

function isJpeg(bytes: Uint8Array): boolean {
  return bytes.length >= 2 && bytes[0] === 0xff && bytes[1] === 0xd8;
}

function isPnm(bytes: Uint8Array): boolean {
  return bytes.length >= 3 && bytes[0] === 0x50 && [0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x20].includes(bytes[2]!);
}

function isClassicTiff(bytes: Uint8Array): boolean {
  return TIFF_SIGNATURES.some((signature) => signature.every((byte, index) => bytes[index] === byte));
}

function assertDimensions(width: number, height: number): number {
  const pixels = width * height;
  if (!Number.isSafeInteger(width) || width < 1 || !Number.isSafeInteger(height) || height < 1)
    throw new RangeError('Kanana image dimensions must be positive safe integers.');
  if (!Number.isSafeInteger(pixels) || !Number.isSafeInteger(pixels * RGB_CHANNELS))
    throw new RangeError('Kanana image dimensions exceed the browser pixel representation.');
  return pixels;
}

function checkPillowPixelLimit(width: number, height: number): void {
  const pixels = BigInt(Math.max(1, width)) * BigInt(Math.max(1, height));
  const warningLimit = BigInt(PILLOW_MAX_IMAGE_PIXELS);
  const errorLimit = BigInt(PILLOW_ERROR_IMAGE_PIXELS);

  if (pixels > errorLimit) {
    throw new RangeError(
      `Kanana image size (${pixels} pixels) exceeds Pillow 12.1.1's decompression-bomb limit of ${errorLimit} pixels.`
    );
  }
  if (pixels > warningLimit) {
    globalThis.console?.warn?.(
      `Kanana image size (${pixels} pixels) exceeds Pillow 12.1.1's warning limit of ${warningLimit} pixels.`
    );
  }
}

function readUint32(bytes: Uint8Array, offset: number): number {
  return bytes[offset]! * 0x1000000 + (bytes[offset + 1]! << 16) + (bytes[offset + 2]! << 8) + bytes[offset + 3]!;
}

function pngDimensions(bytes: Uint8Array): {width: number; height: number} {
  const isImageHeader =
    bytes.length >= 24 &&
    readUint32(bytes, 8) === 13 &&
    bytes[12] === 73 &&
    bytes[13] === 72 &&
    bytes[14] === 68 &&
    bytes[15] === 82;
  if (!isImageHeader) throw new TypeError('Kanana PNG has no readable IHDR dimensions.');

  return {width: readUint32(bytes, 16), height: readUint32(bytes, 20)};
}

function skipJpegFillBytes(bytes: Uint8Array, offset: number): number {
  while (bytes[offset] === 0xff) offset++;
  return offset;
}

function isStandaloneJpegMarker(marker: number): boolean {
  return marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7);
}

function isJpegFrameMarker(marker: number): boolean {
  return [0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf].includes(marker);
}

function jpegSegmentLength(bytes: Uint8Array, offset: number): number | null {
  if (offset + 1 >= bytes.length) return null;
  const length = (bytes[offset]! << 8) | bytes[offset + 1]!;
  return length >= 2 && offset + length <= bytes.length ? length : null;
}

function jpegFrameHeader(bytes: Uint8Array, offset: number, length: number): JpegFrame | null {
  if (length < 8) return null;
  return {
    height: (bytes[offset + 3]! << 8) | bytes[offset + 4]!,
    width: (bytes[offset + 5]! << 8) | bytes[offset + 6]!,
    components: bytes[offset + 7]!
  };
}

function jpegFrame(bytes: Uint8Array): JpegFrame {
  let offset = 2;

  while (offset < bytes.length) {
    offset = skipJpegFillBytes(bytes, offset);
    const marker = bytes[offset++];
    if (marker === undefined || marker === 0xd9 || marker === 0xda) break;
    if (isStandaloneJpegMarker(marker)) continue;

    const length = jpegSegmentLength(bytes, offset);
    if (length === null) break;
    if (isJpegFrameMarker(marker)) {
      const frame = jpegFrameHeader(bytes, offset, length);
      if (frame !== null) return frame;
      break;
    }
    offset += length;
  }

  throw new TypeError('Kanana JPEG has no readable frame header.');
}

/** Read Pillow-compatible dimensions before a full decoder allocates pixel buffers. */
export function preflightKananaImageHeader(bytes: Uint8Array): ImageHeader | null {
  if (isPng(bytes)) {
    const {width, height} = pngDimensions(bytes);
    checkPillowPixelLimit(width, height);
    return {format: 'png', width, height};
  }
  if (isJpeg(bytes)) {
    const frame = jpegFrame(bytes);
    checkPillowPixelLimit(frame.width, frame.height);
    return {format: 'jpeg', ...frame};
  }
  return null;
}

function byteView(data: PngDataArray | Uint8ClampedArray): Uint8Array {
  return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
}

function compositeWhite(color: number, alpha: number): number {
  return Math.floor((color * alpha + 255 * (255 - alpha) + 127) / 255);
}

function readPackedSample(data: PngDataArray, width: number, x: number, y: number, depth: number): number {
  const rowBytes = Math.ceil((width * depth) / 8);
  const bit = x * depth;
  const byte = data[y * rowBytes + Math.floor(bit / 8)]!;
  const shift = 8 - depth - (bit % 8);
  return (byte >> shift) & ((1 << depth) - 1);
}

function pngSample(
  data: PngDataArray,
  width: number,
  x: number,
  y: number,
  channel: number,
  channels: number,
  depth: number
): number {
  if (depth < 8) return readPackedSample(data, width, x, y, depth);
  const index = (y * width + x) * channels + channel;
  return data[index]!;
}

function sampleToByte(sample: number, depth: number, grayscale: boolean): number {
  if (depth === 16) return grayscale ? Math.min(sample, 255) : sample >>> 8;
  if (depth < 8) return (sample * 255) / ((1 << depth) - 1);
  return sample;
}

function assertPngData(image: DecodedPng, pixels: number): void {
  const {data, depth, channels, width, height, palette} = image;
  if (![1, 2, 3, 4].includes(channels)) throw new TypeError('Kanana PNG has an unsupported channel layout.');
  if (![1, 2, 4, 8, 16].includes(depth)) throw new TypeError('Kanana PNG has an unsupported bit depth.');
  if (depth < 8 && channels !== 1) throw new TypeError('Kanana PNG has an unsupported packed channel layout.');
  if (channels === 1 && depth < 8 && !palette && ![1, 2, 4].includes(depth))
    throw new TypeError('Kanana PNG has an unsupported grayscale bit depth.');

  const expected = depth < 8 ? height * Math.ceil((width * depth) / 8) : pixels * channels;
  if (data.length !== expected) throw new TypeError('Kanana PNG decoder returned incomplete pixel data.');
  if (channels === 1 && palette && depth === 16) throw new TypeError('Kanana PNG palette depth is unsupported.');
}

function createRgbTarget(image: DecodedPng, pixels: number): Uint8Array {
  const {data, depth, channels} = image;
  const sourceBytesPerPixel = depth >= 8 ? channels * (depth / 8) : 0;
  if (sourceBytesPerPixel >= RGB_CHANNELS) {
    const bytes = byteView(data);
    return bytes.subarray(0, pixels * RGB_CHANNELS);
  }
  return new Uint8Array(pixels * RGB_CHANNELS);
}

function writePngPixel(image: DecodedPng, rgb: Uint8Array, target: number, x: number, y: number): void {
  const {channels} = image;
  if (channels === 4) return writeRgbaPixel(image, rgb, target, x, y);
  if (channels === 3) return writeRgbPixel(image, rgb, target, x, y);
  if (channels === 2) return writeGrayAlphaPixel(image, rgb, target, x, y);
  return writeGrayOrPalettePixel(image, rgb, target, x, y);
}

function writeRgbaPixel(image: DecodedPng, rgb: Uint8Array, target: number, x: number, y: number): void {
  const {data, width, depth, channels} = image;
  const red = sampleToByte(pngSample(data, width, x, y, 0, channels, depth), depth, false);
  const green = sampleToByte(pngSample(data, width, x, y, 1, channels, depth), depth, false);
  const blue = sampleToByte(pngSample(data, width, x, y, 2, channels, depth), depth, false);
  const alpha = sampleToByte(pngSample(data, width, x, y, 3, channels, depth), depth, false);
  rgb[target] = compositeWhite(red, alpha);
  rgb[target + 1] = compositeWhite(green, alpha);
  rgb[target + 2] = compositeWhite(blue, alpha);
}

function writeRgbPixel(image: DecodedPng, rgb: Uint8Array, target: number, x: number, y: number): void {
  const {data, width, depth, channels} = image;
  for (let channel = 0; channel < RGB_CHANNELS; channel++) {
    const sample = pngSample(data, width, x, y, channel, channels, depth);
    rgb[target + channel] = sampleToByte(sample, depth, false);
  }
}

function writeGrayAlphaPixel(image: DecodedPng, rgb: Uint8Array, target: number, x: number, y: number): void {
  const {data, width, depth, channels} = image;
  const sample = pngSample(data, width, x, y, 0, channels, depth);
  const gray = sampleToByte(sample, depth, false);
  rgb[target] = gray;
  rgb[target + 1] = gray;
  rgb[target + 2] = gray;
}

function writeGrayOrPalettePixel(image: DecodedPng, rgb: Uint8Array, target: number, x: number, y: number): void {
  const {data, width, depth, channels, palette} = image;
  const sample = pngSample(data, width, x, y, 0, channels, depth);
  if (palette) return writePalettePixel(palette, sample, rgb, target);

  const gray = sampleToByte(sample, depth, depth === 16);
  rgb[target] = gray;
  rgb[target + 1] = gray;
  rgb[target + 2] = gray;
}

function writePalettePixel(
  palette: NonNullable<DecodedPng['palette']>,
  index: number,
  rgb: Uint8Array,
  target: number
): void {
  const color = palette[index];
  if (!Array.isArray(color) || color.length < RGB_CHANNELS) throw new TypeError('Kanana PNG palette index is invalid.');
  rgb[target] = color[0]!;
  rgb[target + 1] = color[1]!;
  rgb[target + 2] = color[2]!;
}

function normalizePng(image: DecodedPng): DecodedImage {
  const {width, height, channels, palette} = image;
  const pixels = assertDimensions(width, height);
  assertPngData(image, pixels);
  if (channels === 1 && palette && !Array.isArray(palette)) throw new TypeError('Kanana PNG palette is malformed.');

  const rgb = createRgbTarget(image, pixels);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) writePngPixel(image, rgb, (y * width + x) * RGB_CHANNELS, x, y);
  }
  return {rgb, width, height};
}

function normalizeJpeg(imageData: ImageData): DecodedImage {
  const {width, height, data} = imageData;
  const pixels = assertDimensions(width, height);
  if (!(data instanceof Uint8Array) && !(data instanceof Uint8ClampedArray))
    throw new TypeError('Kanana JPEG decoder returned unsupported pixel data.');
  if (data.length === pixels * RGB_CHANNELS) return {rgb: byteView(data), width, height};
  if (data.length !== pixels * 4) throw new TypeError('Kanana JPEG decoder returned incomplete pixel data.');

  const bytes = byteView(data);
  const rgb = bytes.subarray(0, pixels * RGB_CHANNELS);
  for (let pixel = 0, source = 0, target = 0; pixel < pixels; pixel++, source += 4, target += 3) {
    const red = bytes[source]!;
    const green = bytes[source + 1]!;
    const blue = bytes[source + 2]!;
    rgb[target] = red;
    rgb[target + 1] = green;
    rgb[target + 2] = blue;
  }
  return {rgb, width, height};
}

/** Decode supported encoded image bytes to Pillow-compatible RGB pixels. */
export async function decodeKananaEncodedImageSource(
  source: unknown,
  options: KananaImageDecodeOptions = {}
): Promise<DecodedImage> {
  const signal = options.signal;
  signal?.throwIfAborted();
  const bytes = await readImageBytes(source, options, signal);
  signal?.throwIfAborted();

  const header = preflightKananaImageHeader(bytes);
  if (header?.format === 'png') return normalizePng(decodePng(bytes, {checkCrc: true}));
  if (header?.format === 'jpeg') {
    const {components} = header;
    if (components === 4) {
      const {decodeKananaMagickImage} = await import('./magick-image.js');
      return decodeKananaMagickImage(bytes, {signal, wasm: options.magickWasm});
    }
    if (components !== 1 && components !== 3)
      throw new TypeError(`Kanana JPEG with ${components} color components cannot match the pinned Pillow conversion.`);
    const imageData = await decodeJpeg(bytes.buffer as ArrayBuffer, {preserveOrientation: false});
    signal?.throwIfAborted();
    return normalizeJpeg(imageData);
  }

  if (isPnm(bytes)) return decodeKananaPnmImage(bytes);

  if (isClassicTiff(bytes)) {
    const {decodeKananaTiff16Image} = await import('./tiff16-image.js');
    const image = await decodeKananaTiff16Image(bytes, {signal});
    signal?.throwIfAborted();
    if (image) return image;
  }

  const {decodeKananaMagickImage} = await import('./magick-image.js');
  return decodeKananaMagickImage(bytes, {signal, wasm: options.magickWasm});
}
