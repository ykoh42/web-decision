import {decode as decodePng} from 'fast-png';
import decodeJpeg from '@jsquash/jpeg/decode.js';
import type {ByteArray, IMagickImage} from '@imagemagick/magick-wasm';
import {packKananaImagePatches, resizePillowBicubic} from '../kanana/image-processing.ts';

export const QWEN35_IMAGE_DEFAULTS = Object.freeze({
  imageLongSide: 448,
  maxImagePixels: 4096 * 4096,
  maxImages: 4
});
const RESIZE_FACTOR = 32;
const MIN_IMAGE_PIXELS = 65_536;
const MAX_PROCESSOR_PIXELS = 16_777_216;
const PATCH_SIZE = 16;
const PATCH_WIDTH = 3 * 2 * PATCH_SIZE * PATCH_SIZE;
const PYTHON_WHITESPACE =
  '[\\u0009-\\u000d\\u001c-\\u0020\\u0085\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]';
const trimPythonWhitespaceStart = new RegExp(`^${PYTHON_WHITESPACE}+`, 'u');
const trimPythonWhitespaceEnd = new RegExp(`${PYTHON_WHITESPACE}+$`, 'u');

type ImageFormat = 'png' | 'jpeg' | 'gif' | 'webp';
type ImageHeader = Readonly<{format: ImageFormat; width: number; height: number; components?: number}>;
type DecodedImage = Readonly<{width: number; height: number; rgb: Uint8Array}>;
type MagickModule = typeof import('@imagemagick/magick-wasm');
type MagickWasm = URL | ByteArray | WebAssembly.Module;
type JpegFrame = Readonly<{width: number; height: number; components: number}>;
type PngImage = ReturnType<typeof decodePng>;
type PngPalette = Readonly<{colors: NonNullable<PngImage['palette']>; hasTransparency: boolean}>;

export type Qwen35ImageInputOptions = Readonly<{
  magickWasm?: MagickWasm;
  /** Strands VisionEngineConfig.image_long_side; zero disables the first resize. */
  imageLongSide?: number;
  /** Strands VisionEngineConfig.max_images. */
  maxImages?: number;
  /** Strands VisionEngineConfig.max_image_pixels. */
  maxImagePixels?: number;
}>;

export type Qwen35ImageInput = Readonly<{
  decodedSize: readonly [width: number, height: number];
  fittedSize: readonly [width: number, height: number];
  processedSize: readonly [width: number, height: number];
  imageGridThw: readonly [frames: 1, gridHeight: number, gridWidth: number];
  pixelShape: readonly [patchRows: number, patchWidth: number];
  pixelValues: Float32Array;
}>;

export type Qwen35ImagesInput = Readonly<{
  imageGridThw: readonly (readonly [frames: 1, gridHeight: number, gridWidth: number])[];
  pixelShape: readonly [patchRows: number, patchWidth: number];
  pixelValues: Float32Array;
}>;

let magickPromise: Promise<MagickModule> | undefined;

function readU16(bytes: Uint8Array, offset: number, littleEndian = false): number {
  if (offset < 0 || offset + 2 > bytes.length) throw new TypeError('Image header is truncated.');
  return littleEndian ? bytes[offset]! | (bytes[offset + 1]! << 8) : (bytes[offset]! << 8) | bytes[offset + 1]!;
}

function readU24(bytes: Uint8Array, offset: number, littleEndian = false): number {
  if (offset < 0 || offset + 3 > bytes.length) throw new TypeError('Image header is truncated.');
  return littleEndian
    ? bytes[offset]! | (bytes[offset + 1]! << 8) | (bytes[offset + 2]! << 16)
    : (bytes[offset]! << 16) | (bytes[offset + 1]! << 8) | bytes[offset + 2]!;
}

function readU32(bytes: Uint8Array, offset: number, littleEndian = false): number {
  if (offset < 0 || offset + 4 > bytes.length) throw new TypeError('Image header is truncated.');
  if (littleEndian)
    return (bytes[offset]! | (bytes[offset + 1]! << 8) | (bytes[offset + 2]! << 16) | (bytes[offset + 3]! << 24)) >>> 0;
  return bytes[offset]! * 0x1000000 + (bytes[offset + 1]! << 16) + (bytes[offset + 2]! << 8) + bytes[offset + 3]!;
}

function hasAscii(bytes: Uint8Array, offset: number, text: string): boolean {
  if (offset < 0 || offset + text.length > bytes.length) return false;
  for (let index = 0; index < text.length; index += 1)
    if (bytes[offset + index] !== text.charCodeAt(index)) return false;
  return true;
}

function tiffOrientation(bytes: Uint8Array): number | undefined {
  const offset = hasAscii(bytes, 0, 'Exif\0\0') ? 6 : 0;
  if (offset + 8 > bytes.length) throw new TypeError('Image EXIF header is truncated.');
  const little = hasAscii(bytes, offset, 'II');
  if (!little && !hasAscii(bytes, offset, 'MM')) throw new TypeError('Image EXIF byte order is invalid.');
  if (readU16(bytes, offset + 2, little) !== 42) throw new TypeError('Image EXIF signature is invalid.');
  const directory = offset + readU32(bytes, offset + 4, little);
  const entries = readU16(bytes, directory, little);
  for (let index = 0; index < entries; index += 1) {
    const entry = directory + 2 + index * 12;
    const tag = readU16(bytes, entry, little);
    if (tag !== 0x0112) continue;
    const type = readU16(bytes, entry + 2, little);
    const count = readU32(bytes, entry + 4, little);
    if (type !== 3 || count < 1) throw new TypeError('Image EXIF orientation field is invalid.');
    const value =
      count === 1
        ? readU16(bytes, entry + 8, little)
        : readU16(bytes, offset + readU32(bytes, entry + 8, little), little);
    if (value < 1 || value > 8) throw new TypeError('Image EXIF orientation must be between 1 and 8.');
    return value;
  }
  return undefined;
}

const PNG_XMP = 'XML:com.adobe.xmp';
const PNG_RAW_EXIF = 'Raw profile type exif';
type PngTextChunk = {type: string; data: Uint8Array};

async function readPngText(chunk: PngTextChunk | undefined): Promise<string> {
  if (!chunk) return '';
  const {type, data} = chunk;
  const decoder = new TextDecoder(type === 'iTXt' ? 'utf-8' : 'latin1');
  if (type === 'tEXt') return decoder.decode(data);
  let start = 1;
  if (type === 'iTXt') {
    const languageEnd = data.indexOf(0, 2);
    const keywordEnd = data.indexOf(0, languageEnd + 1);
    if (languageEnd < 0 || keywordEnd < 0) return '';
    start = keywordEnd + 1;
    if (data[0] === 0) return decoder.decode(data.subarray(start));
    if (data[1] !== 0) return '';
  }
  const compressed = new Blob([data.slice(start)]).stream();
  const reader = compressed.pipeThrough(new DecompressionStream('deflate')).getReader();
  let size = 0;
  let text = '';
  for (;;) {
    const {done, value} = await reader.read();
    if (done) return text + decoder.decode();
    size += value.byteLength;
    if (size > 1024 * 1024) {
      await reader.cancel();
      throw new RangeError('PNG text exceeds the Pillow decompression limit of 1 MiB.');
    }
    text += decoder.decode(value, {stream: true});
  }
}

function rawProfileExif(text: string): Uint8Array | undefined {
  const hex = text.split('\n').slice(3).join('');
  const pairs = hex.match(/[0-9a-f]{2}|[^\t\v\f\r ]/giu);
  if (!pairs) return undefined;
  return Uint8Array.from(pairs, (pair) => {
    if (pair.length !== 2) throw new TypeError('PNG raw EXIF profile contains invalid hexadecimal bytes.');
    return Number.parseInt(pair, 16);
  });
}

function xmpOrientation(text: string): number {
  const value = Number(/tiff:Orientation(?:="|>)([0-9])/u.exec(text)?.[1]);
  return value >= 2 && value <= 8 ? value : 1;
}

async function pngOrientation(bytes: Uint8Array): Promise<number> {
  let exif: Uint8Array | undefined;
  const text = new Map<string, PngTextChunk>();
  let offset = 8;
  while (offset + 12 <= bytes.length) {
    const length = readU32(bytes, offset);
    const payload = offset + 8;
    const next = payload + length + 4;
    if (next > bytes.length) throw new TypeError('PNG chunk is truncated.');
    const type = String.fromCharCode(...bytes.subarray(offset + 4, payload));
    const data = bytes.subarray(payload, payload + length);
    if (type === 'IEND') break;
    if (type === 'eXIf') exif = data;
    if (['tEXt', 'zTXt', 'iTXt'].includes(type)) {
      const keyword = [PNG_RAW_EXIF, PNG_XMP].find((name) => hasAscii(data, 0, name + '\0'));
      if (keyword) text.set(keyword, {type, data: data.subarray(keyword.length + 1)});
    }
    offset = next;
  }
  // Pillow uses the raw profile only when native EXIF is absent, then consults XMP for a missing orientation tag.
  exif ??= rawProfileExif(await readPngText(text.get(PNG_RAW_EXIF)));
  const orientation = exif && tiffOrientation(exif);
  return orientation ?? xmpOrientation(await readPngText(text.get(PNG_XMP)));
}

const JPEG_FRAME_MARKERS = [0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf];

function jpegMarkerOffset(bytes: Uint8Array, offset: number): number {
  if (bytes[offset] !== 0xff) throw new TypeError('JPEG marker prefix is invalid.');
  while (bytes[offset] === 0xff) offset += 1;
  return offset;
}

function jpegStandaloneMarker(marker: number): boolean {
  return marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7);
}

function readJpegFrame(bytes: Uint8Array, offset: number, length: number): JpegFrame {
  if (length < 8) throw new TypeError('JPEG frame header is truncated.');
  return {
    height: readU16(bytes, offset + 3),
    width: readU16(bytes, offset + 5),
    components: bytes[offset + 7]!
  };
}

function jpegHeader(bytes: Uint8Array): JpegFrame & {orientation: number} {
  let offset = 2;
  let orientation = 1;
  let frame: JpegFrame | undefined;
  while (offset < bytes.length) {
    offset = jpegMarkerOffset(bytes, offset);
    const marker = bytes[offset++];
    if (marker === undefined || marker === 0xd9 || marker === 0xda) break;
    if (jpegStandaloneMarker(marker)) continue;
    const length = readU16(bytes, offset);
    if (length < 2 || offset + length > bytes.length) throw new TypeError('JPEG segment is truncated.');
    if (marker === 0xe1 && hasAscii(bytes, offset + 2, 'Exif\0\0'))
      orientation = tiffOrientation(bytes.subarray(offset + 2, offset + length)) ?? 1;
    if (JPEG_FRAME_MARKERS.includes(marker)) frame = readJpegFrame(bytes, offset, length);
    offset += length;
  }
  if (!frame) throw new TypeError('JPEG has no supported frame header.');
  return {...frame, orientation};
}

function webpHeader(bytes: Uint8Array): {width: number; height: number; orientation: number} {
  if (!hasAscii(bytes, 0, 'RIFF') || !hasAscii(bytes, 8, 'WEBP')) throw new TypeError('WebP signature is invalid.');
  const end = Math.min(bytes.length, 8 + readU32(bytes, 4, true));
  let offset = 12;
  let dimensions: {width: number; height: number} | undefined;
  let orientation = 1;
  while (offset + 8 <= end) {
    const length = readU32(bytes, offset + 4, true);
    const payload = offset + 8;
    if (payload + length > end) throw new TypeError('WebP chunk is truncated.');
    if (hasAscii(bytes, offset, 'VP8X') && length >= 10) {
      dimensions = {
        width: readU24(bytes, payload + 4, true) + 1,
        height: readU24(bytes, payload + 7, true) + 1
      };
    } else if (hasAscii(bytes, offset, 'VP8 ') && length >= 10 && hasAscii(bytes, payload + 3, '\x9d\x01\x2a')) {
      dimensions = {
        width: readU16(bytes, payload + 6, true) & 0x3fff,
        height: readU16(bytes, payload + 8, true) & 0x3fff
      };
    } else if (hasAscii(bytes, offset, 'VP8L') && length >= 5 && bytes[payload] === 0x2f) {
      const bits0 = bytes[payload + 1]!;
      const bits1 = bytes[payload + 2]!;
      const bits2 = bytes[payload + 3]!;
      const bits3 = bytes[payload + 4]!;
      dimensions = {
        width: 1 + bits0 + ((bits1 & 0x3f) << 8),
        height: 1 + ((bits1 & 0xc0) >> 6) + (bits2 << 2) + ((bits3 & 0x0f) << 10)
      };
    } else if (hasAscii(bytes, offset, 'EXIF')) {
      orientation = tiffOrientation(bytes.subarray(payload, payload + length)) ?? 1;
    }
    offset = payload + length + (length & 1);
  }
  if (!dimensions) throw new TypeError('WebP has no supported image frame.');
  return {...dimensions, orientation};
}

function readHeader(bytes: Uint8Array): ImageHeader & {orientation?: number} {
  if (hasAscii(bytes, 0, '\x89PNG\r\n\x1a\n')) {
    if (bytes.length < 24 || readU32(bytes, 8) !== 13 || !hasAscii(bytes, 12, 'IHDR'))
      throw new TypeError('PNG has no readable IHDR dimensions.');
    return {format: 'png', width: readU32(bytes, 16), height: readU32(bytes, 20)};
  }
  if (hasAscii(bytes, 0, '\xff\xd8')) return {format: 'jpeg', ...jpegHeader(bytes)};
  if (hasAscii(bytes, 0, 'GIF87a') || hasAscii(bytes, 0, 'GIF89a'))
    return {format: 'gif', width: readU16(bytes, 6, true), height: readU16(bytes, 8, true), orientation: 1};
  if (hasAscii(bytes, 0, 'RIFF') && hasAscii(bytes, 8, 'WEBP')) return {format: 'webp', ...webpHeader(bytes)};
  throw new TypeError('Qwen3.5 images must be PNG, JPEG, WEBP, or GIF.');
}

function decodeBase64(data: string): Uint8Array {
  let payload = data.replace(trimPythonWhitespaceStart, '').replace(trimPythonWhitespaceEnd, '');
  const prefix = /^data:image\/[\w.+-]+;base64,/u;
  payload = payload.replace(prefix, '');
  const paddingStart = payload.indexOf('=');
  const encoded = paddingStart < 0 ? payload : payload.slice(0, paddingStart);
  const padding = paddingStart < 0 ? '' : payload.slice(paddingStart);
  const remainder = encoded.length % 4;
  const requiredPadding = remainder === 2 ? 2 : remainder === 3 ? 1 : 0;
  if (
    !/^[A-Za-z0-9+/]*$/u.test(encoded) ||
    !/^=*$/u.test(padding) ||
    (encoded.length === 0 && padding.length > 0) ||
    remainder === 1 ||
    (remainder !== 0 && padding.length !== requiredPadding)
  )
    throw new TypeError('Qwen3.5 image input is not valid base64.');
  // Python's pinned base64.b64decode(validate=True) accepts extra trailing '='
  // after a complete quartet; normalize that case before the stricter atob API.
  const decoded = globalThis.atob(encoded + '='.repeat(requiredPadding));
  const bytes = new Uint8Array(decoded.length);
  for (let index = 0; index < decoded.length; index += 1) bytes[index] = decoded.charCodeAt(index);
  return bytes;
}

function assertImageSize(header: ImageHeader, maxPixels: number): void {
  if (
    !Number.isSafeInteger(header.width) ||
    header.width < 1 ||
    !Number.isSafeInteger(header.height) ||
    header.height < 1
  )
    throw new TypeError('Qwen3.5 image dimensions must be positive safe integers.');
  if (BigInt(header.width) * BigInt(header.height) > BigInt(maxPixels))
    throw new RangeError(`Qwen3.5 images may contain at most ${maxPixels.toLocaleString('en-US')} pixels.`);
}

function resolveImageLimit(value: number | undefined, fallback: number, label: string): number {
  const limit = value ?? fallback;
  if (!Number.isSafeInteger(limit) || limit < 0)
    throw new RangeError(`Qwen3.5 ${label} must be a non-negative safe integer.`);
  return limit;
}

function sampleByte(value: number, depth: number, gray = false): number {
  if (depth === 16) return gray ? Math.min(value, 255) : value >>> 8;
  if (depth < 8) return Math.floor((value * 255) / (2 ** depth - 1));
  return value;
}

function pngSample(image: PngImage, x: number, y: number, channel: number): number {
  const {data, width, depth, channels} = image;
  if (depth < 8) {
    const rowBytes = Math.ceil((width * depth) / 8);
    const bit = x * depth;
    const byte = data[y * rowBytes + Math.floor(bit / 8)]!;
    return (byte >> (8 - depth - (bit % 8))) & (2 ** depth - 1);
  }
  return data[(y * width + x) * channels + channel]!;
}

function compositeWhite(color: number, alpha: number): number {
  return Math.floor((color * alpha + 255 * (255 - alpha) + 127) / 255);
}

function writePalettePixel(output: Uint8Array, target: number, index: number, palette: PngPalette): void {
  const color = palette.colors[index];
  if (!color || color.length < 3) throw new TypeError('PNG palette index is invalid.');
  if (!palette.hasTransparency) {
    output[target] = color[0]!;
    output[target + 1] = color[1]!;
    output[target + 2] = color[2]!;
    return;
  }
  const alpha = color[3] ?? 255;
  output[target] = compositeWhite(color[0]!, alpha);
  output[target + 1] = compositeWhite(color[1]!, alpha);
  output[target + 2] = compositeWhite(color[2]!, alpha);
}

function writePngPixel(
  image: PngImage,
  output: Uint8Array,
  target: number,
  x: number,
  y: number,
  palette: PngPalette | undefined
): void {
  const {channels, depth} = image;
  const first = pngSample(image, x, y, 0);
  switch (channels) {
    case 4: {
      const alpha = sampleByte(pngSample(image, x, y, 3), depth);
      output[target] = compositeWhite(sampleByte(first, depth), alpha);
      output[target + 1] = compositeWhite(sampleByte(pngSample(image, x, y, 1), depth), alpha);
      output[target + 2] = compositeWhite(sampleByte(pngSample(image, x, y, 2), depth), alpha);
      break;
    }
    case 2: {
      const alpha = sampleByte(pngSample(image, x, y, 1), depth);
      const gray = compositeWhite(sampleByte(first, depth), alpha);
      output[target] = gray;
      output[target + 1] = gray;
      output[target + 2] = gray;
      break;
    }
    case 1: {
      if (palette) {
        writePalettePixel(output, target, first, palette);
        break;
      }
      const gray = sampleByte(first, depth, true);
      output[target] = gray;
      output[target + 1] = gray;
      output[target + 2] = gray;
      break;
    }
    default:
      output[target] = sampleByte(first, depth);
      output[target + 1] = sampleByte(pngSample(image, x, y, 1), depth);
      output[target + 2] = sampleByte(pngSample(image, x, y, 2), depth);
  }
}

function decodePngRgb(bytes: Uint8Array): DecodedImage {
  const image = decodePng(bytes, {checkCrc: true});
  const {width, height, channels, depth, palette} = image;
  const pixels = width * height;
  if (![1, 2, 3, 4].includes(channels) || ![1, 2, 4, 8, 16].includes(depth))
    throw new TypeError('PNG has an unsupported channel layout or bit depth.');
  if (depth < 8 && channels !== 1) throw new TypeError('PNG has an unsupported packed channel layout.');
  if (image.data.length !== (depth < 8 ? height * Math.ceil((width * depth) / 8) : pixels * channels))
    throw new TypeError('PNG decoder returned incomplete pixel data.');
  const output = new Uint8Array(pixels * 3);
  const indexedPalette =
    bytes[25] === 3 && channels === 1 && palette
      ? {colors: palette, hasTransparency: palette.some((color) => (color[3] ?? 255) !== 255)}
      : undefined;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) writePngPixel(image, output, (y * width + x) * 3, x, y, indexedPalette);
  }
  return {width, height, rgb: output};
}

function decodeJpegRgb(image: ImageData): DecodedImage {
  const {width, height, data} = image;
  if (
    !Number.isSafeInteger(width * height * 3) ||
    (data.length !== width * height * 3 && data.length !== width * height * 4)
  )
    throw new TypeError('JPEG decoder returned an unsupported RGB layout.');
  if (data.length === width * height * 3) return {width, height, rgb: new Uint8Array(data)};
  const rgb = new Uint8Array(width * height * 3);
  for (let pixel = 0, source = 0, target = 0; pixel < width * height; pixel += 1, source += 4, target += 3) {
    rgb[target] = data[source]!;
    rgb[target + 1] = data[source + 1]!;
    rgb[target + 2] = data[source + 2]!;
  }
  return {width, height, rgb};
}

async function defaultMagickWasm(): Promise<URL> {
  const asset = await import('@imagemagick/magick-wasm/magick.wasm?url&no-inline');
  return new URL(asset.default, import.meta.url);
}

async function loadMagick(wasm?: MagickWasm): Promise<MagickModule> {
  if (!magickPromise) {
    magickPromise = (async () => {
      const magick = await import('@imagemagick/magick-wasm');
      await magick.initializeImageMagick(wasm ?? (await defaultMagickWasm()));
      return magick;
    })().catch((error) => {
      magickPromise = undefined;
      throw error;
    });
  }
  return magickPromise;
}

async function decodeMagickRgb(bytes: Uint8Array, wasm?: MagickWasm): Promise<DecodedImage> {
  const magick = await loadMagick(wasm);
  const result = magick.ImageMagick.read(bytes, (image: IMagickImage) => {
    const width = image.width;
    const height = image.height;
    if (image.hasAlpha) {
      image.backgroundColor = magick.MagickColors.White;
      image.alpha(magick.AlphaAction.Remove);
    }
    image.colorSpace = magick.ColorSpace.sRGB;
    const rgb = image.getPixels((pixels) => pixels.toByteArray(0, 0, width, height, 'RGB'));
    if (!(rgb instanceof Uint8Array) || rgb.length !== width * height * 3)
      throw new TypeError('ImageMagick returned incomplete RGB pixels.');
    return {width, height, rgb};
  });
  return result;
}

function orientImage(image: DecodedImage, orientation: number): DecodedImage {
  if (orientation === 1) return image;
  const swap = orientation >= 5;
  const width = swap ? image.height : image.width;
  const height = swap ? image.width : image.height;
  const rgb = new Uint8Array(image.rgb.length);
  for (let y = 0; y < image.height; y += 1) {
    for (let x = 0; x < image.width; x += 1) {
      let targetX: number;
      let targetY: number;
      if (orientation === 2) [targetX, targetY] = [image.width - 1 - x, y];
      else if (orientation === 3) [targetX, targetY] = [image.width - 1 - x, image.height - 1 - y];
      else if (orientation === 4) [targetX, targetY] = [x, image.height - 1 - y];
      else if (orientation === 5) [targetX, targetY] = [y, x];
      else if (orientation === 6) [targetX, targetY] = [image.height - 1 - y, x];
      else if (orientation === 7) [targetX, targetY] = [image.height - 1 - y, image.width - 1 - x];
      else [targetX, targetY] = [y, image.width - 1 - x];
      const source = (y * image.width + x) * 3;
      const target = (targetY * width + targetX) * 3;
      rgb[target] = image.rgb[source]!;
      rgb[target + 1] = image.rgb[source + 1]!;
      rgb[target + 2] = image.rgb[source + 2]!;
    }
  }
  return {width, height, rgb};
}

function roundPython(value: number): number {
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower % 2 === 0 ? lower : lower + 1;
}

function transposeRgb(rgb: Uint8Array, width: number, height: number): Uint8Array {
  const transposed = new Uint8Array(rgb.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const source = (y * width + x) * 3;
      const target = (x * height + y) * 3;
      transposed[target] = rgb[source]!;
      transposed[target + 1] = rgb[source + 1]!;
      transposed[target + 2] = rgb[source + 2]!;
    }
  }
  return transposed;
}

function resizeQwen35Bicubic(
  rgb: Uint8Array,
  width: number,
  height: number,
  outputWidth: number,
  outputHeight: number
): Uint8Array {
  // Match Pillow 13's separable-resize heuristic for choosing the first axis.
  const verticalFirst = height - outputHeight > 0 && height - outputHeight > (width - outputWidth) * 2;
  if (!verticalFirst) return resizePillowBicubic(rgb, width, height, outputWidth, outputHeight);

  const resizedTransposed = resizePillowBicubic(
    transposeRgb(rgb, width, height),
    height,
    width,
    outputHeight,
    outputWidth
  );
  return transposeRgb(resizedTransposed, outputHeight, outputWidth);
}

function fitImage(image: DecodedImage, longSide: number): {image: DecodedImage; size: readonly [number, number]} {
  const scale =
    longSide > 0 && Math.max(image.width, image.height) > longSide ? longSide / Math.max(image.width, image.height) : 1;
  const width = Math.max(1, roundPython(image.width * scale));
  const height = Math.max(1, roundPython(image.height * scale));
  const rgb =
    width === image.width && height === image.height
      ? image.rgb
      : resizeQwen35Bicubic(image.rgb, image.width, image.height, width, height);
  return {image: {width, height, rgb}, size: [width, height]};
}

function roundToFactor(value: number, factor: number): number {
  return roundPython(value / factor) * factor;
}

function smartResizeQwen35(height: number, width: number): readonly [number, number] {
  const aspectRatio = Math.max(height, width) / Math.min(height, width);
  if (aspectRatio > 200) throw new RangeError(`Qwen3.5 image aspect ratio must not exceed 200; got ${aspectRatio}.`);
  let resizedHeight = roundToFactor(height, RESIZE_FACTOR);
  let resizedWidth = roundToFactor(width, RESIZE_FACTOR);
  if (resizedHeight * resizedWidth > MAX_PROCESSOR_PIXELS) {
    const scale = Math.sqrt((height * width) / MAX_PROCESSOR_PIXELS);
    resizedHeight = Math.max(RESIZE_FACTOR, Math.floor(height / scale / RESIZE_FACTOR) * RESIZE_FACTOR);
    resizedWidth = Math.max(RESIZE_FACTOR, Math.floor(width / scale / RESIZE_FACTOR) * RESIZE_FACTOR);
  } else if (resizedHeight * resizedWidth < MIN_IMAGE_PIXELS) {
    const scale = Math.sqrt(MIN_IMAGE_PIXELS / (height * width));
    resizedHeight = Math.ceil((height * scale) / RESIZE_FACTOR) * RESIZE_FACTOR;
    resizedWidth = Math.ceil((width * scale) / RESIZE_FACTOR) * RESIZE_FACTOR;
  }
  return [resizedHeight, resizedWidth];
}

function preprocessRgb(image: DecodedImage, imageLongSide: number): Qwen35ImageInput {
  const fitted = fitImage(image, imageLongSide);
  const [processedHeight, processedWidth] = smartResizeQwen35(fitted.image.height, fitted.image.width);
  const processedRgb =
    processedHeight === fitted.image.height && processedWidth === fitted.image.width
      ? fitted.image.rgb
      : resizeQwen35Bicubic(fitted.image.rgb, fitted.image.width, fitted.image.height, processedWidth, processedHeight);
  const pixelValues = packKananaImagePatches(processedRgb, processedWidth, processedHeight);
  const gridHeight = processedHeight / PATCH_SIZE;
  const gridWidth = processedWidth / PATCH_SIZE;
  const patchRows = gridHeight * gridWidth;
  if (pixelValues.length !== patchRows * PATCH_WIDTH)
    throw new Error('Qwen3.5 patch layout does not match the processor contract.');
  return {
    decodedSize: [image.width, image.height],
    fittedSize: fitted.size,
    processedSize: [processedWidth, processedHeight],
    imageGridThw: [1, gridHeight, gridWidth],
    pixelShape: [patchRows, PATCH_WIDTH],
    pixelValues
  };
}

export async function preprocessQwen35Image(
  data: string,
  options: Qwen35ImageInputOptions = {}
): Promise<Qwen35ImageInput> {
  const imageLongSide = resolveImageLimit(options.imageLongSide, QWEN35_IMAGE_DEFAULTS.imageLongSide, 'imageLongSide');
  const maxImagePixels = resolveImageLimit(
    options.maxImagePixels,
    QWEN35_IMAGE_DEFAULTS.maxImagePixels,
    'maxImagePixels'
  );
  const bytes = decodeBase64(data);
  const header = readHeader(bytes);
  assertImageSize(header, maxImagePixels);
  let decoded: DecodedImage;
  if (header.format === 'png') decoded = decodePngRgb(bytes);
  else if (header.format === 'jpeg' && header.components !== 4) {
    const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
    decoded = decodeJpegRgb(await decodeJpeg(buffer, {preserveOrientation: false}));
  } else decoded = await decodeMagickRgb(bytes, options.magickWasm);

  if (decoded.width !== header.width || decoded.height !== header.height)
    throw new TypeError(
      `Image decoder returned ${decoded.width}x${decoded.height}; header reports ${header.width}x${header.height}.`
    );
  return preprocessRgb(orientImage(decoded, header.orientation ?? (await pngOrientation(bytes))), imageLongSide);
}

export async function preprocessQwen35Images(
  data: readonly string[],
  options: Qwen35ImageInputOptions = {}
): Promise<Qwen35ImagesInput> {
  const maxImages = resolveImageLimit(options.maxImages, QWEN35_IMAGE_DEFAULTS.maxImages, 'maxImages');
  if (data.length < 1) throw new RangeError('Qwen3.5 image input requires at least one image.');
  if (data.length > maxImages) throw new RangeError(`${data.length} images; this server takes at most ${maxImages}.`);

  const images: Qwen35ImageInput[] = [];
  for (const image of data) images.push(await preprocessQwen35Image(image, options));

  const patchRows = images.reduce((total, image) => total + image.pixelShape[0], 0);
  const pixelValues = new Float32Array(patchRows * PATCH_WIDTH);
  const imageGridThw = images.map((image) => image.imageGridThw);
  let offset = 0;
  for (const image of images) {
    pixelValues.set(image.pixelValues, offset);
    offset += image.pixelValues.length;
  }
  return {imageGridThw, pixelShape: [patchRows, PATCH_WIDTH], pixelValues};
}
