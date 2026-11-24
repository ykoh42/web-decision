import {decode as decodePng} from 'fast-png';
import {unzlibSync, zlibSync} from 'fflate';
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
type GifFrame = Readonly<{
  x: number;
  y: number;
  width: number;
  height: number;
  fill: readonly [number, number, number];
  preserveTransparentRgb: boolean;
}>;
type ImageHeader = Readonly<{format: ImageFormat; width: number; height: number; components?: number; gif?: GifFrame}>;
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

function tiffOrientationValue(bytes: Uint8Array, entry: number, origin: number, little: boolean): number | undefined {
  const type = readU16(bytes, entry + 2, little);
  const count = readU32(bytes, entry + 4, little);
  const sizes: Readonly<Record<number, number>> = {
    1: 1,
    2: 1,
    3: 2,
    4: 4,
    5: 8,
    6: 1,
    7: 1,
    8: 2,
    9: 4,
    10: 8,
    11: 4,
    12: 8,
    13: 4,
    16: 8
  };
  const size = sizes[type];
  if (!size || count === 0) return undefined;
  const total = count * size;
  const position = total <= 4 ? entry + 8 : origin + readU32(bytes, entry + 8, little);
  if (position < 0 || total > bytes.length - position) return undefined;
  // Pillow exposes BYTE, ASCII and UNDEFINED as byte or string values, not numeric orientations.
  if (type === 1 || type === 2 || type === 7) return 1;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  switch (type) {
    case 3:
      return view.getUint16(position, little);
    case 4:
    case 13:
      return view.getUint32(position, little);
    case 5:
      return view.getUint32(position, little) / view.getUint32(position + 4, little);
    case 6:
      return view.getInt8(position);
    case 8:
      return view.getInt16(position, little);
    case 9:
      return view.getInt32(position, little);
    case 10:
      return view.getInt32(position, little) / view.getInt32(position + 4, little);
    case 11:
      return view.getFloat32(position, little);
    case 12:
      return view.getFloat64(position, little);
    case 16:
      return Number(view.getBigUint64(position, little));
    default:
      return undefined;
  }
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
    const value = tiffOrientationValue(bytes, entry, offset, little);
    if (value === undefined) return undefined;
    return Number.isInteger(value) && value >= 1 && value <= 8 ? value : 1;
  }
  return undefined;
}

const PNG_METADATA_LIMIT = 1024 * 1024;
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
  // Keep the amount of output buffered by any single transform bounded as well.
  const compressed = new ReadableStream<Uint8Array<ArrayBuffer>>({
    pull(controller) {
      if (start >= data.length) controller.close();
      else {
        controller.enqueue(data.slice(start, start + 1024));
        start += 1024;
      }
    }
  });
  const reader = compressed.pipeThrough(new DecompressionStream('deflate')).getReader();
  let size = 0;
  let text = '';
  for (;;) {
    const {done, value} = await reader.read();
    if (done) return text + decoder.decode();
    size += value.byteLength;
    if (size > PNG_METADATA_LIMIT) {
      await reader.cancel().catch(() => {});
      throw new RangeError('PNG text exceeds the Pillow decompression limit of 1 MiB.');
    }
    text += decoder.decode(value, {stream: true});
  }
}

function* pngChunks(bytes: Uint8Array) {
  for (let offset = 8; offset + 12 <= bytes.length;) {
    const length = readU32(bytes, offset);
    const payload = offset + 8;
    const end = payload + length + 4;
    if (end > bytes.length) throw new TypeError('PNG chunk is truncated.');
    const type = String.fromCharCode(...bytes.subarray(offset + 4, payload));
    yield {type, data: bytes.subarray(payload, payload + length)};
    if (type === 'IEND') return;
    offset = end;
  }
  throw new TypeError('PNG has no complete IEND chunk.');
}

const PNG_SINGLETON_CHUNKS = new Set(['IHDR', 'PLTE', 'tRNS', 'iCCP']);
const PNG_CHANNELS: Readonly<Partial<Record<number, number>>> = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4};
const PNG_SCANLINE_PASS = [[0, 0, 1, 1]] as const;
const PNG_ADAM7_PASSES = [
  [0, 0, 8, 8],
  [4, 0, 8, 8],
  [0, 4, 4, 8],
  [2, 0, 4, 4],
  [0, 2, 2, 4],
  [1, 0, 2, 2],
  [0, 1, 1, 2]
] as const;

function pngFilteredSize(bytes: Uint8Array, {width, height}: ImageHeader): number {
  const depth = bytes[24] ?? 0;
  const channels = PNG_CHANNELS[bytes[25] ?? -1];
  if (!channels || ![1, 2, 4, 8, 16].includes(depth) || (depth < 8 && channels !== 1))
    throw new TypeError('PNG has an unsupported channel layout or bit depth.');
  const interlace = bytes[28];
  if (interlace !== 0 && interlace !== 1) throw new TypeError('PNG has an unsupported interlace method.');
  const passes = interlace === 1 ? PNG_ADAM7_PASSES : PNG_SCANLINE_PASS;
  let size = 0;
  for (const [x, y, stepX, stepY] of passes) {
    const columns = Math.ceil((width - x) / stepX);
    const rows = Math.ceil((height - y) / stepY);
    // Each non-empty scanline is byte-aligned and starts with a filter byte.
    if (columns > 0 && rows > 0) size += rows * (1 + Math.ceil((columns * channels * depth) / 8));
  }
  return size;
}

function boundedPngInflatedSize(data: Uint8Array, limit: number, message: string): number {
  // fflate truncates output to the buffer size. The extra byte detects overflow.
  let size: number;
  try {
    size = unzlibSync(data, {out: new Uint8Array(limit + 1)}).length;
  } catch (error) {
    if (error instanceof RangeError) throw new RangeError(message, {cause: error});
    throw error;
  }
  if (size > limit) throw new RangeError(message);
  return size;
}

function validatePngChunk(type: string, data: Uint8Array, seen: Set<string>): void {
  if (PNG_SINGLETON_CHUNKS.has(type)) {
    if (seen.has(type)) throw new TypeError(`PNG has more than one ${type} chunk.`);
    seen.add(type);
  }
  if ((type === 'PLTE' && data.length > 768) || (type === 'tRNS' && data.length > 256))
    throw new TypeError('PNG palette or transparency chunk exceeds its maximum size.');
  if (type !== 'iCCP') return;
  const nameEnd = data.indexOf(0);
  if (nameEnd < 1 || nameEnd > 79 || data[nameEnd + 1] !== 0) throw new TypeError('PNG ICC profile header is invalid.');
  boundedPngInflatedSize(
    data.subarray(nameEnd + 2),
    PNG_METADATA_LIMIT,
    'PNG ICC profile exceeds the Pillow decompression limit of 1 MiB.'
  );
}

function choosePngSingletons(chunks: PngTextChunk[]) {
  const chosen = new Map<string, Uint8Array>();
  let duplicates = false;
  let imageStarted = false;
  for (const {type, data} of chunks) {
    if (type === 'IDAT') imageStarted = true;
    if (!PNG_SINGLETON_CHUNKS.has(type)) continue;
    if (imageStarted || chosen.has(type)) duplicates = true;
    if (!imageStarted || type === 'tRNS' || type === 'iCCP') {
      // Validate every encountered compressed profile, even those later replaced.
      validatePngChunk(type, data, new Set());
      if (!imageStarted || type === 'tRNS') chosen.set(type, data);
    }
  }
  return {chosen, duplicates};
}

function pngPaletteChunks(chosen: Map<string, Uint8Array>, raw: (data: Uint8Array) => Uint8Array) {
  return ['PLTE', 'tRNS'].flatMap((type) => {
    const data = chosen.get(type);
    return data ? [raw(data)] : [];
  });
}

function normalizedPngSingletons(bytes: Uint8Array, chunks: PngTextChunk[], chosen: Map<string, Uint8Array>) {
  const raw = (data: Uint8Array) => {
    const start = data.byteOffset - bytes.byteOffset - 8;
    return bytes.subarray(start, start + data.length + 12);
  };
  const header = chosen.get('IHDR');
  if (!header || header.length !== 13) throw new TypeError('PNG has no readable IHDR dimensions.');
  const parts: Uint8Array[] = [bytes.subarray(0, 8), raw(header)];
  let insertedPalette = false;
  for (const {type, data} of chunks) {
    if (type === 'IDAT' && !insertedPalette) {
      parts.push(...pngPaletteChunks(chosen, raw));
      insertedPalette = true;
    }
    if (type === 'IHDR' || type === 'PLTE' || type === 'tRNS') continue;
    if (type === 'iCCP' && chosen.get(type) !== data) continue;
    parts.push(raw(data));
  }
  const normalized = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    normalized.set(part, offset);
    offset += part.length;
  }
  return normalized;
}

function canonicalizePngSingletons(bytes: Uint8Array): Uint8Array {
  const chunks = [...pngChunks(bytes)];
  const {chosen, duplicates} = choosePngSingletons(chunks);
  return duplicates ? normalizedPngSingletons(bytes, chunks, chosen) : bytes;
}

function pngImageData(bytes: Uint8Array): Uint8Array {
  let first: Uint8Array = new Uint8Array(0);
  let size = 0;
  const seen = new Set<string>();
  for (const {type, data} of pngChunks(bytes)) {
    if (type !== 'IDAT') validatePngChunk(type, data, seen);
    else {
      if (size === 0) first = data;
      size += data.length;
    }
  }
  if (size === first.length) return first;
  const combined = new Uint8Array(size);
  let position = 0;
  for (const {type, data} of pngChunks(bytes)) {
    if (type !== 'IDAT') continue;
    combined.set(data, position);
    position += data.length;
  }
  return combined;
}

function pngCrc(data: Uint8Array): number {
  let crc = 0xffffffff;
  for (const value of data) {
    crc ^= value;
    for (let i = 0; i < 8; i++) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function pngWithImageData(bytes: Uint8Array, imageData: Uint8Array): Uint8Array {
  const compressed = zlibSync(imageData);
  const chunk = new Uint8Array(compressed.length + 12);
  const view = new DataView(chunk.buffer);
  view.setUint32(0, compressed.length);
  chunk.set([73, 68, 65, 84], 4);
  chunk.set(compressed, 8);
  view.setUint32(chunk.length - 4, pngCrc(chunk.subarray(4, chunk.length - 4)));
  const parts: Uint8Array[] = [bytes.subarray(0, 8)];
  let added = false;
  for (const {type, data} of pngChunks(bytes)) {
    if (type === 'IDAT') {
      if (!added) parts.push(chunk);
      added = true;
    } else {
      const start = data.byteOffset - bytes.byteOffset - 8;
      parts.push(bytes.subarray(start, start + data.length + 12));
    }
  }
  const result = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    result.set(part, offset);
    offset += part.length;
  }
  return result;
}

function validatePngImageData(bytes: Uint8Array, header: ImageHeader): Uint8Array {
  const expected = pngFilteredSize(bytes, header);
  const inflated = unzlibSync(pngImageData(bytes), {out: new Uint8Array(expected + 1)});
  if (inflated.length < expected) throw new TypeError('PNG decoder returned incomplete scanline data.');
  // Pillow consumes the declared scanlines and ignores any following decompressed bytes.
  // Recompress only those bytes so fast-png never expands an unbounded IDAT stream.
  return inflated.length > expected ? pngWithImageData(bytes, inflated.subarray(0, expected)) : bytes;
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
  for (const {type, data} of pngChunks(bytes)) {
    if (type === 'eXIf') exif = data;
    if (['tEXt', 'zTXt', 'iTXt'].includes(type)) {
      const keyword = [PNG_RAW_EXIF, PNG_XMP].find((name) => hasAscii(data, 0, name + '\0'));
      if (keyword) text.set(keyword, {type, data: data.subarray(keyword.length + 1)});
    }
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
  let orientation: number | undefined;
  let xmp = '';
  let frame: JpegFrame | undefined;
  while (offset < bytes.length) {
    offset = jpegMarkerOffset(bytes, offset);
    const marker = bytes[offset++];
    if (marker === undefined || marker === 0xd9 || marker === 0xda) break;
    if (jpegStandaloneMarker(marker)) continue;
    const length = readU16(bytes, offset);
    if (length < 2 || offset + length > bytes.length) throw new TypeError('JPEG segment is truncated.');
    if (marker === 0xe1 && hasAscii(bytes, offset + 2, 'Exif\0\0'))
      orientation = tiffOrientation(bytes.subarray(offset + 2, offset + length));
    else if (marker === 0xe1 && hasAscii(bytes, offset + 2, 'http://ns.adobe.com/xap/1.0/\0'))
      xmp = new TextDecoder().decode(bytes.subarray(offset + 31, offset + length));
    if (JPEG_FRAME_MARKERS.includes(marker)) frame = readJpegFrame(bytes, offset, length);
    offset += length;
  }
  if (!frame) throw new TypeError('JPEG has no supported frame header.');
  return {...frame, orientation: orientation ?? xmpOrientation(xmp)};
}

function webpHeader(bytes: Uint8Array): {width: number; height: number; orientation: number} {
  if (!hasAscii(bytes, 0, 'RIFF') || !hasAscii(bytes, 8, 'WEBP')) throw new TypeError('WebP signature is invalid.');
  const end = Math.min(bytes.length, 8 + readU32(bytes, 4, true));
  let offset = 12;
  let dimensions: {width: number; height: number} | undefined;
  let orientation: number | undefined;
  let xmp = '';
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
      orientation = tiffOrientation(bytes.subarray(payload, payload + length));
    } else if (hasAscii(bytes, offset, 'XMP ')) {
      xmp = new TextDecoder().decode(bytes.subarray(payload, payload + length));
    }
    offset = payload + length + (length & 1);
  }
  if (!dimensions) throw new TypeError('WebP has no supported image frame.');
  return {...dimensions, orientation: orientation ?? xmpOrientation(xmp)};
}

type GifPalette = Readonly<{gray: boolean; first: readonly [number, number, number]}>;

function gifPalette(bytes: Uint8Array, offset: number, entries: number): GifPalette {
  if (offset + entries * 3 > bytes.length) throw new TypeError('GIF palette is truncated.');
  let gray = true;
  for (let index = 0; index < entries; index++) {
    const pixel = offset + index * 3;
    if (bytes[pixel] !== index || bytes[pixel + 1] !== index || bytes[pixel + 2] !== index) gray = false;
  }
  return {gray, first: [bytes[offset]!, bytes[offset + 1]!, bytes[offset + 2]!]};
}

function skipGifBlocks(bytes: Uint8Array, start: number): number {
  let offset = start;
  while (offset < bytes.length) {
    const size = bytes[offset++]!;
    if (size === 0) return offset;
    if (offset + size > bytes.length) throw new TypeError('GIF extension is truncated.');
    offset += size;
  }
  throw new TypeError('GIF extension is truncated.');
}

function gifImageHeader(
  bytes: Uint8Array,
  offset: number,
  globalPalette: GifPalette | undefined,
  transparency: number | undefined
): ImageHeader & {orientation: number} {
  if (offset + 9 > bytes.length) throw new TypeError('GIF frame header is invalid.');
  const x = readU16(bytes, offset, true);
  const y = readU16(bytes, offset + 2, true);
  const frameWidth = readU16(bytes, offset + 4, true);
  const frameHeight = readU16(bytes, offset + 6, true);
  const flags = bytes[offset + 8]!;
  const localEntries = flags & 0x80 ? 2 ** ((flags & 7) + 1) : 0;
  const palette = localEntries ? gifPalette(bytes, offset + 9, localEntries) : globalPalette;
  const gray = !palette || palette.gray;
  const shade = transparency ?? 0;
  const fill: readonly [number, number, number] =
    transparency === undefined ? (palette?.first ?? [0, 0, 0]) : gray ? [shade, shade, shade] : [255, 255, 255];
  return {
    format: 'gif',
    width: Math.max(readU16(bytes, 6, true), x + frameWidth),
    height: Math.max(readU16(bytes, 8, true), y + frameHeight),
    orientation: 1,
    gif: {x, y, width: frameWidth, height: frameHeight, fill, preserveTransparentRgb: gray}
  };
}

function gifExtensionTransparency(bytes: Uint8Array, offset: number, label: number | undefined): number | undefined {
  const blockSize = bytes[offset];
  if (label !== 0xf9 || blockSize === undefined || blockSize < 4 || offset + 5 > bytes.length) return undefined;
  return bytes[offset + 1]! & 1 ? bytes[offset + 4] : undefined;
}

function gifHeader(bytes: Uint8Array): ImageHeader & {orientation: number} {
  if (bytes.length < 13) throw new TypeError('GIF header is truncated.');
  const packed = bytes[10]!;
  const globalEntries = packed & 0x80 ? 2 ** ((packed & 7) + 1) : 0;
  const globalPalette = globalEntries ? gifPalette(bytes, 13, globalEntries) : undefined;
  let offset = 13 + globalEntries * 3;
  let transparency: number | undefined;
  while (offset < bytes.length) {
    const marker = bytes[offset++]!;
    if (marker === 0x2c) return gifImageHeader(bytes, offset, globalPalette, transparency);
    if (marker !== 0x21) throw new TypeError('GIF frame header is invalid.');
    const label = bytes[offset++];
    transparency = gifExtensionTransparency(bytes, offset, label) ?? transparency;
    offset = skipGifBlocks(bytes, offset);
  }
  throw new TypeError('GIF has no image frame.');
}

function readHeader(bytes: Uint8Array): ImageHeader & {orientation?: number} {
  if (hasAscii(bytes, 0, '\x89PNG\r\n\x1a\n')) {
    if (bytes.length < 24 || readU32(bytes, 8) !== 13 || !hasAscii(bytes, 12, 'IHDR'))
      throw new TypeError('PNG has no readable IHDR dimensions.');
    return {format: 'png', width: readU32(bytes, 16), height: readU32(bytes, 20)};
  }
  if (hasAscii(bytes, 0, '\xff\xd8')) return {format: 'jpeg', ...jpegHeader(bytes)};
  if (hasAscii(bytes, 0, 'GIF87a') || hasAscii(bytes, 0, 'GIF89a')) return gifHeader(bytes);
  if (hasAscii(bytes, 0, 'RIFF') && hasAscii(bytes, 8, 'WEBP')) return {format: 'webp', ...webpHeader(bytes)};
  throw new TypeError('Qwen3.5 images must be PNG, JPEG, WEBP, or GIF.');
}

function decodeBase64(data: string, maxImageBytes?: number): Uint8Array {
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
  const byteLength = Math.floor((encoded.length * 3) / 4);
  if (maxImageBytes !== undefined && byteLength > maxImageBytes)
    throw new RangeError(`Image exceeds the ${maxImageBytes} byte limit.`);
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

async function decodeMagickRgb(
  bytes: Uint8Array,
  wasm?: MagickWasm,
  preserveTransparentRgb = false
): Promise<DecodedImage> {
  const magick = await loadMagick(wasm);
  const result = magick.ImageMagick.read(bytes, (image: IMagickImage) => {
    const width = image.width;
    const height = image.height;
    if (image.hasAlpha) {
      if (preserveTransparentRgb) image.alpha(magick.AlphaAction.Off);
      else {
        image.backgroundColor = magick.MagickColors.White;
        image.alpha(magick.AlphaAction.Remove);
      }
    }
    image.colorSpace = magick.ColorSpace.sRGB;
    const rgb = image.getPixels((pixels) => pixels.toByteArray(0, 0, width, height, 'RGB'));
    if (!(rgb instanceof Uint8Array) || rgb.length !== width * height * 3)
      throw new TypeError('ImageMagick returned incomplete RGB pixels.');
    return {width, height, rgb};
  });
  return result;
}

function gifCanvas(image: DecodedImage, header: ImageHeader): DecodedImage {
  const frame = header.gif!;
  if (image.width === header.width && image.height === header.height) return image;
  if (image.width !== frame.width || image.height !== frame.height)
    throw new TypeError('GIF decoder returned unexpected frame dimensions.');
  const rgb = new Uint8Array(header.width * header.height * 3);
  for (let i = 0; i < rgb.length; i += 3) rgb.set(frame.fill, i);
  for (let y = 0; y < image.height; y++) {
    const source = y * image.width * 3;
    const target = ((y + frame.y) * header.width + frame.x) * 3;
    rgb.set(image.rgb.subarray(source, source + image.width * 3), target);
  }
  return {width: header.width, height: header.height, rgb};
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

/** Decode the image formats accepted by Strands, composite alpha and apply EXIF orientation. */
export async function decodeStrandsImage(
  data: string,
  options: Readonly<{
    magickWasm?: MagickWasm | undefined;
    maxImagePixels: number;
    maxImageBytes?: number;
    prepareJpeg?: () => Promise<void>;
    loadMagick?: () => Promise<Uint8Array>;
  }>
): Promise<DecodedImage & {headerWidth: number; headerHeight: number}> {
  const input = decodeBase64(data, options.maxImageBytes);
  const bytes = hasAscii(input, 0, '\x89PNG\r\n\x1a\n') ? canonicalizePngSingletons(input) : input;
  const header = readHeader(bytes);
  assertImageSize(header, options.maxImagePixels);
  let decoded: DecodedImage;
  if (header.format === 'png') {
    decoded = decodePngRgb(validatePngImageData(bytes, header));
  } else if (header.format === 'jpeg' && header.components !== 4) {
    await options.prepareJpeg?.();
    const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
    decoded = decodeJpegRgb(await decodeJpeg(buffer, {preserveOrientation: false}));
  } else {
    const wasm = options.magickWasm ?? (await options.loadMagick?.());
    decoded = await decodeMagickRgb(bytes, wasm, header.gif?.preserveTransparentRgb);
  }
  if (header.gif) decoded = gifCanvas(decoded, header);

  if (decoded.width !== header.width || decoded.height !== header.height)
    throw new TypeError(
      `Image decoder returned ${decoded.width}x${decoded.height}; header reports ${header.width}x${header.height}.`
    );
  return {
    ...orientImage(decoded, header.orientation ?? (await pngOrientation(bytes))),
    headerWidth: header.width,
    headerHeight: header.height
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
  const decoded = await decodeStrandsImage(data, {magickWasm: options.magickWasm, maxImagePixels});
  return preprocessRgb(decoded, imageLongSide);
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
