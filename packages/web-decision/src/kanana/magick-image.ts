const PNG_SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10];
const JPEG_SIGNATURE = [255, 216];
const BMP_SIGNATURE = [66, 77];
const GIF_SIGNATURES = ['GIF87a', 'GIF89a'];
const TIFF_SIGNATURES = [
  [73, 73, 42, 0],
  [77, 77, 0, 42],
  [73, 73, 43, 0],
  [77, 77, 0, 43]
];
const PILLOW_MAX_IMAGE_PIXELS = 89_478_485n;
const PILLOW_ERROR_IMAGE_PIXELS = PILLOW_MAX_IMAGE_PIXELS * 2n;
const RGB_CHANNELS = 3;
const FORMAT_NAMES = 'BMP, GIF, WebP, TIFF, and CMYK JPEG';
type ImageFormat = 'bmp' | 'gif' | 'jpeg' | 'pfm' | 'png' | 'pnm' | 'tiff' | 'webp' | null;
type DecodedImageFormat = 'bmp' | 'gif' | 'jpeg' | 'tiff' | 'webp';
type MagickModule = typeof import('@imagemagick/magick-wasm');
type MagickWasm = URL | ByteArray | WebAssembly.Module;
type DecodedImage = {rgb: Uint8Array; width: number; height: number};

let runtimePromise: Promise<MagickModule> | undefined;

function bytesOf(value: unknown): Uint8Array {
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  throw new TypeError('Kanana Magick decoding requires encoded image bytes.');
}

function startsWith(bytes: Uint8Array, signature: readonly number[]): boolean {
  return signature.every((value, index) => bytes[index] === value);
}

function hasAscii(bytes: Uint8Array, offset: number, value: string): boolean {
  for (let index = 0; index < value.length; index++) {
    if (bytes[offset + index] !== value.charCodeAt(index)) return false;
  }
  return true;
}

function imageFormat(bytes: Uint8Array): ImageFormat {
  if (startsWith(bytes, PNG_SIGNATURE)) return 'png';
  if (startsWith(bytes, JPEG_SIGNATURE)) return 'jpeg';
  if (startsWith(bytes, BMP_SIGNATURE)) return 'bmp';
  if (GIF_SIGNATURES.some((signature) => hasAscii(bytes, 0, signature))) return 'gif';
  if (hasAscii(bytes, 0, 'RIFF') && hasAscii(bytes, 8, 'WEBP')) return 'webp';
  if (TIFF_SIGNATURES.some((signature) => startsWith(bytes, signature))) return 'tiff';
  if (hasAscii(bytes, 0, 'Pf') || hasAscii(bytes, 0, 'PF')) return 'pfm';
  if (bytes[0] === 80 && bytes[1]! >= 49 && bytes[1]! <= 54) return 'pnm';
  return null;
}

function assertFormatSupported(format: ImageFormat): asserts format is DecodedImageFormat {
  if (format === 'png') {
    throw new TypeError('Kanana PNG input must use the existing Pillow-compatible PNG decoder.');
  }
  if (format === 'pfm' || format === 'pnm') {
    throw new TypeError('Kanana PNM/PFM input must use the existing Pillow-compatible PNM decoder.');
  }
  if (format === null) {
    throw new TypeError(`Kanana Magick decoding supports ${FORMAT_NAMES}.`);
  }
}

async function defaultWasmUrl(): Promise<URL> {
  const asset = await import('@imagemagick/magick-wasm/magick.wasm?url');
  return new URL(asset.default, import.meta.url);
}

async function magickRuntime(wasm: MagickWasm | undefined): Promise<MagickModule> {
  if (!runtimePromise) {
    runtimePromise = (async () => {
      const magick = await import('@imagemagick/magick-wasm');
      await magick.initializeImageMagick(wasm ?? (await defaultWasmUrl()));
      return magick;
    })().catch((error) => {
      runtimePromise = undefined;
      throw error;
    });
  }
  return runtimePromise;
}

function checkPillowPixelLimit(width: number, height: number): void {
  const pixels = BigInt(width) * BigInt(height);
  if (pixels > PILLOW_ERROR_IMAGE_PIXELS) {
    throw new RangeError(
      `Kanana image size (${pixels} pixels) exceeds Pillow 12.1.1's decompression-bomb limit of ${PILLOW_ERROR_IMAGE_PIXELS} pixels.`
    );
  }
  if (pixels > PILLOW_MAX_IMAGE_PIXELS) {
    globalThis.console?.warn?.(
      `Kanana image size (${pixels} pixels) exceeds Pillow 12.1.1's warning limit of ${PILLOW_MAX_IMAGE_PIXELS} pixels.`
    );
  }
  if (!Number.isSafeInteger(width * height * RGB_CHANNELS)) {
    throw new RangeError('Kanana image dimensions exceed the browser pixel representation.');
  }
}

function assertSupportedMode(image: IMagickImage, format: DecodedImageFormat, magick: MagickModule): void {
  if (image.depth > 8) {
    throw new TypeError('Kanana Magick decoding only preserves tested 8-bit Pillow image modes.');
  }
  const type = image.colorType;
  const supported =
    (format === 'jpeg' && type === magick.ColorType.ColorSeparation && image.colorSpace === magick.ColorSpace.CMYK) ||
    (format === 'bmp' && type === magick.ColorType.TrueColor && !image.hasAlpha) ||
    (format === 'gif' && (type === magick.ColorType.Palette || type === magick.ColorType.PaletteAlpha)) ||
    (format === 'webp' && (type === magick.ColorType.TrueColor || type === magick.ColorType.TrueColorAlpha)) ||
    (format === 'tiff' && (type === magick.ColorType.TrueColor || type === magick.ColorType.TrueColorAlpha));
  if (!supported) {
    throw new TypeError(`Kanana ${format.toUpperCase()} source mode is not covered by the Pillow parity fixtures.`);
  }
}

function rgbPixels(image: IMagickImage, format: DecodedImageFormat, magick: MagickModule): DecodedImage {
  if (format === 'tiff') image.autoOrient();

  if (image.hasAlpha && image.colorType === magick.ColorType.TrueColorAlpha) {
    image.backgroundColor = magick.MagickColors.White;
    image.alpha(magick.AlphaAction.Remove);
  }

  image.colorSpace = magick.ColorSpace.sRGB;
  const {width, height} = image;
  const rgb = image.getPixels((pixels) => pixels.toByteArray(0, 0, width, height, 'RGB'));
  if (!(rgb instanceof Uint8Array) || rgb.length !== width * height * RGB_CHANNELS) {
    throw new TypeError('Kanana Magick decoder returned incomplete RGB pixels.');
  }
  return {rgb, width, height};
}

/** Decode supported encoded bytes while retaining tested Pillow 12.1.1 RGB semantics. */
export async function decodeKananaMagickImage(
  source: unknown,
  options: {signal?: AbortSignal | undefined; wasm?: MagickWasm | undefined} = {}
): Promise<DecodedImage> {
  const signal = options.signal;
  signal?.throwIfAborted();

  const bytes = bytesOf(source);
  const format = imageFormat(bytes);
  assertFormatSupported(format);

  const magick = await magickRuntime(options.wasm);
  signal?.throwIfAborted();

  const info = magick.MagickImageInfo.create(bytes);
  if (!Number.isSafeInteger(info.width) || info.width < 1 || !Number.isSafeInteger(info.height) || info.height < 1) {
    throw new RangeError('Kanana image dimensions must be positive safe integers.');
  }
  checkPillowPixelLimit(info.width, info.height);

  if (format === 'jpeg' && info.colorSpace !== magick.ColorSpace.CMYK) {
    throw new TypeError('Kanana Magick JPEG fallback only handles CMYK input.');
  }

  const result = magick.ImageMagick.read(bytes, (image) => {
    assertSupportedMode(image, format, magick);
    return rgbPixels(image, format, magick);
  });
  signal?.throwIfAborted();
  return result;
}
import type {ByteArray, IMagickImage} from '@imagemagick/magick-wasm';
