import {resizePillowBicubic} from '../kanana/image-processing.ts';
import type {MagickWasm} from '../kanana/magick-image.js';
import {waitForAbort} from '../abort.js';
import {WebDecisionError} from '../kev.js';
import type {AssetCache} from '../assets.js';
import {cacheMapika2bImageCodecs, createMapika2bImageCodecs} from '../mapika2b/image-codecs.mts';

const PATCH_SIZE = 16;
const POOL_SIZE = 3;
const MAX_TOKENS = 280;
const PATCH_VALUES = PATCH_SIZE * PATCH_SIZE * 3;
const MAX_PATCHES = MAX_TOKENS * POOL_SIZE * POOL_SIZE;
type ImageCodecs = ReturnType<typeof createMapika2bImageCodecs>;
const defaultCodecs = createMapika2bImageCodecs({cache: null});

function bfloat16(value: number): number {
  const float = new Float32Array(1);
  const bits = new Uint32Array(float.buffer);
  float[0] = value;
  bits[0] = (bits[0]! + 0x7fff + ((bits[0]! >>> 16) & 1)) & 0xffff0000;
  return float[0]!;
}

const pixelValues = Float32Array.from({length: 256}, (_, value) => bfloat16(Math.fround(value * Math.fround(1 / 255))));

export function gemma4V2ImageSize(width: number, height: number): {width: number; height: number; tokens: number} {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 1 || height < 1)
    throw new RangeError('Image dimensions must be positive safe integers.');
  const side = PATCH_SIZE * POOL_SIZE;
  const scale = Math.sqrt((MAX_TOKENS * side * side) / (height * width));
  let outputWidth = Math.floor((scale * width) / side) * side;
  let outputHeight = Math.floor((scale * height) / side) * side;
  if (outputHeight === 0) {
    outputHeight = side;
    outputWidth = Math.min(Math.floor(width / height) * side, MAX_TOKENS * side);
  } else if (outputWidth === 0) {
    outputWidth = side;
    outputHeight = Math.min(Math.floor(height / width) * side, MAX_TOKENS * side);
  }
  return {width: outputWidth, height: outputHeight, tokens: (outputWidth / side) * (outputHeight / side)};
}

function writePatch(
  source: Uint8Array,
  width: number,
  x: number,
  y: number,
  output: Float32Array,
  patch: number
): void {
  let target = patch * PATCH_VALUES;
  for (let row = 0; row < PATCH_SIZE; row++) {
    for (let column = 0; column < PATCH_SIZE; column++) {
      const offset = ((y + row) * width + x + column) * 3;
      for (let channel = 0; channel < 3; channel++) output[target++] = pixelValues[source[offset + channel]!]!;
    }
  }
}

/** Prepare already-decoded, upright RGB pixels using the pinned Gemma 4 processor's layout. */
export function prepareGemma4V2Image(rgb: Uint8Array, width: number, height: number) {
  const size = gemma4V2ImageSize(width, height);
  const resized = resizePillowBicubic(rgb, width, height, size.width, size.height);
  const pixels = new Float32Array(MAX_PATCHES * PATCH_VALUES);
  const positions = new Int32Array(MAX_PATCHES * 2).fill(-1);
  let patch = 0;
  for (let y = 0; y < size.height; y += PATCH_SIZE) {
    for (let x = 0; x < size.width; x += PATCH_SIZE) {
      positions[patch * 2] = x / PATCH_SIZE;
      positions[patch * 2 + 1] = y / PATCH_SIZE;
      writePatch(resized, size.width, x, y, pixels, patch++);
    }
  }
  return {pixels, positions, tokens: size.tokens};
}

/** Decode an encoded Strands image and prepare its upright RGB pixels. */
export async function prepareGemma4V2EncodedImage(
  data: string,
  options: {
    magickWasm?: MagickWasm | undefined;
    signal?: AbortSignal | null;
    codecs?: ImageCodecs;
  } = {}
) {
  options.signal?.throwIfAborted();
  const {decodeStrandsImage} = await import('./qwen35-image-input.mts');
  options.signal?.throwIfAborted();
  const codecs = options.codecs ?? defaultCodecs;
  const decoded = await decodeStrandsImage(data, {
    magickWasm: options.magickWasm,
    maxImagePixels: 4096 * 4096,
    maxImageBytes: 16 * 1024 * 1024,
    invalidBase64: () => new WebDecisionError('INVALID_INPUT', 'Gemma 4 v2 image input is not valid base64.'),
    prepareJpeg: () => waitForAbort(() => codecs.prepareJpeg(), options.signal),
    loadMagick: () => waitForAbort(() => codecs.loadMagick(), options.signal)
  });
  options.signal?.throwIfAborted();
  const input = prepareGemma4V2Image(decoded.rgb, decoded.width, decoded.height);
  options.signal?.throwIfAborted();
  const header = gemma4V2ImageSize(decoded.headerWidth, decoded.headerHeight);
  if (input.tokens !== header.tokens) throw new RangeError('Image orientation changes the number of image tokens.');
  return {...input, width: decoded.width, height: decoded.height};
}

/** Reuse verified decoder assets for a model session and its offline preparation. */
export function createGemma4V2ImagePreparer(
  options: {cache?: AssetCache | null | undefined; offline?: boolean | undefined} = {}
) {
  const {cache, offline} = options;
  const codecs = createMapika2bImageCodecs({cache, offline});
  return {
    async prepareImage(data: string, signal?: AbortSignal | null) {
      const value = await prepareGemma4V2EncodedImage(data, signal === undefined ? {codecs} : {codecs, signal});
      return {tokens: value.tokens, value};
    },
    async prepareOffline(signal?: AbortSignal) {
      signal?.throwIfAborted();
      if (!cache) throw new WebDecisionError('ASSET_LOAD_FAILED', 'Image decoder cache is unavailable.');
      await cacheMapika2bImageCodecs({cache, offline, signal});
      await cacheMapika2bImageCodecs({cache, offline: true, signal});
    },
    async isOfflineReady(signal?: AbortSignal) {
      signal?.throwIfAborted();
      if (!cache) return false;
      try {
        await cacheMapika2bImageCodecs({cache, offline: true, signal});
        return true;
      } catch {
        signal?.throwIfAborted();
        return false;
      }
    }
  };
}
