import {decodeKananaEncodedImageSource} from './encoded-image.js';

export type KananaImageDecodeOptions = {
  signal?: AbortSignal;
  fetch?: typeof fetch;
  magickWasm?: URL | Int8Array | Uint8Array | Uint8ClampedArray | WebAssembly.Module;
};

const DECODED_IMAGE_GLOBALS = ['ImageBitmap', 'HTMLCanvasElement', 'OffscreenCanvas'] as const;

function isDecodedImageSource(value: unknown): boolean {
  return DECODED_IMAGE_GLOBALS.some((name) => {
    const ImageSource = Reflect.get(globalThis, name) as (abstract new (...args: never[]) => object) | undefined;
    return typeof ImageSource === 'function' && value instanceof ImageSource;
  });
}

/** Decode original encoded bytes to row-major RGB while preserving Pillow source-mode semantics. */
export async function decodeKananaImageSource(source: unknown, options: KananaImageDecodeOptions = {}) {
  const signal = options.signal;
  signal?.throwIfAborted();

  if (isDecodedImageSource(source)) {
    throw new TypeError(
      'Kanana image decoding requires a Blob, File, URL, or HTMLImageElement so the original image mode and orientation can be preserved.'
    );
  }

  return decodeKananaEncodedImageSource(source, {fetch: options.fetch, signal, magickWasm: options.magickWasm});
}
