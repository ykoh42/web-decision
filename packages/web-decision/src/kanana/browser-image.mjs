import {decodeKananaEncodedImageSource} from './encoded-image.mjs';

const DECODED_IMAGE_GLOBALS = ['ImageBitmap', 'HTMLCanvasElement', 'OffscreenCanvas'];

function isDecodedImageSource(value) {
  return DECODED_IMAGE_GLOBALS.some((name) => {
    const ImageSource = globalThis[name];
    return typeof ImageSource === 'function' && value instanceof ImageSource;
  });
}

/** Decode original encoded bytes to row-major RGB while preserving Pillow source-mode semantics. */
export async function decodeKananaImageSource(source, options = {}) {
  const signal = options.signal;
  signal?.throwIfAborted();

  if (isDecodedImageSource(source)) {
    throw new TypeError(
      'Kanana image decoding requires a Blob, File, URL, or HTMLImageElement so the original image mode and orientation can be preserved.'
    );
  }

  return decodeKananaEncodedImageSource(source, {fetch: options.fetch, signal, magickWasm: options.magickWasm});
}
