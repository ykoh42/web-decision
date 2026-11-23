// Resolve the same package modules used by the actual image loader before disconnecting.
import '@imagemagick/magick-wasm';
import '@jsquash/jpeg/decode.js';
export {default as jpegUrl} from '@jsquash/jpeg/codec/dec/mozjpeg_dec.wasm?url&no-inline';
export {default as magickUrl} from '@imagemagick/magick-wasm/magick.wasm?url&no-inline';
