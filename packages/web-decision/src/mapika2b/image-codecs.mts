import {readVerifiedAsset, type AssetFileMetadata} from '../assets.js';
import type {OpenMapika2bVisionOptions} from './vision-runtime.mts';

type CodecAssetOptions = Pick<OpenMapika2bVisionOptions, 'cache' | 'offline' | 'signal'>;

// Locked browser decoder artifacts from @jsquash/jpeg 1.6.0 and @imagemagick/magick-wasm 0.0.46.
const MOZJPEG = {bytes: 166_470, sha256: 'a7c4b12169817e779ff4af137981393ae924944e167ad1bd95747c9199162d3e'};
const MAGICK = {bytes: 15_488_888, sha256: '9d4d47d8cce4317c8dd2c05a57d896d71d99af454ac5d2ab51b43058cee417ae'};

function readCodec(url: string, metadata: AssetFileMetadata, options: CodecAssetOptions): Promise<ArrayBuffer> {
  const asset = new URL(url, import.meta.url);
  const name = decodeURIComponent(asset.pathname.slice(asset.pathname.lastIndexOf('/') + 1));
  return readVerifiedAsset(new URL('.', asset), {files: {[name]: metadata}}, name, options);
}

async function readMozJpeg(options: CodecAssetOptions): Promise<ArrayBuffer> {
  const {default: url} = await import('@jsquash/jpeg/codec/dec/mozjpeg_dec.wasm?url&no-inline');
  return readCodec(url, MOZJPEG, options);
}

async function readMagick(options: CodecAssetOptions): Promise<ArrayBuffer> {
  const {default: url} = await import('@imagemagick/magick-wasm/magick.wasm?url&no-inline');
  return readCodec(url, MAGICK, options);
}

export async function cacheMapika2bImageCodecs(options: CodecAssetOptions): Promise<void> {
  await Promise.all([readMozJpeg(options), readMagick(options)]);
}

export function createMapika2bImageCodecs(options: CodecAssetOptions) {
  let jpeg: Promise<void> | undefined;
  let magick: Promise<Uint8Array> | undefined;
  return {
    prepareJpeg(): Promise<void> {
      return (jpeg ??= readMozJpeg(options)
        .then(async (wasmBinary) => {
          const {init} = await import('@jsquash/jpeg/decode.js');
          await init({wasmBinary});
        })
        .catch((cause) => {
          jpeg = undefined;
          throw cause;
        }));
    },
    loadMagick(): Promise<Uint8Array> {
      return (magick ??= readMagick(options)
        .then((bytes) => new Uint8Array(bytes))
        .catch((cause) => {
          magick = undefined;
          throw cause;
        }));
    }
  };
}
