import {createMapika2bImageCodecs} from '../../packages/web-decision/src/mapika2b/image-codecs.mts';
import {prepareGemma4V2EncodedImage} from '../../packages/web-decision/src/strands/gemma4-v2-image.mts';
import {decodeStrandsImage} from '../../packages/web-decision/src/strands/qwen35-image-input.mts';

type Result = {
  width: number;
  height: number;
  tokens: number;
  rgbSha256: string;
  pixelsSha256: string;
  positionsSha256: string;
};
declare global {
  interface Window {
    imageCompatibility: (encoded: string) => Promise<Result>;
  }
}
const codecs = createMapika2bImageCodecs({cache: null});
async function hash(bytes: Uint8Array): Promise<string> {
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new Uint8Array(bytes))), (value) =>
    value.toString(16).padStart(2, '0')
  ).join('');
}

window.imageCompatibility = async (encoded) => {
  const decoded = await decodeStrandsImage(encoded, {
    maxImagePixels: 4096 * 4096,
    prepareJpeg: codecs.prepareJpeg,
    loadMagick: codecs.loadMagick
  });
  const image = await prepareGemma4V2EncodedImage(encoded, {codecs});
  return {
    width: decoded.width,
    height: decoded.height,
    tokens: image.tokens,
    rgbSha256: await hash(decoded.rgb),
    pixelsSha256: await hash(new Uint8Array(image.pixels.buffer)),
    positionsSha256: await hash(new Uint8Array(image.positions.buffer))
  };
};
