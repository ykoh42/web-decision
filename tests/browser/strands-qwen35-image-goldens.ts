import {createMapika2bImageCodecs} from '../../packages/web-decision/src/mapika2b/image-codecs.mts';
import {preprocessQwen35Image} from '../../packages/web-decision/src/strands/qwen35-image-input.mts';

type Result = {
  decodedSize: readonly number[];
  fittedSize: readonly number[];
  processedSize: readonly number[];
  imageGridThw: readonly number[];
  pixelShape: readonly number[];
  pixelValuesSha256: string;
};
declare global {
  interface Window {
    preprocessQwen35Golden: (encoded: string) => Promise<Result>;
  }
}

const codecs = createMapika2bImageCodecs({cache: null});
window.preprocessQwen35Golden = async (encoded) => {
  await codecs.prepareJpeg();
  const magickWasm = await codecs.loadMagick();
  const image = await preprocessQwen35Image(encoded, {magickWasm});
  const bytes = new Uint8Array(image.pixelValues.length * 4);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < image.pixelValues.length; i++) view.setFloat32(i * 4, image.pixelValues[i]!, true);
  const hash = await crypto.subtle.digest('SHA-256', bytes);
  return {
    decodedSize: image.decodedSize,
    fittedSize: image.fittedSize,
    processedSize: image.processedSize,
    imageGridThw: image.imageGridThw,
    pixelShape: image.pixelShape,
    pixelValuesSha256: Array.from(new Uint8Array(hash), (value) => value.toString(16).padStart(2, '0')).join('')
  };
};
