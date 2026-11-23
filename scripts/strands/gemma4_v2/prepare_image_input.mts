import {resizePillowBicubic} from '../../../packages/web-decision/src/kanana/image-processing.ts';
import {gemma4ImageGeometry} from './image_geometry.mts';

/** Candidate Gemma 4 RGB preprocessing after image decoding and EXIF orientation. */
export function prepareGemma4ImageInput(rgb: Uint8Array, width: number, height: number) {
  const size = gemma4ImageGeometry(width, height);
  const resized = resizePillowBicubic(rgb, width, height, size.width, size.height);
  const pixels = new Float32Array(2520 * 768);
  const positions = new Int32Array(2520 * 2).fill(-1);
  const floats = new Float32Array(1);
  const words = new Uint32Array(floats.buffer);
  const bfloat = (byte: number) => {
    floats[0] = Math.fround(byte * Math.fround(1 / 255));
    words[0] = ((words[0]! + 0x7fff + ((words[0]! >>> 16) & 1)) & 0xffff0000) >>> 0;
    return floats[0]!;
  };
  let patch = 0;
  for (let y = 0; y < size.height; y += 16) {
    for (let x = 0; x < size.width; x += 16) {
      positions[patch * 2] = x / 16;
      positions[patch * 2 + 1] = y / 16;
      let offset = patch * 768;
      for (let py = 0; py < 16; py++) {
        for (let px = 0; px < 16; px++) {
          const source = ((y + py) * size.width + x + px) * 3;
          for (let channel = 0; channel < 3; channel++) pixels[offset++] = bfloat(resized[source + channel]!);
        }
      }
      patch++;
    }
  }
  return {pixels, positions, tokens: size.tokens};
}
