import {preprocessKananaImage} from '../../packages/web-decision/src/kanana/image-processing.ts';

export type KananaImageProcessingReport = Readonly<{
  imageTokenThw: number[];
  pixelCount: number;
  pixelSha256: string;
  pixelType: string;
  visionGridThw: number[];
  width: number;
  height: number;
}>;

declare global {
  interface Window {
    kananaImageProcessingReport?: KananaImageProcessingReport;
  }
}

const width = 128;
const height = 32;
const rgb = new Uint8Array(width * height * 3);

for (let y = 0; y < height; y++) {
  for (let x = 0; x < width; x++) {
    const offset = (y * width + x) * 3;
    rgb[offset] = (17 * x + 3 * y) % 256;
    rgb[offset + 1] = (5 * x + 11 * y) % 256;
    rgb[offset + 2] = (13 * x + 7 * y) % 256;
  }
}

const image = preprocessKananaImage(rgb, width, height);
const pixelBytes = new Uint8Array(image.pixelValues.buffer, image.pixelValues.byteOffset, image.pixelValues.byteLength);
const digest = await crypto.subtle.digest('SHA-256', pixelBytes.slice().buffer as ArrayBuffer);
const pixelSha256 = Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, '0')).join('');

window.kananaImageProcessingReport = {
  imageTokenThw: image.imageTokenThw,
  pixelCount: image.pixelValues.length,
  pixelSha256,
  pixelType: image.pixelValues.constructor.name,
  visionGridThw: image.visionGridThw,
  width: image.width,
  height: image.height
};
