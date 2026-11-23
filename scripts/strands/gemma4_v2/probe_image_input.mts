import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {prepareGemma4V2Image} from '../../../packages/web-decision/src/strands/gemma4-v2-image.mts';
const sizes: Array<[number, number]> = [
  [1, 1],
  [7, 11],
  [31, 47],
  [121, 73],
  [640, 480],
  [280, 1],
  [1, 280],
  [48, 48]
];
for (const [width, height] of sizes) {
  const root = `../e2b-v2-resize-reference/${width}x${height}`;
  const input = await readFile(`${root}.input`);
  const pixels = await readFile(`${root}.pixels`);
  const positions = await readFile(`${root}.positions`);
  const actual = prepareGemma4V2Image(new Uint8Array(input), width, height);
  assert.deepEqual(new Uint8Array(actual.pixels.buffer), new Uint8Array(pixels), `${width}x${height} pixels`);
  assert.deepEqual(new Uint8Array(actual.positions.buffer), new Uint8Array(positions), `${width}x${height} positions`);
  console.log(`${width}x${height}: pixel and position data exact`);
}
