import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {resizePillowBicubic} from '../../../packages/web-decision/src/kanana/image-processing.ts';
import {gemma4ImageGeometry} from './image_geometry.mts';
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
  const input = await readFile(`../e2b-v2-resize-reference/${width}x${height}.input`);
  const expected = await readFile(`../e2b-v2-resize-reference/${width}x${height}.output`);
  const target = gemma4ImageGeometry(width, height);
  const actual = resizePillowBicubic(new Uint8Array(input), width, height, target.width, target.height);
  assert.deepEqual(actual, new Uint8Array(expected), `${width}x${height}`);
  console.log(`${width}x${height} -> ${target.width}x${target.height}: byte exact`);
}
