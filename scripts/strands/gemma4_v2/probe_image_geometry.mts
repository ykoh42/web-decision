import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {gemma4ImageGeometry} from './image_geometry.mts';
const cases: Array<{input: [number, number]; output: {width: number; height: number; tokens: number}}> = JSON.parse(
  await readFile('../e2b-v2-image-geometry-reference.json', 'utf8')
);
for (const entry of cases) assert.deepEqual(gemma4ImageGeometry(...entry.input), entry.output, String(entry.input));
console.log(`${cases.length} image geometries match the official implementation`);
