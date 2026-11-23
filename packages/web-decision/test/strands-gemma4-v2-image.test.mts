import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import {gemma4V2ImageSize, prepareGemma4V2Image} from '../src/strands/gemma4-v2-image.mjs';

type Geometry = {input: [number, number]; output: {width: number; height: number; tokens: number}};
type Golden = {
  width: number;
  height: number;
  resizedWidth: number;
  resizedHeight: number;
  tokens: number;
  pixelsSha256: string;
  positionsSha256: string;
};

async function readFixture<T>(name: string): Promise<T> {
  return JSON.parse(await readFile(new URL(`./fixtures/${name}`, import.meta.url), 'utf8')) as T;
}

function sha256(values: Float32Array | Int32Array): string {
  return createHash('sha256')
    .update(new Uint8Array(values.buffer, values.byteOffset, values.byteLength))
    .digest('hex');
}

test('image dimensions match the pinned official processor', async () => {
  const cases = await readFixture<Geometry[]>('strands-gemma4-v2-geometry.json');
  for (const item of cases) assert.deepEqual(gemma4V2ImageSize(...item.input), item.output, String(item.input));
  for (const size of [0, -1, 1.5, Infinity, NaN]) assert.throws(() => gemma4V2ImageSize(size, 1), RangeError);
});

test('RGB patches and positions match pinned upstream image processing', async () => {
  const cases = await readFixture<Golden[]>('strands-gemma4-v2-image.json');
  for (const item of cases) {
    const pixels = Uint8Array.from(
      {length: item.width * item.height * 3},
      (_, i) => (i * 29 + Math.floor(i / 17) * 31 + 7) % 256
    );
    const size = gemma4V2ImageSize(item.width, item.height);
    assert.deepEqual(size, {width: item.resizedWidth, height: item.resizedHeight, tokens: item.tokens});
    const result = prepareGemma4V2Image(pixels, item.width, item.height);
    assert.equal(result.tokens, item.tokens);
    assert.equal(sha256(result.pixels), item.pixelsSha256, `${item.width}x${item.height} pixels`);
    assert.equal(sha256(result.positions), item.positionsSha256, `${item.width}x${item.height} positions`);
  }
  assert.throws(() => prepareGemma4V2Image(new Uint8Array(2), 1, 1), RangeError);
});
