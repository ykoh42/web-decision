import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {test} from 'node:test';
import {init as initJpeg} from '@jsquash/jpeg/decode.js';
import {prepareGemma4V2EncodedImage} from '../src/strands/gemma4-v2-image.mjs';

const require = createRequire(import.meta.url);
const directory = new URL('./fixtures/qwen35-image-input/', import.meta.url);
const digest = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');
type Golden = {
  name: string;
  inputSha256: string;
  width: number;
  height: number;
  tokens: number;
  pixelsSha256: string;
  positionsSha256: string;
};

test('encoded images, transparency and all EXIF orientations match the pinned Strands processor', async () => {
  const rows = JSON.parse(
    await readFile(new URL('./fixtures/strands-gemma4-v2-encoded-image.json', import.meta.url), 'utf8')
  ) as Golden[];
  const jpeg = await readFile(require.resolve('@jsquash/jpeg/codec/dec/mozjpeg_dec.wasm'));
  await initJpeg({wasmBinary: jpeg});
  const magickWasm = new Uint8Array(await readFile(require.resolve('@imagemagick/magick-wasm/magick.wasm')));
  for (const expected of rows) {
    const raw = await readFile(new URL(expected.name, directory));
    assert.equal(digest(raw), expected.inputSha256, `${expected.name} source`);
    const codecs = {prepareJpeg: async () => {}, loadMagick: async () => magickWasm};
    const image = await prepareGemma4V2EncodedImage(raw.toString('base64'), {magickWasm, codecs});
    assert.deepEqual(
      [image.width, image.height, image.tokens],
      [expected.width, expected.height, expected.tokens],
      expected.name
    );
    assert.equal(digest(new Uint8Array(image.pixels.buffer)), expected.pixelsSha256, `${expected.name} pixels`);
    assert.equal(
      digest(new Uint8Array(image.positions.buffer)),
      expected.positionsSha256,
      `${expected.name} positions`
    );
  }
});

test('rejects invalid encoded input and respects already-aborted requests', async () => {
  await assert.rejects(prepareGemma4V2EncodedImage('not base64!'));
  const complete = await readFile(new URL('png-orientation-1.png', directory));
  assert.equal(complete.length % 3, 0);
  await assert.rejects(prepareGemma4V2EncodedImage(complete.toString('base64') + '=A'));
  const controller = new AbortController();
  controller.abort(new Error('cancelled'));
  await assert.rejects(prepareGemma4V2EncodedImage('', {signal: controller.signal}), /cancelled/);
});

test('unknown EXIF orientation values preserve the unrotated image as Pillow does', async () => {
  const upright = new Uint8Array(3 * 4 * 3);
  for (let y = 0; y < 4; y++) {
    for (let x = 0; x < 3; x++) {
      upright.set([x * 23, y * 39, (x + y) * 19], (y * 3 + x) * 3);
    }
  }
  const {prepareGemma4V2Image} = await import('../src/strands/gemma4-v2-image.mjs');
  const expected = prepareGemma4V2Image(upright, 3, 4);
  for (const orientation of [0, 9, 65535]) {
    const raw = await readFile(
      new URL(`./fixtures/strands-gemma4-v2-encoded/orientation-${orientation}.png`, import.meta.url)
    );
    const actual = await prepareGemma4V2EncodedImage(raw.toString('base64'));
    assert.equal(actual.width, 3);
    assert.equal(actual.height, 4);
    assert.equal(digest(new Uint8Array(actual.pixels.buffer)), digest(new Uint8Array(expected.pixels.buffer)));
    assert.deepEqual(actual.positions, expected.positions);
  }
});
