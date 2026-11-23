import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {test} from 'node:test';
import {init as initJpeg} from '@jsquash/jpeg/decode.js';
import {prepareGemma4V2EncodedImage} from '../src/strands/gemma4-v2-image.mts';
import {decodeStrandsImage} from '../src/strands/qwen35-image-input.mts';

type Row = {
  name: string;
  width: number;
  height: number;
  tokens: number;
  rgbSha256: string;
  pixelsSha256: string;
  positionsSha256: string;
};
const directory = new URL('./fixtures/strands-gemma4-v2-image-compatibility/', import.meta.url);
const digest = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');

test('Pillow EXIF types and animated images preserve Qwen and Gemma image semantics', async () => {
  const require = createRequire(import.meta.url);
  await initJpeg({wasmBinary: await readFile(require.resolve('@jsquash/jpeg/codec/dec/mozjpeg_dec.wasm'))});
  const magickWasm = new Uint8Array(await readFile(require.resolve('@imagemagick/magick-wasm/magick.wasm')));
  const codecs = {prepareJpeg: async () => {}, loadMagick: async () => magickWasm};
  const rows = JSON.parse(await readFile(new URL('reference.json', directory), 'utf8')) as Row[];
  for (const expected of rows) {
    const encoded = (await readFile(new URL(expected.name, directory))).toString('base64');
    const decoded = await decodeStrandsImage(encoded, {maxImagePixels: 4096 * 4096, magickWasm, ...codecs});
    assert.deepEqual([decoded.width, decoded.height], [expected.width, expected.height], expected.name);
    assert.equal(digest(decoded.rgb), expected.rgbSha256, `${expected.name} RGB`);
    const image = await prepareGemma4V2EncodedImage(encoded, {magickWasm, codecs});
    assert.equal(image.tokens, expected.tokens, expected.name);
    assert.equal(digest(new Uint8Array(image.pixels.buffer)), expected.pixelsSha256, `${expected.name} pixels`);
    assert.equal(
      digest(new Uint8Array(image.positions.buffer)),
      expected.positionsSha256,
      `${expected.name} positions`
    );
  }
});
