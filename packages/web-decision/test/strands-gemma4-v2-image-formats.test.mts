import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {test} from 'node:test';
import {init as initJpeg} from '@jsquash/jpeg/decode.js';
import {decodeStrandsImage} from '../src/strands/qwen35-image-input.mts';

type Row = {name: string; width: number; height: number; rgbSha256: string};
const directory = new URL('./fixtures/strands-gemma4-v2-image-formats/', import.meta.url);

test('image color modes and bit depths match the original Pillow RGB output', async () => {
  const require = createRequire(import.meta.url);
  await initJpeg({wasmBinary: await readFile(require.resolve('@jsquash/jpeg/codec/dec/mozjpeg_dec.wasm'))});
  const magickWasm = new Uint8Array(await readFile(require.resolve('@imagemagick/magick-wasm/magick.wasm')));
  const rows = JSON.parse(await readFile(new URL('reference.json', directory), 'utf8')) as Row[];
  for (const row of rows) {
    const encoded = (await readFile(new URL(row.name, directory))).toString('base64');
    const image = await decodeStrandsImage(encoded, {
      maxImagePixels: 4096 * 4096,
      magickWasm,
      prepareJpeg: async () => {},
      loadMagick: async () => magickWasm
    });
    assert.deepEqual([image.width, image.height], [row.width, row.height], row.name);
    assert.equal(createHash('sha256').update(image.rgb).digest('hex'), row.rgbSha256, row.name);
  }
});
