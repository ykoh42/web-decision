import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {test} from 'node:test';
import {init as initJpeg} from '@jsquash/jpeg/decode.js';
import {preprocessQwen35Image} from '../src/strands/qwen35-image-input.mts';

type Row = {
  name: string;
  file: string;
  status: string;
  encodedSha256: string;
  decodedSize: number[];
  fittedSize: number[];
  processedSize: number[];
  imageGridThw: number[];
  pixelShape: number[];
  pixelValuesSha256: string;
};
const dir = new URL('./fixtures/qwen35-image-input/', import.meta.url);
const sha256 = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');
function float32LittleEndian(values: Float32Array): Uint8Array {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < values.length; index++) view.setFloat32(index * 4, values[index]!, true);
  return bytes;
}

test('all pinned Qwen Pillow/Transformers image cases preserve preprocessing', async () => {
  const require = createRequire(import.meta.url);
  await initJpeg({wasmBinary: await readFile(require.resolve('@jsquash/jpeg/codec/dec/mozjpeg_dec.wasm'))});
  const magickWasm = new Uint8Array(await readFile(require.resolve('@imagemagick/magick-wasm/magick.wasm')));
  const goldens = JSON.parse(await readFile(new URL('goldens.json', dir), 'utf8')) as {cases: Row[]};
  for (const row of goldens.cases) {
    const bytes = await readFile(new URL(row.file, dir));
    assert.equal(sha256(bytes), row.encodedSha256, `${row.name} fixture`);
    if (row.status !== 'generated') {
      await assert.rejects(preprocessQwen35Image(bytes.toString('base64'), {magickWasm}), /aspect ratio/);
      continue;
    }
    const actual = await preprocessQwen35Image(bytes.toString('base64'), {magickWasm});
    assert.deepEqual(actual.decodedSize, row.decodedSize, `${row.name} decoded`);
    assert.deepEqual(actual.fittedSize, row.fittedSize, `${row.name} fitted`);
    assert.deepEqual(actual.processedSize, row.processedSize, `${row.name} processed`);
    assert.deepEqual(actual.imageGridThw, row.imageGridThw, `${row.name} grid`);
    assert.deepEqual(actual.pixelShape, row.pixelShape, `${row.name} shape`);
    assert.equal(sha256(float32LittleEndian(actual.pixelValues)), row.pixelValuesSha256, `${row.name} pixels`);
  }
});
