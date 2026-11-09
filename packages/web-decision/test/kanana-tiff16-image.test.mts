import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {tiff16Image} from './kanana-core-modules.mjs';

const {decodeKananaTiff16Image} = tiff16Image;

const fixtureDirectory = new URL('./fixtures/kanana-image/', import.meta.url);

async function fixture(name: string) {
  return new Uint8Array(await readFile(new URL(name, fixtureDirectory)));
}

function setClassicTag(bytes: Uint8Array, tag: number, value: number) {
  const littleEndian = bytes[0] === 73;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const directory = view.getUint32(4, littleEndian);
  const entryCount = view.getUint16(directory, littleEndian);
  for (let index = 0; index < entryCount; index++) {
    const entry = directory + 2 + index * 12;
    if (view.getUint16(entry, littleEndian) === tag) {
      const type = view.getUint16(entry + 2, littleEndian);
      if (type === 4) view.setUint32(entry + 8, value, littleEndian);
      else view.setUint16(entry + 8, value, littleEndian);
      return;
    }
  }
  throw new Error(`Missing TIFF tag ${tag}`);
}

async function assertFixture(name: string, width = 4, height = 2) {
  const encoded = await fixture(`${name}.tif`);
  const expected = await fixture(`${name}.rgb`);
  const image = await decodeKananaTiff16Image(encoded);

  assert.equal(image!.width, width);
  assert.equal(image!.height, height);
  assert.deepEqual(Buffer.from(image!.rgb), Buffer.from(expected));
}

test('16-bit grayscale conversion matches Pillow around byte boundaries', async () => {
  await assertFixture('gray-16bit-le');
  await assertFixture('gray-16bit-be-deflate');
});

test('preserves iterable byte inputs accepted by the Uint8Array constructor', async () => {
  const encoded = await fixture('gray-16bit-le.tif');
  const expected = await fixture('gray-16bit-le.rgb');
  const image = await decodeKananaTiff16Image(encoded.values());

  assert.equal(image!.width, 4);
  assert.equal(image!.height, 2);
  assert.deepEqual(Buffer.from(image!.rgb), Buffer.from(expected));
});

test('16-bit grayscale LZW decodes across TIFF code-width changes', async () => {
  await assertFixture('gray-16bit-lzw');
  await assertFixture('gray-16bit-lzw-wide', 64, 16);
});

test('16-bit chunky RGB conversion matches Pillow across endian and Deflate cases', async () => {
  for (const name of ['rgb-16bit-le', 'rgb-16bit-be-deflate', 'rgb-16bit-be-deflate-strips']) {
    await assertFixture(name);
  }
});

test('non-16-bit TIFF is declined and unsupported 16-bit layouts fail explicitly', async () => {
  const eightBit = await fixture('gray-16bit-le.tif');
  setClassicTag(eightBit, 258, 8);
  assert.equal(await decodeKananaTiff16Image(eightBit), null);

  const unsupportedCompression = await fixture('gray-16bit-le.tif');
  setClassicTag(unsupportedCompression, 259, 32773);
  await assert.rejects(decodeKananaTiff16Image(unsupportedCompression), /16-bit TIFF layout/);

  const unsupportedPhotometric = await fixture('rgb-16bit-le.tif');
  setClassicTag(unsupportedPhotometric, 262, 1);
  await assert.rejects(decodeKananaTiff16Image(unsupportedPhotometric), /16-bit TIFF layout/);
});

test('malformed TIFF offsets are declined before the decoder reads pixel strips', async () => {
  const outOfBoundsStrip = await fixture('gray-16bit-le.tif');
  setClassicTag(outOfBoundsStrip, 273, 0xffff_ffff);
  await assert.rejects(decodeKananaTiff16Image(outOfBoundsStrip), /16-bit TIFF layout/);

  const outOfBoundsDirectory = Uint8Array.of(73, 73, 42, 0, 255, 255, 255, 255);
  assert.equal(await decodeKananaTiff16Image(outOfBoundsDirectory), null);
});

test('Pillow decompression-bomb boundaries are checked before TIFF pixel allocation', async () => {
  const oversized = await fixture('gray-16bit-le.tif');
  setClassicTag(oversized, 256, 178_956_971);
  await assert.rejects(decodeKananaTiff16Image(oversized), {
    name: 'RangeError',
    message: /Pillow 12\.1\.1's decompression-bomb limit of 178956970 pixels/
  });

  const warning = await fixture('gray-16bit-le.tif');
  setClassicTag(warning, 256, 89_478_486);
  setClassicTag(warning, 257, 1);
  const warnings: unknown[] = [];
  const previousWarn = globalThis.console.warn;
  globalThis.console.warn = (message: unknown) => warnings.push(message);
  try {
    await assert.rejects(decodeKananaTiff16Image(warning), /16-bit TIFF layout/);
  } finally {
    globalThis.console.warn = previousWarn;
  }
  assert.equal(warnings.length, 1);
  assert.match(warnings[0] as string, /Pillow 12\.1\.1's warning limit of 89478485 pixels/);
});

test('compressed TIFF strips cannot expand past their declared dimensions', async () => {
  const lzw = await fixture('gray-16bit-lzw.tif');
  setClassicTag(lzw, 256, 3);
  await assert.rejects(decodeKananaTiff16Image(lzw), /exceeds its declared strip size/);

  const deflate = await fixture('rgb-16bit-be-deflate.tif');
  setClassicTag(deflate, 256, 3);
  await assert.rejects(decodeKananaTiff16Image(deflate), /exceeds its declared strip size/);
});

test('aborted TIFF decode rejects before reading strips', async () => {
  const controller = new AbortController();
  controller.abort(new Error('cancelled'));
  await assert.rejects(
    decodeKananaTiff16Image(await fixture('gray-16bit-le.tif'), {signal: controller.signal}),
    /cancelled/
  );
});
