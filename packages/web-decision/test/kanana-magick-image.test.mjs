import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {fileURLToPath} from 'node:url';
import {browserImage, magickImage} from './kanana-core-modules.mjs';

const {decodeKananaImageSource} = browserImage;
const {decodeKananaMagickImage} = magickImage;

const fixtureDirectory = new URL('./fixtures/kanana-image/', import.meta.url);
const wasmPath = fileURLToPath(import.meta.resolve('@imagemagick/magick-wasm/magick.wasm'));
const wasm = await readFile(wasmPath);

function encoded(base64) {
  return Buffer.from(base64, 'base64');
}

function rgbHex(image) {
  return Buffer.from(image.rgb).toString('hex');
}

function oversizedTiffHeader() {
  const bytes = encoded(
    'SUkqAAgAAAAKAAABBAABAAAABAAAAAEBBAABAAAAAwAAAAIBAwADAAAAhgAAAAMBAwABAAAAAQAAAAYBAwABAAAAAgAAABEBBAABAAAAjAAAABUBAwABAAAAAwAAABYBBAABAAAAAwAAABcBBAABAAAAJAAAABwBAwABAAAAAQAAAAAAAAAIAAgACAAeUHgeUHgeUHgeUHgeUHgeUHgeUHgeUHgeUHgeUHgeUHgeUHg='
  );
  const entryCount = bytes.readUInt16LE(8);
  for (let index = 0; index < entryCount; index++) {
    const entryOffset = 10 + index * 12;
    const tag = bytes.readUInt16LE(entryOffset);
    if (tag === 256) bytes.writeUInt32LE(178_956_971, entryOffset + 8);
    if (tag === 257) bytes.writeUInt32LE(1, entryOffset + 8);
  }
  return bytes;
}

function bigTiffRgb() {
  const ifdOffset = 16;
  const entryCount = 10;
  const entrySize = 20;
  const pixelsOffset = ifdOffset + 8 + entryCount * entrySize + 8;
  const bytes = Buffer.alloc(pixelsOffset + 3);
  bytes.write('II', 0, 'ascii');
  bytes.writeUInt16LE(43, 2);
  bytes.writeUInt16LE(8, 4);
  bytes.writeBigUInt64LE(BigInt(ifdOffset), 8);
  bytes.writeBigUInt64LE(BigInt(entryCount), ifdOffset);

  const entries = [
    [256, 4, 1, 1],
    [257, 4, 1, 1],
    [258, 3, 3, [8, 8, 8]],
    [259, 3, 1, 1],
    [262, 3, 1, 2],
    [273, 16, 1, pixelsOffset],
    [277, 3, 1, 3],
    [278, 4, 1, 1],
    [279, 16, 1, 3],
    [284, 3, 1, 1]
  ];

  for (let index = 0; index < entries.length; index++) {
    const [tag, type, count, value] = entries[index];
    const offset = ifdOffset + 8 + index * entrySize;
    bytes.writeUInt16LE(tag, offset);
    bytes.writeUInt16LE(type, offset + 2);
    bytes.writeBigUInt64LE(BigInt(count), offset + 4);
    if (Array.isArray(value)) {
      value.forEach((sample, sampleIndex) => bytes.writeUInt16LE(sample, offset + 12 + sampleIndex * 2));
    } else if (type === 3) {
      bytes.writeUInt16LE(value, offset + 12);
    } else if (type === 16) {
      bytes.writeBigUInt64LE(BigInt(value), offset + 12);
    } else {
      bytes.writeUInt32LE(value, offset + 12);
    }
  }

  bytes.writeBigUInt64LE(0n, ifdOffset + 8 + entryCount * entrySize);
  bytes.set([30, 80, 120], pixelsOffset);
  return bytes;
}

async function decode(bytes) {
  return decodeKananaMagickImage(bytes, {wasm});
}

test('CMYK JPEG matches the pinned Pillow 12.1.1 RGB golden', async () => {
  const encodedCmyk = await readFile(new URL('cmyk.jpg', fixtureDirectory));
  const expected = await readFile(new URL('cmyk.rgb', fixtureDirectory));
  const image = await decode(encodedCmyk);

  assert.equal(image.width, 16);
  assert.equal(image.height, 8);
  assert.deepEqual(Buffer.from(image.rgb), expected);
});

test('browser image entry point routes encoded CMYK JPEG to MagickWasm', async () => {
  const encodedCmyk = await readFile(new URL('cmyk.jpg', fixtureDirectory));
  const expected = await readFile(new URL('cmyk.rgb', fixtureDirectory));
  const image = await decodeKananaImageSource(new Blob([encodedCmyk]), {magickWasm: wasm});

  assert.equal(image.width, 16);
  assert.equal(image.height, 8);
  assert.deepEqual(Buffer.from(image.rgb), expected);
});

test('browser image entry point routes PNM to the pure Pillow-compatible decoder', async () => {
  const image = await decodeKananaImageSource(new Blob([Buffer.from('P6\n1 1\n255\n\x01\x02\x03', 'binary')]));

  assert.equal(image.width, 1);
  assert.equal(image.height, 1);
  assert.deepEqual(image.rgb, Uint8Array.of(1, 2, 3));
});

test('24-bit BMP truecolor matches Pillow RGB bytes', async () => {
  const image = await decode(
    encoded(
      'Qk1aAAAAAAAAADYAAAAoAAAABAAAAAMAAAABABgAAAAAACQAAADEDgAAxA4AAAAAAAAAAAAAeFAeeFAeeFAeeFAeeFAeeFAeeFAeeFAeeFAeeFAeeFAeeFAe'
    )
  );

  assert.equal(image.width, 4);
  assert.equal(image.height, 3);
  assert.equal(rgbHex(image), '1e5078'.repeat(12));
});

test('8-bit BigTIFF truecolor decodes to Pillow-compatible RGB bytes', async () => {
  const image = await decode(bigTiffRgb());

  assert.equal(image.width, 1);
  assert.equal(image.height, 1);
  assert.equal(rgbHex(image), '1e5078');
});

test('GIF palette transparency retains palette RGB without white compositing', async () => {
  const image = await decode(encoded('R0lGODlhAwABAIEAAAoUHigyPEZQWgAAACH5BAEAAAAALAAAAAADAAEAAAgGAAEEEBAQADs='));

  assert.equal(image.width, 3);
  assert.equal(image.height, 1);
  assert.equal(rgbHex(image), '0a141e28323c46505a');
});

test('animated GIF returns the first frame', async () => {
  const image = await decode(
    encoded(
      'R0lGODlhAgACAIEAAP8AAAAAAAAAAAAAACH/C05FVFNDQVBFMi4wAwEAAAAh+QQACgAAACwAAAAAAgACAAAIBgABCAQQEAAh+QQBCgABACwAAAAAAgACAIEAAP8AAAAAAAAAAAAIBgABCAQQEAA7'
    )
  );

  assert.equal(image.width, 2);
  assert.equal(image.height, 2);
  assert.equal(rgbHex(image), 'ff0000'.repeat(4));
});

test('RGB WebP bytes match Pillow RGB conversion', async () => {
  const image = await decode(
    encoded(
      'UklGRkgAAABXRUJQVlA4IDwAAADwAQCdASoEAAIAAUAmJYgCdLoAAwkVujAA/u+tjEf6nO8btu80tE6ISiDBepGGLsf5i9Xt0WFCMe3QYAA='
    )
  );

  assert.equal(image.width, 4);
  assert.equal(image.height, 2);
  assert.equal(rgbHex(image), '00001d16042a5c305fffeaff6562836554797448777b4376');
});

test('RGBA WebP composites over white with Pillow byte rounding', async () => {
  const image = await decode(
    encoded(
      'UklGRlgAAABXRUJQVlA4WAoAAAAQAAAAAwAAAgAAQUxQSAoAAAABB1DAiAhERP8DVlA4ICgAAADQAQCdASoEAAMAAUAmJZgCdLoB+AADsAD+9Sdf9T6rnr9/33oDeBgA'
    )
  );

  assert.equal(image.width, 4);
  assert.equal(image.height, 3);
  assert.equal(rgbHex(image), '8fa7ba'.repeat(12));
});

test('TIFF orientation follows Pillow and returns oriented dimensions', async () => {
  const image = await decode(
    encoded(
      'SUkqAAgAAAALAAABBAABAAAAAwAAAAEBBAABAAAAAgAAAAIBAwADAAAAkgAAAAMBAwABAAAAAQAAAAYBAwABAAAAAgAAABEBBAABAAAAmAAAABIBAwABAAAABgAAABUBAwABAAAAAwAAABYBBAABAAAAAgAAABcBBAABAAAAEgAAABwBAwABAAAAAQAAAAAAAAAIAAgACAABAgMeKDI8RlBaZG54goyWoKo='
    )
  );

  assert.equal(image.width, 2);
  assert.equal(image.height, 3);
  assert.equal(rgbHex(image), '5a646e01020378828c1e283296a0aa3c4650');
});

test('RGBA TIFF composites over white', async () => {
  const image = await decode(
    encoded(
      'SUkqAAgAAAALAAABBAABAAAAAwAAAAEBBAABAAAAAQAAAAIBAwAEAAAAkgAAAAMBAwABAAAAAQAAAAYBAwABAAAAAgAAABEBBAABAAAAmgAAABUBAwABAAAABAAAABYBBAABAAAAAQAAABcBBAABAAAADAAAABwBAwABAAAAAQAAAFIBAwABAAAAAgAAAAAAAAAIAAgACAAIAAoUHgAyPEaAZG54/w=='
    )
  );

  assert.equal(image.width, 3);
  assert.equal(image.height, 1);
  assert.equal(rgbHex(image), 'ffffff989da2646e78');
});

test('multi-page TIFF returns Pillow first-page pixels', async () => {
  const image = await decode(
    encoded(
      'SUkqAAgAAAAKAAABBAABAAAAAgAAAAEBBAABAAAAAQAAAAIBAwADAAAAhgAAAAMBAwABAAAAAQAAAAYBAwABAAAAAgAAABEBBAABAAAAjAAAABUBAwABAAAAAwAAABYBBAABAAAAAQAAABcBBAABAAAABgAAABwBAwABAAAAAQAAAKgAAAAIAAgACAD/AAD/AAAAAAAAAAAAAAAAAAAAAElJKgAIAAAACgAAAQQAAQAAAAIAAAABAQQAAQAAAAEAAAACAQMAAwAAACYBAAADAQMAAQAAAAEAAAAGAQMAAQAAAAIAAAARAQQAAQAAACwBAAAVAQMAAQAAAAMAAAAWAQQAAQAAAAEAAAAXAQQAAQAAAAYAAAAcAQMAAQAAAAEAAAAAAAAACAAIAAgAAAD/AAD/AAAAAAAAAAAAAAAAAAA='
    )
  );

  assert.equal(image.width, 2);
  assert.equal(image.height, 1);
  assert.equal(rgbHex(image), 'ff0000'.repeat(2));
});

test('PNG, PNM/PFM, RGB JPEG, and unknown bytes stay on their existing paths', async () => {
  const png = Uint8Array.from([137, 80, 78, 71, 13, 10, 26, 10]);
  const pnm = Buffer.from('P6\n1 1\n255\n\x01\x02\x03', 'binary');
  const pfm = Buffer.from('Pf\n1 1\n-1.0\n\x00\x00\x00\x00', 'binary');
  const rgbJpeg = await readFile(new URL('gray.jpg', fixtureDirectory));
  const unknown = Buffer.from('not an image');

  await assert.rejects(decode(png), /existing Pillow-compatible PNG decoder/);
  await assert.rejects(decode(pnm), /existing Pillow-compatible PNM decoder/);
  await assert.rejects(decode(pfm), /existing Pillow-compatible PNM decoder/);
  await assert.rejects(decode(rgbJpeg), /only handles CMYK input/);
  await assert.rejects(decode(unknown), /supports BMP, GIF, WebP, TIFF, and CMYK JPEG/);
});

test('Pillow decompression-bomb limits are checked from image info before pixel decoding', async () => {
  await assert.rejects(decode(oversizedTiffHeader()), {
    name: 'RangeError',
    message: /Pillow 12\.1\.1's decompression-bomb limit of 178956970 pixels/
  });
});

test('aborted decoding stops before loading MagickWasm', async () => {
  const controller = new AbortController();
  controller.abort(new Error('cancelled'));

  await assert.rejects(decodeKananaMagickImage(new Uint8Array(), {signal: controller.signal}), /cancelled/);
});
