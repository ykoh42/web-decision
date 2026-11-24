import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFile} from 'node:fs/promises';
import {deflateSync} from 'node:zlib';
import {preprocessQwen35Image} from '../src/strands/qwen35-image-input.mts';
import {prepareGemma4V2EncodedImage} from '../src/strands/gemma4-v2-image.mts';
function chunk(type: string, data: Buffer) {
  const bytes = Buffer.alloc(data.length + 12);
  bytes.writeUInt32BE(data.length, 0);
  bytes.write(type, 4);
  data.copy(bytes, 8);
  let crc = 0xffffffff;
  for (const b of bytes.subarray(4, -4)) {
    crc ^= b;
    for (let i = 0; i < 8; i++) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
  }
  bytes.writeUInt32BE((crc ^ 0xffffffff) >>> 0, bytes.length - 4);
  return bytes;
}
function png(extra: number, truncate = false, split = false) {
  const hdr = Buffer.alloc(13);
  hdr.writeUInt32BE(1, 0);
  hdr.writeUInt32BE(1, 4);
  hdr[8] = 8;
  hdr[9] = 2;
  const scan = truncate
    ? Buffer.from([0, 17, 34])
    : Buffer.concat([Buffer.from([0, 17, 34, 51]), Buffer.alloc(extra, 0x55)]);
  const idat = deflateSync(scan);
  const chunks = split ? [chunk('IDAT', idat.subarray(0, 2)), chunk('IDAT', idat.subarray(2))] : [chunk('IDAT', idat)];
  return Buffer.concat([
    Buffer.from('89504e470d0a1a0a', 'hex'),
    chunk('IHDR', hdr),
    ...chunks,
    chunk('IEND', Buffer.alloc(0))
  ]).toString('base64');
}
test('Qwen and Gemma ignore extra PNG scanline data while retaining declared pixels', async () => {
  const standard = png(0);
  const expectedQwen = await preprocessQwen35Image(standard);
  const expectedGemma = await prepareGemma4V2EncodedImage(standard);
  for (const [extra, split] of [
    [1, false],
    [4, true]
  ] as const) {
    const input = png(extra, false, split);
    assert.deepEqual(await preprocessQwen35Image(input), expectedQwen, `Qwen trailing ${extra}`);
    assert.deepEqual(await prepareGemma4V2EncodedImage(input), expectedGemma, `Gemma trailing ${extra}`);
  }
});
test('Qwen and Gemma reject incomplete scanline data', async () => {
  const input = png(0, true);
  await assert.rejects(preprocessQwen35Image(input), /incomplete|unexpected end/i);
  await assert.rejects(prepareGemma4V2EncodedImage(input), /incomplete|unexpected end/i);
});

test('duplicate PNG palette and transparency follow Pillow last-chunk RGB semantics', async () => {
  const {decodeStrandsImage} = await import('../src/strands/qwen35-image-input.mts');
  const {createHash} = await import('node:crypto');
  const base = new URL('./fixtures/strands-gemma4-v2-image-formats/', import.meta.url);
  for (const [file, type, late, expected] of [
    ['palette4.png', 'PLTE', false, 'fe23ebc3229e72ac682171501a62025518f61c1b07363b3deb6a663750913e00'],
    ['palette4.png', 'PLTE', true, '44b9a4e4f2534598633556d37a951b6fe6129b02fabf8672bd3a486581e7b4a2'],
    ['palette-alpha.png', 'tRNS', false, 'a27a401a95f8c4cb15960042d50126d0046bda9b735ee4c23aa3d437999d7afb'],
    ['palette-alpha.png', 'tRNS', true, 'a27a401a95f8c4cb15960042d50126d0046bda9b735ee4c23aa3d437999d7afb']
  ] as const) {
    const bytes = await readFile(new URL(file, base));
    let off = 8;
    while (bytes.toString('ascii', off + 4, off + 8) !== type) off += bytes.readUInt32BE(off) + 12;
    const len = bytes.readUInt32BE(off);
    const end = off + 12 + len;
    const copy = Buffer.from(bytes.subarray(off + 8, end - 4));
    copy[0] = copy[0]! ^ 255;
    const insert = late ? bytes.length - 12 : end;
    const image = Buffer.concat([bytes.subarray(0, insert), chunk(type, copy), bytes.subarray(insert)]).toString(
      'base64'
    );
    const decoded = await decodeStrandsImage(image, {maxImagePixels: 100});
    assert.equal(createHash('sha256').update(decoded.rgb).digest('hex'), expected);
    assert.ok((await preprocessQwen35Image(image)).pixelValues.length > 0);
    assert.ok((await prepareGemma4V2EncodedImage(image)).tokens > 0);
  }
});
test('the final PNG header supplies the dimensions for compatibility decoding', async () => {
  const {decodeStrandsImage} = await import('../src/strands/qwen35-image-input.mts');
  const bytes = Buffer.from(png(0), 'base64');
  const header = Buffer.from(bytes.subarray(16, 29));
  header.writeUInt32BE(2, 0);
  const scanline = chunk('IDAT', deflateSync(Buffer.from([0, 17, 34, 51, 68, 85, 102])));
  const image = Buffer.concat([
    bytes.subarray(0, 33),
    chunk('IHDR', header),
    scanline,
    chunk('IEND', Buffer.alloc(0))
  ]).toString('base64');
  const decoded = await decodeStrandsImage(image, {maxImagePixels: 100});
  assert.deepEqual([...decoded.rgb], [17, 34, 51, 68, 85, 102]);
});
