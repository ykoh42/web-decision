import assert from 'node:assert/strict';
import {test} from 'node:test';
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
