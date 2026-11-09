import assert from 'node:assert/strict';
import test from 'node:test';
import {pnmImage} from './kanana-core-modules.mjs';

type DecodedImage = {rgb: Uint8Array; width: number; height: number};

const {decodeKananaPnmImage} = pnmImage;

function image(header: string, rasterHex: string): Buffer {
  return Buffer.concat([Buffer.from(header, 'ascii'), Buffer.from(rasterHex, 'hex')]);
}

function rgbHex(image: DecodedImage): string {
  return Buffer.from(image.rgb).toString('hex');
}

function makePfm(scale: string, values: number[], littleEndian: boolean): Buffer {
  const raster = Buffer.alloc(values.length * 4);
  values.forEach((value, index) => {
    if (littleEndian) raster.writeFloatLE(value, index * 4);
    else raster.writeFloatBE(value, index * 4);
  });
  return Buffer.concat([Buffer.from(`Pf\n2 2\n${scale}\n`, 'ascii'), raster]);
}

function pnmHeader(magic: string, width: number): Buffer {
  const sampleHeader = magic === 'P1' || magic === 'P4' ? '' : magic === 'Pf' ? '-1.0\n' : '255\n';
  return Buffer.from(`${magic}\n${width} 1\n${sampleHeader}`, 'ascii');
}

test('P6 decodes the pinned RGB PPM fixture bytes', () => {
  const decoded = decodeKananaPnmImage(image('P6\n4 3\n255\n', '1e5078'.repeat(12)));

  assert.equal(decoded.width, 4);
  assert.equal(decoded.height, 3);
  assert.equal(rgbHex(decoded), '1e5078'.repeat(12));
});

test('P5 16-bit grayscale matches Pillow RGB conversion', () => {
  const decoded = decodeKananaPnmImage(image('P5\n4 1\n65535\n', '000000010100ffff'));

  assert.equal(decoded.width, 4);
  assert.equal(decoded.height, 1);
  assert.equal(rgbHex(decoded), '000000010101ffffffffffff');
});

test('P6 16-bit color scales each big-endian channel to RGB bytes', () => {
  const decoded = decodeKananaPnmImage(image('P6\n2 1\n65535\n', '00008000ffff00000101007fff'));

  assert.equal(rgbHex(decoded), '0080ff000100');
});

test('16-bit-range P2 and P5 grayscale follow Pillow integer-mode conversion', () => {
  const plain = Buffer.from('P2\n2 1\n4095\n0 1\n', 'ascii');
  const raw = image('P5\n4 1\n4095\n', '0000000100ff0fff');

  assert.equal(rgbHex(decodeKananaPnmImage(plain)), '000000101010');
  assert.equal(rgbHex(decodeKananaPnmImage(raw)), '000000101010ffffffffffff');
});

test('Pf little-endian float samples match Pillow conversion and bottom-up row order', () => {
  const bytes = Buffer.from(
    '50660a3420320a2d312e300a0000003f0000403f0000004000000043000000000000803e0000803f00007f43',
    'hex'
  );
  const decoded = decodeKananaPnmImage(bytes);

  assert.equal(decoded.width, 4);
  assert.equal(decoded.height, 2);
  assert.equal(rgbHex(decoded), '000000000000010101ffffff000000000000020202808080');
});

test('P2 and P3 accept header comments and scale samples with ties-to-even rounding', () => {
  const grayscale = Buffer.from('P2\n# dimensions follow\n2 1\n5\n0 5\n', 'ascii');
  const color = Buffer.from('P3\n2 1\n6\n0 1 2 # pixel boundary\n3 4 6\n', 'ascii');

  assert.equal(rgbHex(decodeKananaPnmImage(grayscale)), '000000ffffff');
  assert.equal(rgbHex(decodeKananaPnmImage(color)), '002a5580aaff');
});

test('P5 handles CRLF separators and comments before raster data', () => {
  const bytes = Buffer.concat([Buffer.from('P5\r\n# sample size\r\n2 1\r\n5\r\n', 'ascii'), Buffer.from([0, 5])]);

  assert.equal(rgbHex(decodeKananaPnmImage(bytes)), '000000ffffff');
});

test('Pf uses the scale sign for byte order and ignores scale magnitude like Pillow', () => {
  const storedBottomUp = [1.5, 255.9, -2, 12.9];
  const little = decodeKananaPnmImage(makePfm('-2.0', storedBottomUp, true));
  const big = decodeKananaPnmImage(makePfm('2.0', storedBottomUp, false));
  const expected = '0000000c0c0c010101ffffff';

  assert.equal(rgbHex(little), expected);
  assert.equal(rgbHex(big), expected);
});

test('P1 and P4 bitmap samples convert zero to white and one to black', () => {
  const plain = Buffer.from('P1\n4 1\n0 1 1 0\n', 'ascii');
  const raw = Buffer.concat([Buffer.from('P4\n4 1\n', 'ascii'), Buffer.from([0b01010000])]);

  assert.equal(rgbHex(decodeKananaPnmImage(plain)), 'ffffff000000000000ffffff');
  assert.equal(rgbHex(decodeKananaPnmImage(raw)), 'ffffff000000ffffff000000');
});

test('P1 and P4 rows honor padding bits when width is not byte aligned', () => {
  const plain = Buffer.from('P1\n3 2\n0 1 0\n1 0 1\n', 'ascii');
  const raw = Buffer.concat([Buffer.from('P4\n3 2\n', 'ascii'), Buffer.from([0x40, 0xa0])]);
  const expected = 'ffffff000000ffffff000000ffffff000000';

  assert.equal(rgbHex(decodeKananaPnmImage(plain)), expected);
  assert.equal(rgbHex(decodeKananaPnmImage(raw)), expected);
});

test('P1 reads compact bits and skips whitespace and comments between samples', () => {
  const compact = Buffer.from('P1\n4 1\n0101\n', 'ascii');
  const commented = Buffer.from('P1\n4 1\n01# between samples\n01\n', 'ascii');
  const expected = 'ffffff000000ffffff000000';

  assert.equal(rgbHex(decodeKananaPnmImage(compact)), expected);
  assert.equal(rgbHex(decodeKananaPnmImage(commented)), expected);
});

test('Pillow PNM pixel limits warn at the boundary and reject above twice the limit', () => {
  const originalWarn = console.warn;
  const warnings: string[] = [];
  console.warn = (message: string) => warnings.push(message);

  try {
    assert.throws(() => decodeKananaPnmImage(pnmHeader('P6', 89_478_485)), /raster is truncated/);
    assert.equal(warnings.length, 0);

    assert.throws(() => decodeKananaPnmImage(pnmHeader('P6', 89_478_486)), /raster is truncated/);
    assert.match(warnings[0]!, /warning limit of 89478485 pixels/);

    assert.throws(() => decodeKananaPnmImage(pnmHeader('P6', 178_956_970)), /raster is truncated/);
    assert.match(warnings[1]!, /warning limit of 89478485 pixels/);

    for (const magic of ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'Pf']) {
      assert.throws(() => decodeKananaPnmImage(pnmHeader(magic, 178_956_971)), {
        name: 'RangeError',
        message: /Pillow 12\.1\.1's decompression-bomb limit of 178956970 pixels/
      });
    }

    assert.equal(warnings.length, 2);
  } finally {
    console.warn = originalWarn;
  }
});

test('unsupported formats and malformed or truncated samples are rejected', () => {
  const unsupported = Buffer.from('PF\n1 1\n-1.0\n', 'ascii');
  const truncatedPnm = image('P6\n1 1\n255\n', '0102');
  const invalidSample = Buffer.from('P3\n1 1\n5\n0 0 6\n', 'ascii');
  const invalidPbm = Buffer.from('P1\n1 1\n2\n', 'ascii');
  const invalidScale = Buffer.from('Pf\n1 1\n0\n', 'ascii');
  const truncatedPfm = Buffer.from('Pf\n1 1\n-1.0\n\x00\x00', 'binary');

  assert.throws(() => decodeKananaPnmImage(unsupported), /Unsupported/);
  assert.throws(() => decodeKananaPnmImage(truncatedPnm), /truncated/);
  assert.throws(() => decodeKananaPnmImage(invalidSample), /exceeds maxval/);
  assert.throws(() => decodeKananaPnmImage(invalidPbm), /must be 0 or 1/);
  assert.throws(() => decodeKananaPnmImage(invalidScale), /scale/);
  assert.throws(() => decodeKananaPnmImage(truncatedPfm), /truncated/);
});
