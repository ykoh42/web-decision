import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import test from 'node:test';
import {imageProcessing} from './kanana-core-modules.mjs';

const {
  packKananaImagePatches,
  preprocessKananaImage,
  resizePillowBicubic,
  resizeTorchvisionBicubicAntialias,
  smartResize
} = imageProcessing;

function makeRgb(
  width: number,
  height: number,
  pattern: 'highfreq' | 'checker' | 'random',
  seed = 0x12345678
): Uint8Array {
  const pixels = new Uint8Array(width * height * 3);
  let state = seed;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const offset = (y * width + x) * 3;
      if (pattern === 'highfreq') {
        pixels[offset] = (17 * x + 3 * y) % 256;
        pixels[offset + 1] = (5 * x + 11 * y) % 256;
        pixels[offset + 2] = (13 * x + 7 * y) % 256;
      } else if (pattern === 'checker') {
        pixels[offset] = ((x + y) & 1) * 255;
        pixels[offset + 1] = ((Math.floor(x / 2) + Math.floor(y / 3)) & 1) * 255;
        pixels[offset + 2] = ((Math.floor(x / 7) + Math.floor(y / 5)) & 1) * 255;
      } else {
        for (let channel = 0; channel < 3; channel++) {
          state = (1664525 * state + 1013904223) >>> 0;
          pixels[offset + channel] = (state >>> 16) & 0xff;
        }
      }
    }
  }

  return pixels;
}

function hash(value: Uint8Array): string {
  return createHash('sha256').update(value).digest('hex');
}

test('smart resize preserves Python ties-to-even and pixel boundaries', () => {
  assert.deepEqual(smartResize(80, 1024, 4096, 1_638_400), [64, 1024]);
  assert.deepEqual(smartResize(32, 128, 4096, 1_638_400), [32, 128]);
  assert.deepEqual(smartResize(32, 96, 4096, 1_638_400), [64, 128]);
  assert.deepEqual(smartResize(640, 960, 4096, 1_638_400), [640, 960]);
  assert.deepEqual(smartResize(1024, 1600, 4096, 1_638_400), [1024, 1600]);
  assert.deepEqual(smartResize(1024, 1632, 4096, 1_638_400), [992, 1600]);
  assert.deepEqual(smartResize(32, 2048, 65_536, 16_777_216), [32, 2048]);
  assert.deepEqual(smartResize(32, 2016, 65_536, 16_777_216), [64, 2048]);
});

test('smart resize matches the pinned minimum dimension and aspect-ratio errors', () => {
  assert.deepEqual(smartResize(32, 6400, 4096, 1_638_400), [32, 6400]);
  assert.throws(() => smartResize(32, 6432, 4096, 1_638_400), {
    name: 'RangeError',
    message: 'absolute aspect ratio must be smaller than 200, got 201.0'
  });
  assert.throws(() => smartResize(31, 6400, 4096, 1_638_400), {
    name: 'RangeError',
    message: 'height:31 or width:6400 must be larger than factor:32'
  });
});

test('image processor accepts an aspect ratio of 200 and rejects larger ratios', () => {
  const boundary = preprocessKananaImage(new Uint8Array(32 * 6400 * 3), 6400, 32);

  assert.deepEqual([boundary.width, boundary.height], [6400, 32]);
  assert.throws(() => preprocessKananaImage(new Uint8Array(32 * 6432 * 3), 6432, 32), {
    name: 'RangeError',
    message: 'absolute aspect ratio must be smaller than 200, got 201.0'
  });
});

test('fixed point Pillow bicubic matches pinned preprocessing references', () => {
  const nearDown = resizePillowBicubic(makeRgb(641, 479, 'highfreq'), 641, 479, 640, 480);
  const checkerDown = resizePillowBicubic(makeRgb(257, 193, 'checker'), 257, 193, 128, 96);
  const oddUp = resizePillowBicubic(makeRgb(17, 13, 'random'), 17, 13, 32, 32);

  assert.equal(hash(nearDown), 'f2bbd5bc4594a7bde0ca6b1cb84361192f896cba08ab82b500c23635035a44e1');
  assert.equal(hash(checkerDown), '29f5d2f5a699ee69c066e4e94a818dcb5c2ee0fb18aa3e29185a8c244312cdec');
  assert.equal(hash(oddUp), 'ee834c20ecd736ccec658fc74e095bb0be2548f0c622e93cdff1a962ef5dcb6c');
});

test('antialiased uint8 bicubic upscaling matches pinned PyTorch references', () => {
  const oddUp = resizeTorchvisionBicubicAntialias(makeRgb(7, 5, 'random', 0x9e3779b9), 7, 5, 11, 9);
  const wideUp = resizeTorchvisionBicubicAntialias(makeRgb(9, 5, 'random', 0x9e3779b9), 9, 5, 15, 8);

  assert.equal(hash(oddUp), '875c324eaf384c5dbc208609e07eb46aff00d03f398af487e8577bf57719773b');
  assert.equal(hash(wideUp), '6ab7f8155d864489a1f5c0462dee2c1ebc3188363244eed327cc42fd05c7c415');
});

test('patch packing follows temporal, channel, and spatial order', () => {
  const width = 32;
  const height = 32;
  const pixels = new Uint8Array(width * height * 3);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const offset = (y * width + x) * 3;
      pixels[offset] = x;
      pixels[offset + 1] = y;
      pixels[offset + 2] = (x + y) & 0xff;
    }
  }

  const patches = packKananaImagePatches(pixels, width, height);

  assert.equal(patches.length, 4 * 3 * 2 * 16 * 16);
  assert.equal(patches[0], -1);
  assert.equal(patches[1], -0.9921568632125854);
  assert.equal(patches[256], -1);
  assert.equal(patches[512], -1);
  assert.equal(patches[1536], -0.8745098114013672);
  assert.equal(patches[3072], -1);
  assert.equal(patches[4608], -0.8745098114013672);
});

test('preprocessing returns source grid metadata and normalized patch values', () => {
  const result = preprocessKananaImage(new Uint8Array(32 * 128 * 3), 128, 32);

  assert.deepEqual(result.visionGridThw, [1, 8, 32]);
  assert.deepEqual(result.imageTokenThw, [1, 4, 16]);
  assert.equal(result.width, 512);
  assert.equal(result.height, 128);
  assert.equal(result.pixelValues.length, 256 * 3 * 2 * 16 * 16);
  assert.equal(result.pixelValues[0], -1);
  assert.equal(result.pixelValues.at(-1), -1);
});

test('full preprocessing matches pinned Python output through downscale and antialiased-upscale paths', () => {
  const downscaled = preprocessKananaImage(makeRgb(641, 479, 'highfreq'), 641, 479);
  const upscaled = preprocessKananaImage(makeRgb(128, 32, 'highfreq'), 128, 32);

  assert.deepEqual([downscaled.width, downscaled.height], [640, 480]);
  assert.deepEqual(downscaled.visionGridThw, [1, 30, 40]);
  assert.deepEqual(downscaled.imageTokenThw, [1, 15, 20]);
  assert.ok(downscaled.pixelValues instanceof Float32Array);
  assert.equal(downscaled.pixelValues.length, 1200 * 3 * 2 * 16 * 16);
  assert.equal(
    hash(
      new Uint8Array(
        downscaled.pixelValues.buffer,
        downscaled.pixelValues.byteOffset,
        downscaled.pixelValues.byteLength
      )
    ),
    '57c32ef78d83d2652d785bf75cd0d099ae252ab8ee5760e826f5b96156f030c0'
  );

  assert.deepEqual([upscaled.width, upscaled.height], [512, 128]);
  assert.deepEqual(upscaled.visionGridThw, [1, 8, 32]);
  assert.deepEqual(upscaled.imageTokenThw, [1, 4, 16]);
  assert.ok(upscaled.pixelValues instanceof Float32Array);
  assert.equal(
    upscaled.imageTokenThw.reduce((product, dimension) => product * dimension, 1),
    64
  );
  assert.equal(upscaled.pixelValues.length, 256 * 3 * 2 * 16 * 16);
  assert.equal(
    hash(new Uint8Array(upscaled.pixelValues.buffer, upscaled.pixelValues.byteOffset, upscaled.pixelValues.byteLength)),
    'c6d09a833f0da491714d71e112cfeda0bf9cc362dce502450b79eb39d936cdc1'
  );
});

test('preprocessing rejects invalid pixel buffers and patch dimensions', () => {
  assert.throws(() => preprocessKananaImage(new Uint8Array(2), 1, 1), RangeError);
  assert.throws(() => packKananaImagePatches(new Uint8Array(31 * 32 * 3), 31, 32), RangeError);
  assert.throws(() => resizeTorchvisionBicubicAntialias(new Uint8Array(3), 1, 1, 1, 0), RangeError);
});
