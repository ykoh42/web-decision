import assert from 'node:assert/strict';
import {test} from 'node:test';
import {encode as encodePng} from 'fast-png';
import {decodeStrandsImage} from '../src/strands/qwen35-image-input.mts';

test('the encoded image byte budget is checked before base64 decoding', async (t) => {
  const data = Uint8Array.of(20, 40, 60);
  const png = encodePng({width: 1, height: 1, channels: 3, data});
  const bytes = Buffer.from(png);
  const payload = `\u0085data:image/png;base64,${bytes.toString('base64')}\u3000`;
  const maxImagePixels = 1;
  assert.deepEqual((await decodeStrandsImage(payload, {maxImagePixels, maxImageBytes: bytes.length})).rgb, data);

  const decode = t.mock.method(globalThis, 'atob');
  await assert.rejects(decodeStrandsImage(payload, {maxImagePixels, maxImageBytes: bytes.length - 1}), /byte limit/u);
  assert.equal(decode.mock.callCount(), 0);
});

test('scanline budgeting preserves PNG pixels and the inclusive pixel limit with or without Adam7', async () => {
  for (const [width, height] of [
    [1, 3],
    [9, 11]
  ] as const) {
    const data = Uint8Array.from({length: width * height * 3}, (_, index) => (index * 17) % 256);
    for (const interlace of ['null', 'Adam7'] as const) {
      const png = encodePng({width, height, channels: 3, data}, {interlace});
      const encoded = Buffer.from(png).toString('base64');
      const decoded = await decodeStrandsImage(encoded, {maxImagePixels: width * height});
      assert.deepEqual([decoded.width, decoded.height], [width, height]);
      assert.deepEqual(decoded.rgb, data, interlace);
      await assert.rejects(decodeStrandsImage(encoded, {maxImagePixels: width * height - 1}), /at most .*pixels/u);
    }
  }
});

test('consecutive IDAT chunks decode as one PNG stream', async () => {
  // A 2x1 RGB PNG whose compressed scanline is split across two IDAT chunks.
  const png =
    'iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAIAAAB7QOjdAAAAA0lEQVR4nGOqZ4K+AAAADElEQVRgZGJmYWUDAAA/ABbgMOXrAAAAAElFTkSuQmCC';
  const decoded = await decodeStrandsImage(png, {maxImagePixels: 2});
  assert.deepEqual(decoded.rgb, Uint8Array.of(1, 2, 3, 4, 5, 6));
});
