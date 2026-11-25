import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import {encode as encodePng} from 'fast-png';
import {prepareGemma4V2EncodedImage, prepareGemma4V2Image} from '../src/strands/gemma4-v2-image.mjs';
import {decodeStrandsImage, preprocessQwen35Image} from '../src/strands/qwen35-image-input.mjs';

interface Fixture {
  name: string;
  width: number;
  height: number;
  base64: string;
  pillowRgbSha256: string;
}

const fixturesUrl = new URL('./fixtures/strands-image-packed-adam7.json', import.meta.url);

test('Strands preserves pinned Pillow RGB for packed Adam7 PNGs before Gemma and Qwen preprocessing', async () => {
  const cases = JSON.parse(await readFile(fixturesUrl, 'utf8')) as Fixture[];
  const magickWasm = await readFile(new URL(import.meta.resolve('@imagemagick/magick-wasm/magick.wasm')));
  let codecReads = 0;
  for (const {name, width, height, base64, pillowRgbSha256} of cases) {
    await assert.rejects(
      decodeStrandsImage(base64, {
        maxImagePixels: width * height - 1,
        async loadMagick() {
          throw new Error('Image limits must be enforced before loading a decoder.');
        }
      }),
      RangeError,
      name
    );
    const {
      rgb,
      width: actualWidth,
      height: actualHeight
    } = await decodeStrandsImage(base64, {
      maxImagePixels: 4096 * 4096,
      maxImageBytes: 16 * 1024 * 1024,
      async loadMagick() {
        codecReads++;
        return magickWasm;
      }
    });
    assert.deepEqual([actualWidth, actualHeight], [width, height], name);
    assert.equal(createHash('sha256').update(rgb).digest('hex'), pillowRgbSha256, name);

    const gemma = await prepareGemma4V2EncodedImage(base64, {magickWasm});
    const expectedGemma = prepareGemma4V2Image(rgb, width, height);
    assert.deepEqual(gemma, {...expectedGemma, width, height}, name);

    const plain = Buffer.from(encodePng({width, height, channels: 3, data: rgb})).toString('base64');
    const qwen = await preprocessQwen35Image(base64, {magickWasm});
    const expectedQwen = await preprocessQwen35Image(plain, {magickWasm});
    assert.deepEqual(qwen, expectedQwen, name);
  }
  assert.equal(codecReads, cases.length);
});
