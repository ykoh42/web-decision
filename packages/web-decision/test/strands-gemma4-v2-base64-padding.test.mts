import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import {WebDecisionError} from '../src/curated/errors.ts';
import {prepareGemma4V2EncodedImage} from '../src/strands/gemma4-v2-image.mts';
import {decodeStrandsImage} from '../src/strands/qwen35-image-input.mts';

test('Gemma and Qwen honor pinned Python base64 validation at every padding boundary', async () => {
  const png = await readFile(new URL('./fixtures/qwen35-image-input/png-orientation-1.png', import.meta.url));
  assert.equal(png.length % 3, 0);
  for (let trailingBytes = 0; trailingBytes < 3; trailingBytes++) {
    const payload = Buffer.concat([png, Buffer.alloc(trailingBytes)])
      .toString('base64')
      .replace(/=+$/u, '');
    for (let padding = 0; padding <= 5; padding++) {
      const image = payload + '='.repeat(padding);
      const accepted = trailingBytes === 0 || padding === 3 - trailingBytes;
      if (accepted) {
        const qwen = await decodeStrandsImage(image, {maxImagePixels: 4096 * 4096});
        const gemma = await prepareGemma4V2EncodedImage(image);
        assert.deepEqual(
          [qwen.width, qwen.height, gemma.width, gemma.height],
          [64, 32, 64, 32],
          `${trailingBytes}:${padding}`
        );
      } else {
        await assert.rejects(
          decodeStrandsImage(image, {maxImagePixels: 4096 * 4096}),
          /base64/i,
          `${trailingBytes}:${padding} Qwen`
        );
        await assert.rejects(prepareGemma4V2EncodedImage(image), /base64/i, `${trailingBytes}:${padding} Gemma`);
      }
    }
  }
  const valid = png.toString('base64');
  for (const invalid of [valid.slice(0, 8) + ' ' + valid.slice(8), valid + '=A', '=' + valid]) {
    await assert.rejects(decodeStrandsImage(invalid, {maxImagePixels: 4096 * 4096}), /base64/i);
    await assert.rejects(prepareGemma4V2EncodedImage(invalid), /base64/i);
  }
  const dataUri = `data:image/png;base64,${valid}===`;
  assert.equal((await prepareGemma4V2EncodedImage(dataUri)).width, 64);
});

test('Gemma classifies invalid base64 without rewriting Qwen or codec asset errors', async () => {
  await assert.rejects(prepareGemma4V2EncodedImage('!'), (error) => {
    assert.ok(error instanceof WebDecisionError);
    assert.equal(error.code, 'INVALID_INPUT');
    assert.equal(error.message, 'Gemma 4 v2 image input is not valid base64.');
    return true;
  });
  await assert.rejects(decodeStrandsImage('!', {maxImagePixels: 4096 * 4096}), {
    name: 'TypeError',
    message: 'Qwen3.5 image input is not valid base64.'
  });
  const jpeg = await readFile(new URL('./fixtures/strands-gemma4-v2-image-formats/progressive.jpg', import.meta.url));
  const unavailable = new WebDecisionError('ASSET_LOAD_FAILED', 'Pinned JPEG codec unavailable.');
  await assert.rejects(
    prepareGemma4V2EncodedImage(jpeg.toString('base64'), {
      codecs: {
        prepareJpeg: () => Promise.reject(unavailable),
        loadMagick: () => Promise.reject(new Error('Unexpected Magick codec request.'))
      }
    }),
    (error) => error === unavailable
  );
});
