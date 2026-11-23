import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import {prepareGemma4V2EncodedImage} from '../src/strands/gemma4-v2-image.mts';
import {decodeStrandsImage} from '../src/strands/qwen35-image-input.mts';

const directory = new URL('./fixtures/strands-gemma4-v2-image-icc-boundary/', import.meta.url);
async function encoded(name: string) {
  return (await readFile(new URL(name, directory))).toString('base64');
}

test('PNG compressed ICC data respects the original Pillow 1 MiB decompression boundary', async () => {
  const allowed = await encoded('icc-1048576.png');
  const tooLarge = await encoded('icc-1048577.png');
  const decoded = await decodeStrandsImage(allowed, {maxImagePixels: 4096 * 4096});
  assert.deepEqual([decoded.width, decoded.height, ...decoded.rgb], [1, 1, 20, 30, 40]);
  assert.equal((await prepareGemma4V2EncodedImage(allowed)).width, 1);
  await assert.rejects(
    decodeStrandsImage(tooLarge, {maxImagePixels: 4096 * 4096}),
    /decompression limit|decompressed data too large/i
  );
  await assert.rejects(prepareGemma4V2EncodedImage(tooLarge), /decompression limit|decompressed data too large/i);
});
