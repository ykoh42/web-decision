import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import {magickRuntime} from '../src/kanana/magick-image.js';
import {MAGICK} from '../src/mapika2b/image-codec-pins.mjs';
import {prepareGemma4V2EncodedImage} from '../src/strands/gemma4-v2-image.mjs';
import {preprocessQwen35Image} from '../src/strands/qwen35-image-input.mjs';

test('shared ImageMagick initialization verifies the usual codec and permits retry after failure', async (t) => {
  const wasm = await readFile(new URL(import.meta.resolve('@imagemagick/magick-wasm/magick.wasm')));
  const url = new URL('https://example.test/codecs/magick.wasm');
  let fetches = 0;
  t.mock.method(globalThis, 'fetch', async (request: string) => {
    assert.equal(request, url.href);
    fetches++;
    return new Response(new Uint8Array(wasm));
  });

  await assert.rejects(magickRuntime(new Uint8Array()), {code: 'INVALID_ASSET'});
  assert.equal(fetches, 0);

  const digest = crypto.subtle.digest.bind(crypto.subtle);
  let checks = 0;
  t.mock.method(crypto.subtle, 'digest', (algorithm: AlgorithmIdentifier, bytes: BufferSource) => {
    if (bytes.byteLength === MAGICK.bytes) checks++;
    return digest(algorithm, bytes);
  });
  const first = magickRuntime(url);
  assert.equal(magickRuntime(wasm), first);
  await first;
  assert.equal(fetches, 1);
  assert.equal(checks, 1);

  const fixture = JSON.parse(
    await readFile(new URL('./fixtures/strands-image-packed-adam7.json', import.meta.url), 'utf8')
  ) as Array<{base64: string}>;
  const [gemma, qwen] = await Promise.all([
    prepareGemma4V2EncodedImage(fixture[0]!.base64, {magickWasm: wasm}),
    preprocessQwen35Image(fixture[1]!.base64, {magickWasm: wasm})
  ]);
  assert.ok(gemma.tokens > 0);
  assert.ok(qwen.pixelValues.length > 0);
  assert.equal(fetches, 1);
  assert.equal(checks, 1);
});
