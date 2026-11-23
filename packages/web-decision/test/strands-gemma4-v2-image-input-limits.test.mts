import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import {createGemma4V2CuratedAdapter} from '../src/curated/adapters/strands-gemma4-v2.mts';
import {createGemma4V2ImagePreparer, prepareGemma4V2EncodedImage} from '../src/strands/gemma4-v2-image.mts';
import {GEMMA4_V2_MODEL} from '../src/strands/gemma4-v2-loader.mts';
import {createGemma4V2Session, type Gemma4V2RunnerInput} from '../src/strands/gemma4-v2-session.mts';

const png = () => readFile(new URL('./fixtures/strands-gemma4-v2-matrix/1x1.png', import.meta.url));

test('the adapter preserves five real images without an extra image-count limit', async () => {
  const calls: Gemma4V2RunnerInput[] = [];
  const session = createGemma4V2Session({
    tokenizer: {
      encode(text) {
        return {ids: [...new TextEncoder().encode(text)]};
      }
    },
    runner: {
      async run(input) {
        calls.push(input);
        return [0, 0];
      },
      releaseCachedMemory() {},
      dispose() {}
    },
    prepareImage: createGemma4V2ImagePreparer().prepareImage,
    modelName: GEMMA4_V2_MODEL,
    temperature: 1.5456,
    softcap: 30
  });
  const adapter = createGemma4V2CuratedAdapter(
    async () => ({
      ...session,
      async prepareOffline() {},
      async isOfflineReady() {
        return false;
      }
    }),
    async () => null
  );
  const model = await adapter.loadModel({cache: false}, {modelId: GEMMA4_V2_MODEL});
  const encoded = (await png()).toString('base64');
  try {
    await model.systemOne({state: '', images: Array(5).fill(encoded), questions: {q: {instructions: 'yes?'}}});
    assert.equal(calls.length, 1);
    assert.equal(calls[0]!.images.length, 5);
    assert.deepEqual(
      calls[0]!.imageSpans.map((span) => span.end - span.start),
      [256, 256, 256, 256, 256]
    );
  } finally {
    await model.dispose();
  }
});

test('encoded PNG byte length matches the original 16 MiB boundary', async () => {
  const input = await png();
  const bytes = Buffer.alloc(16 * 1024 * 1024 + 1);
  input.copy(bytes);
  const maximum = bytes.subarray(0, bytes.length - 1).toString('base64');
  const accepted = await prepareGemma4V2EncodedImage(maximum);
  assert.equal(accepted.tokens, 256);
  const oversized = bytes.toString('base64');
  await assert.rejects(prepareGemma4V2EncodedImage(oversized), /byte limit/i);
});
