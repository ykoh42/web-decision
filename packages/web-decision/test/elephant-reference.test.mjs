import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import test from 'node:test';
import {packElephantRequest, verifySlots} from '../src/elephant/protocol.mjs';
import {createElephantTokenizer} from '../src/elephant/tokenizer.mjs';

const require = createRequire(import.meta.url);
const {Tokenizer} = require('@huggingface/tokenizers');
const root = new URL('../../../', import.meta.url);
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');

test('production Elephant protocol matches pinned upstream tokenization', async (t) => {
  if (process.env.WEB_DECISION_SKIP_LOCAL_MODEL_ASSETS === '1') return t.skip('Local model assets disabled.');
  let tokenizerBytes, configBytes, fixtureBytes;
  try {
    [tokenizerBytes, configBytes, fixtureBytes] = await Promise.all([
      readFile(new URL('models/elephant-integration-20261030/source/tokenizer.json', root)),
      readFile(new URL('models/elephant-integration-20261030/source/tokenizer_config.json', root)),
      readFile(new URL('experiments/elephant-integration-20261030/reference.json', root))
    ]);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip('Pinned local Elephant reference assets are unavailable.');
    throw error;
  }
  assert.equal(hash(tokenizerBytes), 'a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f');
  assert.equal(hash(configBytes), '2984c03425a28d242744e234a5003786fd289b3713db79fe0958c65eb59d85b2');
  assert.equal(hash(fixtureBytes), 'b82b4c342d3bd42efc298ed9470307609138c1c30b99c384a2128dfea5ada7ae');
  const fixture = JSON.parse(fixtureBytes);
  const {tokenizer, addedTokens} = await createElephantTokenizer(
    Tokenizer,
    JSON.parse(tokenizerBytes),
    JSON.parse(configBytes)
  );
  assert.deepEqual(verifySlots(tokenizer), fixture.slotIds);
  for (const sample of fixture.cases) {
    const packed = packElephantRequest(tokenizer, sample.request, {
      addedTokens,
      maxBranchTokens: 16384
    });
    assert.deepEqual(packed.prefixIds, sample.prefixIds);
    assert.deepEqual(
      packed.branches.map((branch) => branch.ids),
      sample.branches.map((branch) => branch.ids)
    );
  }
});
