import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {test} from 'node:test';
import binding from '../assets/strands-gemma4-v2/decoder-bindings.json' with {type: 'json'};
import inventory from '../assets/strands-gemma4-v2/source-inventory.json' with {type: 'json'};
import {
  assertGemma4V2GraphBindings,
  loadGemma4V2BindingsFromManifests,
  loadPinnedGemma4V2Bindings
} from '../src/strands/gemma4-v2-bindings.mjs';

const hash = (value: unknown) =>
  createHash('sha256')
    .update(JSON.stringify(value) + '\n')
    .digest('hex');

test('pinned decoder metadata covers every layer and rejects incorrect graph input order or size', async () => {
  const bindings = await loadPinnedGemma4V2Bindings();
  for (let index = 0; index < 35; index++) {
    const layer = bindings.layer(index);
    assert.equal(layer.layer, index);
    assert.equal(layer.family, index < 15 ? (index % 5 === 4 ? 4 : 0) : index % 5 === 4 ? 19 : 15);
    assertGemma4V2GraphBindings(layer, layer.parameters);
    const reversed = [...layer.parameters].reverse();
    assert.throws(() => assertGemma4V2GraphBindings(layer, reversed), /binding/i);
    assert.throws(
      () =>
        assertGemma4V2GraphBindings(
          layer,
          layer.parameters.map((item, i) => (i ? item : {...item, packedWords: item.packedWords + 1}))
        ),
      /binding/i
    );
  }
  assert.equal(bindings.sourceGraph(0).sha256, '5eae05c57b90ba0278441b812ce2b1dfb2b57306bc142d31e657ebf99dc129e0');
  assert.throws(() => bindings.layer(35), /layer/i);
});

test('rejects tampering and inconsistent parameter shapes', async () => {
  await assert.rejects(loadGemma4V2BindingsFromManifests(binding, '0'.repeat(64), inventory), /digest/i);
  const altered = structuredClone(binding);
  altered.layers[0]!.parameters[0]!.shape[0]! += 1;
  await assert.rejects(loadGemma4V2BindingsFromManifests(altered, hash(altered), inventory), /parameter/i);
  const changedSource = structuredClone(inventory);
  changedSource.bytes += 1;
  await assert.rejects(loadGemma4V2BindingsFromManifests(binding, hash(binding), changedSource), /digest/i);
});
