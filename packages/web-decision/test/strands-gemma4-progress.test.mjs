import assert from 'node:assert/strict';
import test from 'node:test';
import {createGemma4Progress} from '../src/strands/gemma4-progress.mjs';

test('Gemma 4 loader reports each verified byte once against the full shard total', () => {
  const events = [];
  const progress = createGemma4Progress((event) => events.push(event), 100);

  progress(70); // runtime manifest, graph, specs, tokenizer, and cache metadata
  progress(20); // first weight shard
  progress(10); // last weight shard

  assert.deepEqual(events, [
    {loadedBytes: 70, totalBytes: 100, fraction: 0.7},
    {loadedBytes: 90, totalBytes: 100, fraction: 0.9},
    {loadedBytes: 100, totalBytes: 100, fraction: 1}
  ]);
  assert.throws(() => progress(1), RangeError);
});

test('Gemma 4 loader rejects invalid progress totals and increments', () => {
  assert.throws(() => createGemma4Progress(undefined, 0), TypeError);
  const progress = createGemma4Progress(undefined, 10);
  assert.throws(() => progress(0), RangeError);
  assert.throws(() => progress(-1), RangeError);
  assert.throws(() => progress(11), RangeError);
});
