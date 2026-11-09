import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import * as current from '../src/tinydecide/corrections.js';

test('correction helpers match the pinned upstream implementation', async () => {
  const source = await readFile(new URL('./fixtures/vendor/tinydecide-corrections.js.txt', import.meta.url));
  assert.equal(
    createHash('sha256').update(source).digest('hex'),
    'ee0de6af2b114b2c665c07c1e7908646c7afb939f69839184f72bf0436f990b9'
  );
  const upstream = (await import(`data:text/javascript;base64,${source.toString('base64')}`)) as typeof current;
  const beta = [0.1, 0.2, 0.3, 0.4, 0.5];
  const choices = [
    [
      {v: [1, 0], z: [1, 0, -1]},
      {v: [0.5, 0.5], z: [0.5, 0, -0.5]}
    ],
    [{v: [0, 1], z: [-1, 1, 0]}],
    []
  ];
  const yesNo = [[{v: [1, 0], z: [0, -1]}], [{v: [0, 1], z: [0, 1]}]];

  for (const count of [0, 1, 2, 3, 4, 8, 9, 100]) {
    assert.equal(current.bucketK(count), upstream.bucketK(count));
  }
  assert.deepEqual(current.meanVec(choices[0]!, 2), upstream.meanVec(choices[0]!, 2));
  assert.throws(() => current.meanVec([], -1), RangeError);
  assert.throws(() => upstream.meanVec([], -1), RangeError);
  assert.equal(
    current.lambdaFor('choice', choices, [0.25, 0.75], beta),
    upstream.lambdaFor('choice', choices, [0.25, 0.75], beta)
  );
  assert.equal(current.lambdaFor('noul', yesNo, [0.5, 0.5], beta), upstream.lambdaFor('noul', yesNo, [0.5, 0.5], beta));
  assert.deepEqual(current.makeProtos('choice', choices, beta), upstream.makeProtos('choice', choices, beta));
  assert.deepEqual(current.makeProtos('noul', yesNo, beta, [0, 0]), upstream.makeProtos('noul', yesNo, beta, [0, 0]));
  assert.deepEqual(current.makeProtos('score', [[], []], beta), upstream.makeProtos('score', [[], []], beta));
  assert.equal(current.makeProtos('span', null as never, beta), upstream.makeProtos('span', null as never, beta));
});
