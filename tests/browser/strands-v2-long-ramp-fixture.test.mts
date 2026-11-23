import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import {buildLongRampInput} from './strands-v2-long-ramp-fixture.mts';

test('long input ramp preserves both ends and all requested tokens', async () => {
  const path = process.env.WEB_DECISION_STRANDS_V2_CHAIN_ROOT;
  if (!path) throw new Error('Set WEB_DECISION_STRANDS_V2_CHAIN_ROOT to the local reference.');
  const fixture = JSON.parse(await readFile(`${path}/e2b-v2-text-1024-reference/tokens.json`, 'utf8'));
  const original = buildLongRampInput(fixture, 1024);
  assert.deepEqual(original.state, fixture.state);
  assert.deepEqual(original.question, fixture.questions[0]);
  for (const length of [1025, 2048, 4096]) {
    const result = buildLongRampInput(fixture, length);
    assert.equal(result.state.length + result.question.length, length);
    assert.deepEqual(result.state.slice(0, 9), fixture.state.slice(0, 9));
    assert.deepEqual(result.state.slice(-5), fixture.state.slice(-5));
    assert.deepEqual(result.question, fixture.questions[0]);
  }
  assert.throws(() => buildLongRampInput(fixture, 1), /too short/i);
  assert.throws(() => buildLongRampInput(fixture, 1024.5), /integer/i);
  const corrupted = structuredClone(fixture);
  corrupted.state[15] = -1;
  assert.throws(() => buildLongRampInput(corrupted, 1024), /layout/i);
});
