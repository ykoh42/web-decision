import assert from 'node:assert/strict';
import test from 'node:test';
import {StrandsGemma4Runner} from '../src/strands/gemma4-e2b-runner.mjs';

test('runner rejects the full pinned whitespace set as an option label', async () => {
  const whitespace = String.fromCodePoint(
    0x9,
    0xa,
    0xb,
    0xc,
    0xd,
    0x1c,
    0x1d,
    0x1e,
    0x1f,
    0x20,
    0x85,
    0xa0,
    0x1680,
    0x2000,
    0x2001,
    0x2002,
    0x2003,
    0x2004,
    0x2005,
    0x2006,
    0x2007,
    0x2008,
    0x2009,
    0x200a,
    0x2028,
    0x2029,
    0x202f,
    0x205f,
    0x3000
  );
  const runner = Object.create(StrandsGemma4Runner.prototype);

  await assert.rejects(
    runner.run({
      inputIds: [1, 2, 3, 4],
      options: ['red', whitespace],
      optionIndices: [0, 1],
      answerIndex: 3,
      type: 'choice'
    }),
    {code: 'INVALID_INPUT'}
  );
});
