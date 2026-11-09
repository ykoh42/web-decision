import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import {pack, unpack, validatePacked} from '../src/nari/adapter.mjs';
import {decodeNariChoice, packNariRequest} from '../src/nari/protocol.mjs';

interface MutablePacked {
  inputIds: number[];
  optQ: number[];
  optType: number[];
  optPos: number[];
  slotPos: number[];
  nLetters: number[];
  options: string[][];
}

const input = JSON.parse(readFileSync(new URL('./fixtures/nari/short.json', import.meta.url)) as unknown as string);
const tokenizer = {
  encode(text: string) {
    return {ids: Array.from({length: Math.ceil(text.length / 3)}, (_, i) => text.codePointAt(i * 3))};
  }
};
const request = {state: 'x', questions: [{instructions: 'x', options: ['a', 'b']}]};

test('packed choice validation rejects changed positions, owners and malformed option groups', () => {
  for (const change of [
    (v: MutablePacked) => {
      v.inputIds[0] = -1;
    },
    (v: MutablePacked) => {
      v.inputIds = [];
    },
    (v: MutablePacked) => {
      v.optQ[0] = 3;
    },
    (v: MutablePacked) => {
      v.optType[0] = 0;
    },
    (v: MutablePacked) => {
      v.optPos[1] = v.optPos[0]!;
    },
    (v: MutablePacked) => {
      v.slotPos[0] = -1;
    },
    (v: MutablePacked) => {
      v.nLetters[0] = 0;
    },
    (v: MutablePacked) => {
      v.options[0]![1] = v.options[0]![0]!;
    },
    (v: MutablePacked) => {
      v.optPos.pop();
    }
  ]) {
    const copy = structuredClone(input);
    change(copy);
    assert.throws(() => validatePacked(copy), {code: 'INVALID_INPUT'});
  }
});

test('unpack requires finite scores of the correct shapes', () => {
  const good = {headScores: Float32Array.of(0, 1), letterScores: new Float32Array(26)};
  assert.equal(unpack(input, good).answers[0]!.selectedIndex, 1);
  for (const bad of [
    {headScores: new Float64Array(2), letterScores: good.letterScores},
    {headScores: Float32Array.of(NaN, 1), letterScores: good.letterScores},
    {headScores: good.headScores, letterScores: null},
    {headScores: good.headScores, letterScores: new Float32Array(1)},
    {headScores: good.headScores, letterScores: Float32Array.from({length: 26}, () => NaN)}
  ])
    assert.throws(() => unpack(input, bad), {code: 'INFERENCE_ERROR'});
});

test('protocol rejects malformed text, controls and unsupported decision fields', () => {
  for (const bad of [
    null,
    {state: '', questions: []},
    {state: '', questions: [{instructions: '', options: ['a', 'b']}]},
    {state: '\ud800', questions: request.questions},
    {state: '', questions: [{instructions: 'x', options: ['a', 'a']}]},
    {state: '', questions: [{instructions: 'x', options: ['a']}]},
    {state: '', questions: [{instructions: 'x', options: ['a', 'b'], criteria: 'extra'}]}
  ])
    assert.throws(() => pack(tokenizer, bad, [248045]), {code: 'INVALID_INPUT'});
  assert.throws(() => packNariRequest(tokenizer, request, []));
  assert.throws(
    () =>
      packNariRequest(
        {
          encode() {
            return {ids: [248045]};
          }
        },
        request,
        [248045]
      ),
    /reserved control token/
  );
});

test('head-only and letter-weighted choice decoding reject invalid scores and temperatures', () => {
  const options = Array.from({length: 27}, (_, i) => String(i));
  const result = decodeNariChoice(
    options,
    Float32Array.from(options, (_, i) => i),
    null,
    {'choice:27+': 1.1189}
  );
  assert.equal(result.selectedIndex, 26);
  assert.throws(() => decodeNariChoice(['a', 'b'], [0, 1], null, {}));
  assert.throws(() => decodeNariChoice(['a', 'b'], [0, 1], [0, 0], {'choice:2': 0}));
  assert.throws(() => decodeNariChoice(['a', 'b'], [0, 1], [0, 0], {}, 2));
  assert.throws(() => decodeNariChoice(['a', 'b'], [0, NaN], [0, 0], {}));
});
