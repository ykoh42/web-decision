import test from 'node:test';
import assert from 'node:assert/strict';
import {buildMapikaLabels, packMapikaRequest, unpackMapikaScores} from '../src/mapika.js';

// A small deterministic tokenizer keeps these package tests independent of
// downloaded model assets. Actual tokenizer and upstream parity are tested
// separately in experiments/mapika-0.8b-protocol-20261030/.
function tokenizer() {
  return {
    encode(text: string) {
      if (/^[A-Z]{1,2}$/.test(text)) {
        let id = 1000;
        for (const letter of text) id = id * 27 + letter.charCodeAt(0) - 64;
        return {ids: [id]};
      }
      const ids = [];
      for (let i = 0; i < text.length;) {
        if (text.startsWith('<|control|>', i)) {
          ids.push(7);
          i += '<|control|>'.length;
        } else {
          const point = text.codePointAt(i)!;
          ids.push(point + 100);
          i += point > 0xffff ? 2 : 1;
        }
      }
      return {ids};
    }
  };
}

const limits = {maxInputTokens: 8192, maxStateTokens: 1024, controlIds: [7]};
const question = {instructions: 'route', options: ['red', 'blue']};
const request = () => ({state: 'parcel', questions: [{...question, options: [...question.options]}]});

test('packs joint questions with correct slots and snapshots the option arrays', () => {
  const tok = tokenizer(),
    labels = buildMapikaLabels(tok),
    input = request();
  input.questions.push({instructions: 'next', options: ['yes', 'no']});
  const packed = packMapikaRequest(tok, labels, input, limits);
  const start = 'Context:\nparcel';
  const q1 = '\n\nQuestion 1: route\nOptions:\n(A) red\n(B) blue\nAnswer 1: (';
  const q2 = '\n\nQuestion 2: next\nOptions:\n(A) yes\n(B) no\nAnswer 2: (';
  assert.deepEqual(packed.ids, [...tok.encode(start).ids, ...tok.encode(q1).ids, ...tok.encode(q2).ids]);
  assert.deepEqual(
    packed.groups.map((group) => group.slot),
    [tok.encode(start + q1).ids.length - 1, packed.ids.length - 1]
  );
  input.questions[0]!.options[0] = 'changed';
  assert.equal(packed.groups[0]!.options[0], 'red');
});

test('uses upstream label-token construction beyond the ten-option boundary', () => {
  const tok = tokenizer(),
    labels = buildMapikaLabels(tok);
  assert.equal(labels.ids.length, 255);
  for (const count of [10, 11, 255]) {
    const options = Array.from({length: count}, (_, i) => `option ${i}`);
    const packed = packMapikaRequest(tok, labels, {state: '', questions: [{instructions: 'pick', options}]}, limits);
    const prefix = tok.encode('Context:\n').ids;
    const head = '\n\nQuestion: pick\nOptions:';
    const tail = '\nAnswer: (';
    const body =
      count <= 10
        ? tok.encode(head + options.map((value, i) => `\n(${labels.names[i]}) ${value}`).join('') + tail).ids
        : [
            ...tok.encode(head).ids,
            ...options.flatMap((value, i) => [...labels.open, labels.ids[i], ...tok.encode(`) ${value}`).ids]),
            ...tok.encode(tail).ids
          ];
    assert.deepEqual(packed.ids, [...prefix, ...body]);
    assert.equal(packed.groups[0]!.slot, packed.ids.length - 1);
  }
});

test('rejects invalid input, control tokens, unpaired surrogates and limits', () => {
  const tok = tokenizer(),
    labels = buildMapikaLabels(tok);
  const rejected = [
    {state: '', questions: []},
    {state: '<|control|>', questions: [question]},
    {state: '', questions: [{instructions: '\ud800', options: ['yes', 'no']}]},
    {state: '', questions: [{instructions: 'ok', options: ['x', 'x']}]},
    {state: '', questions: [{instructions: 'ok', options: ['x']}]},
    {state: '', questions: [{instructions: 'ok', options: ['x', '']}]}
  ];
  for (const item of rejected)
    assert.throws(() => packMapikaRequest(tok, labels, item, limits), {code: 'INVALID_INPUT'});
  assert.throws(() => packMapikaRequest(tok, labels, request(), {...limits, maxStateTokens: 2}), {
    code: 'INVALID_INPUT'
  });
  assert.throws(() => packMapikaRequest(tok, labels, request(), {...limits, maxInputTokens: 10}), {
    code: 'INVALID_INPUT'
  });
  assert.doesNotThrow(() => packMapikaRequest(tok, labels, {...request(), state: '😀'}, limits));
});

test('rejects inconsistent tokenizer outputs and model configuration', () => {
  const tok = tokenizer(),
    labels = buildMapikaLabels(tok);
  assert.throws(() => buildMapikaLabels({encode: () => ({ids: [4.5]})}), {code: 'INVALID_ASSET'});
  assert.throws(() => buildMapikaLabels({encode: () => ({ids: [4]})}), {code: 'INVALID_ASSET'});
  assert.throws(() => packMapikaRequest(tok, labels, request(), {...limits, controlIds: [7, 7]}), {
    code: 'INVALID_ASSET'
  });
  assert.throws(() => packMapikaRequest(tok, labels, request(), {...limits, maxInputTokens: 0}), {
    code: 'INVALID_ASSET'
  });
});

test('returns unscaled logits and temperature-adjusted probabilities for each option', () => {
  const tok = tokenizer(),
    labels = buildMapikaLabels(tok),
    packed = packMapikaRequest(tok, labels, request(), limits);
  const row = new Float32Array(255);
  row.set([1, 3]);
  const result = unpackMapikaScores([row], packed, 2);
  assert.equal(result.tokenCount, packed.ids.length);
  assert.equal(result.answers[0]!.selected, 'blue');
  assert.deepEqual(
    result.answers[0]!.probabilities.map((entry: {logit: number}) => entry.logit),
    [1, 3]
  );
  assert.ok(Math.abs(result.answers[0]!.probabilities[1]!.probability - 1 / (1 + Math.exp(-1))) < 1e-12);
  row.set([0, 0]);
  assert.equal(unpackMapikaScores([row], packed, 1).answers[0]!.selectedIndex, 0);
  // Preserve the original fixed-length fixture allocation.
  // oxlint-disable-next-line unicorn/no-new-array
  const extreme = new Array(255).fill(0);
  extreme[0] = -Number.MAX_VALUE;
  extreme[1] = Number.MAX_VALUE;
  assert.deepEqual(
    unpackMapikaScores([extreme], packed, 1).answers[0]!.probabilities.map((p: {probability: number}) => p.probability),
    [0, 1]
  );
});

test('rejects malformed model output and non-positive temperature', () => {
  const tok = tokenizer(),
    labels = buildMapikaLabels(tok),
    packed = packMapikaRequest(tok, labels, request(), limits);
  const row = new Float32Array(255);
  for (const [rows, temperature] of [
    [[row], 0],
    [[row], NaN],
    [[], 1],
    [[new Float32Array(2)], 1]
  ])
    assert.throws(() => unpackMapikaScores(rows, packed, temperature), {code: 'INFERENCE_ERROR'});
  row[254] = NaN;
  assert.throws(() => unpackMapikaScores([row], packed, 1), {code: 'INFERENCE_ERROR'});
});
