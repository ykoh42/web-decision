import test from 'node:test';
import assert from 'node:assert/strict';
import {blockMask, packElephantRequest, SLOT_LABELS, unpackElephantScores} from '../src/elephant/protocol.mjs';
import {graphInputs} from '../src/elephant/graph-inputs.mjs';
import type {ElephantPackedRequest, ElephantRequest, ElephantSlotTokenizer} from '../src/elephant/protocol.mjs';

// This tokenizer is synthetic: each UTF-16 unit becomes an independent token.
const tokenizer = {
  encode: (text: string) => ({ids: Array.from(text, (char: string) => char.codePointAt(0)!)})
} as unknown as ElephantSlotTokenizer;
const request: ElephantRequest = {state: 'state', questions: [{instructions: '  choose  ', options: ['one', 'two']}]};
const config: Parameters<typeof packElephantRequest>[2] = {addedTokens: ['<reserved>'], maxBranchTokens: 10000};

test('renders ordered options and counts the shared prefix only once', () => {
  const input = {...request, questions: [...request.questions, {instructions: 'again', options: ['x', 'y']}]};
  const packed = packElephantRequest(tokenizer, input, config);
  assert.match(
    packed.branches[0]!.suffix,
    /^Question: choose\n\nOptions:\nA\. one\nB\. two\n\nAnswer with one letter: A, B\.$/
  );
  assert.equal(
    packed.tokenCount,
    packed.prefixIds.length + packed.branches.reduce((sum, branch) => sum + branch.suffixIds.length, 0)
  );
  assert.deepEqual(packed.branches[1]!.ids, [...packed.prefixIds, ...packed.branches[1]!.suffixIds]);
  assert.equal(SLOT_LABELS.length, 256);
  assert.equal(SLOT_LABELS[26], 'AA');
});

test('enforces the inclusive branch token limit and rejects a crossed token boundary', () => {
  const packed = packElephantRequest(tokenizer, request, config);
  const exact = {...config, maxBranchTokens: packed.branches[0]!.ids.length};
  assert.equal(packElephantRequest(tokenizer, request, exact).tokenCount, packed.tokenCount);
  assert.throws(
    () => packElephantRequest(tokenizer, request, {...exact, maxBranchTokens: exact.maxBranchTokens - 1}),
    /Branch exceeds/
  );
  assert.throws(
    () =>
      packElephantRequest(
        {encode: (text: string) => ({ids: [text.length]})} as unknown as ElephantSlotTokenizer,
        request,
        config
      ),
    /Unsafe token boundary/
  );
});

test('checks invalid text and preserves offered choice order and first tie', () => {
  for (const state of ['<reserved>', 'QEV_PREFIX_MARK', '\ud800', '\udc00'])
    assert.throws(() => packElephantRequest(tokenizer, {...request, state}, config));
  assert.throws(
    () => packElephantRequest(tokenizer, {...request, questions: [{instructions: 'x', options: ['a', 'a']}]}, config),
    /Duplicate/
  );
  const packed = {tokenCount: 5, branches: [{options: ['z', 'a', 'b']}]} as unknown as ElephantPackedRequest;
  const answer = unpackElephantScores([[1000, 1000, 999]] as unknown as Float32Array[], packed).answers[0]!;
  assert.equal(answer.selected, 'z');
  assert.equal(answer.selectedIndex, 0);
  assert.deepEqual(
    answer.probabilities.map(({option}) => option),
    ['z', 'a', 'b']
  );
  assert.ok(Math.abs(answer.probabilities.reduce((sum, {probability}) => sum + probability!, 0) - 1) < 1e-12);
  assert.throws(() => unpackElephantScores([[1, NaN, 2]] as unknown as Float32Array[], packed), /Invalid logits/);
});

test('graph inputs match block attention at both supported token boundaries', () => {
  for (const [length, prefix] of [
    [4, 1],
    [5, 3],
    [128, 64]
  ] as Array<[number, number]>) {
    const ids = Array.from({length}, (_, i) => i);
    const [tokens, positions, mask, valid, slot] = graphInputs(ids, prefix);
    const expected = blockMask(prefix, length, length).allowed.flat();
    assert.deepEqual([...tokens!.data], ids);
    assert.deepEqual([...positions!.data], ids);
    assert.deepEqual(
      [...mask!.data],
      expected.map((allowed) => (allowed ? 0 : 0xfbff))
    );
    assert.ok(valid!.data.every((value) => value === 0x3c00));
    assert.equal(slot!.data[0], length - 1);
  }
  for (const [ids, prefix] of [
    [[1, 2, 3], 1],
    [Array(129).fill(1), 1],
    [[1, 2, 3, 248320], 1],
    [[1, 2, 3, 4], 0],
    [[1, 2, 3, 4], 4]
  ] as Array<[number[], number]>)
    assert.throws(() => graphInputs(ids, prefix));
});

test('labels the 27th option without changing order and supports the 255-option protocol limit', () => {
  const options = Array.from({length: 255}, (_, i) => `option ${i}`);
  const branch = packElephantRequest(tokenizer, {...request, questions: [{instructions: 'Pick', options}]}, config)
    .branches[0]!;
  assert.match(branch.suffix, /Z\. option 25\nAA\. option 26/);
  assert.match(branch.suffix, /Answer with one label: A, B,/);
  assert.deepEqual(branch.options, options);
  assert.throws(
    () =>
      packElephantRequest(
        tokenizer,
        {...request, questions: [{instructions: 'Pick', options: [...options, 'extra']}]},
        config
      ),
    /2 to 255/
  );
});

test('rejects malformed questions and reserved tokens in all prompt fields', () => {
  for (const questions of [
    [],
    [{instructions: '', options: ['a', 'b']}],
    [{instructions: 'ok', options: ['a']}],
    [{instructions: 'ok', options: ['a', ' ']}],
    [{instructions: '<reserved>', options: ['a', 'b']}],
    [{instructions: 'ok', options: ['a', '<reserved>']}],
    [{instructions: 'ok', options: ['a', 1]}]
  ]) {
    assert.throws(() => packElephantRequest(tokenizer, {...request, questions} as unknown as ElephantRequest, config));
  }
  assert.throws(() => packElephantRequest(tokenizer, request, {...config, addedTokens: ['']}), /reserved-token/);
  assert.throws(() => packElephantRequest(tokenizer, request, {...config, maxBranchTokens: 0}), /token limits/);
});
