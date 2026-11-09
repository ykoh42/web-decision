import test from 'node:test';
import assert from 'node:assert/strict';
import {
  KEV_DELIMITERS,
  KEV_MAX_BRANCH_TOKENS,
  KEV_MAX_STATE_TOKENS,
  WebDecisionError,
  packKevRequest,
  unpackKevScores,
  decodeFloat16
} from '../src/kev.js';

function tokenizer() {
  const texts = [];
  return {
    texts,
    encode(text) {
      texts.push(text);
      return {ids: Array.from(text, (c) => c.codePointAt(0) + 200)};
    }
  };
}
const limits = {maxStateTokens: KEV_MAX_STATE_TOKENS, maxBranchTokens: KEV_MAX_BRANCH_TOKENS};

test('packs multiple questions and returns scores for each option', () => {
  const tok = tokenizer();
  const packed = packKevRequest(
    tok,
    {
      state: 'a',
      questions: [
        {instructions: 'x', options: ['b', 'c']},
        {instructions: 'y', options: ['d', 'e']}
      ]
    },
    limits
  );
  assert.equal(packed.ids[0], KEV_DELIMITERS.state);
  for (const group of packed.groups)
    for (const i of group.indices) assert.equal(packed.ids[i], KEV_DELIMITERS.option_end);
  const logits = new Array(packed.ids.length).fill(0);
  logits[packed.groups[0].indices[1]] = 2;
  logits[packed.groups[1].indices[0]] = 1;
  const result = unpackKevScores(logits, packed);
  assert.equal(result.answers[0].selected, 'c');
  assert.equal(result.answers[1].selected, 'd');
  assert.equal(result.tokenCount, packed.ids.length);
  assert.ok(Math.abs(result.answers[0].probabilities.reduce((s, x) => s + x.probability, 0) - 1) < 1e-12);
});

test('an exact probability tie selects the first tied option', () => {
  const packed = packKevRequest(
    tokenizer(),
    {state: '', questions: [{instructions: 'q', options: ['first', 'second']}]},
    limits
  );
  const result = unpackKevScores(new Array(packed.ids.length).fill(0), packed);
  assert.equal(result.answers[0].selected, 'first');
  assert.equal(result.answers[0].selectedIndex, 0);
  assert.deepEqual(
    result.answers[0].probabilities.map((p) => p.probability),
    [0.5, 0.5]
  );
});

test('a single upstream choice is accepted with probability one', () => {
  const packed = packKevRequest(
    tokenizer(),
    {state: '', questions: [{instructions: 'choose if applicable', options: ['only choice']}]},
    limits
  );
  const result = unpackKevScores(new Array(packed.ids.length).fill(0), packed);
  assert.equal(result.answers[0].selected, 'only choice');
  assert.equal(result.answers[0].selectedIndex, 0);
  assert.deepEqual(result.answers[0].probabilities, [{option: 'only choice', probability: 1, logit: 0}]);
});

test('an optional packed-token cap accepts its exact boundary and rejects the next token', () => {
  const tok = tokenizer();
  const cappedLimits = {...limits, maxInputTokens: 1024};
  const cases = [
    {expected: 10, state: '', instructions: 'q'},
    {expected: 512, state: 's'.repeat(127), instructions: 'i'.repeat(376)},
    {expected: 1024, state: 's'.repeat(383), instructions: 'i'.repeat(632)}
  ];
  for (const {expected, state, instructions} of cases) {
    const packed = packKevRequest(tok, {state, questions: [{instructions, options: ['a', 'b']}]}, cappedLimits);
    assert.equal(packed.ids.length, expected);
  }
  assert.throws(
    () =>
      packKevRequest(
        tok,
        {state: 's'.repeat(383), questions: [{instructions: 'i'.repeat(633), options: ['a', 'b']}]},
        cappedLimits
      ),
    {code: 'INVALID_INPUT', message: /packed input exceeds 1024 tokens/i}
  );
});

test('state plus each question branch uses the native 8,192-token limit', () => {
  const tok = tokenizer();
  const requestFor = (instructions) => ({state: '', questions: [{instructions, options: ['']}]});
  const exact = packKevRequest(tok, requestFor('i'.repeat(8187)), limits, {
    allowEmptyOptions: true
  });
  assert.equal(exact.ids.length, KEV_MAX_BRANCH_TOKENS);
  assert.throws(() => packKevRequest(tok, requestFor('i'.repeat(8188)), limits, {allowEmptyOptions: true}), {
    code: 'INVALID_INPUT',
    message: /state and branch exceed 8192/i
  });
});

test('a state-heavy row accepts 8,192 packed tokens and rejects 8,193', () => {
  const exactTokenizer = tokenizer();
  const exact = packKevRequest(
    exactTokenizer,
    {state: 's'.repeat(8185), questions: [{instructions: 'i', options: ['o']}]},
    limits
  );

  assert.equal(exact.ids.length, KEV_MAX_BRANCH_TOKENS);
  assert.equal(exact.ids[0], KEV_DELIMITERS.state);
  assert.deepEqual(
    exactTokenizer.texts.map((text) => text.length),
    [8185, 1, 1]
  );

  const overlongTokenizer = tokenizer();
  assert.throws(
    () =>
      packKevRequest(
        overlongTokenizer,
        {state: 's'.repeat(8186), questions: [{instructions: 'i', options: ['o']}]},
        limits
      ),
    {code: 'INVALID_INPUT', message: /state and branch exceed 8192/i}
  );
  assert.deepEqual(
    overlongTokenizer.texts.map((text) => text.length),
    [8186, 1, 1]
  );
});

test('multiple native-size branches have no fixed aggregate token cap', () => {
  const tok = tokenizer();
  const question = {instructions: 'i'.repeat(8187), options: ['']};
  const packed = packKevRequest(tok, {state: '', questions: [question, question]}, limits, {
    allowEmptyOptions: true
  });
  assert.equal(packed.ids.length, 16_383);
  assert.ok(packed.ids.length > 2_048);
});

test('packed input can exceed the position table size when question branches reuse positions', () => {
  const instructions = [8187, 8187, 8187, 4092, 4096].map((length) => 'i'.repeat(length));
  const request = {state: '', questions: instructions.map((value) => ({instructions: value, options: ['']}))};
  const packed = packKevRequest(tokenizer(), request, limits, {allowEmptyOptions: true});
  assert.equal(packed.ids.length, 32_770);
});

test('accepts 255 distinct options in their original order and rejects 256', () => {
  const tok = tokenizer();
  const options = Array.from({length: 255}, (_, index) => String.fromCodePoint(0x4e00 + index));
  const packed = packKevRequest(tok, {state: '', questions: [{instructions: 'q', options}]}, limits);
  assert.deepEqual(packed.groups[0].options, options);
  assert.equal(packed.groups[0].indices.length, 255);
  assert.throws(
    () =>
      packKevRequest(
        tok,
        {state: '', questions: [{instructions: 'q', options: [...options, String.fromCodePoint(0x4e00 + 255)]}]},
        limits
      ),
    {code: 'INVALID_INPUT', message: /1 to 255 options/i}
  );
  assert.throws(() => packKevRequest(tok, {state: '', questions: [{instructions: 'q', options: []}]}, limits), {
    code: 'INVALID_INPUT',
    message: /1 to 255 options/i
  });
});

test('escapes caller-supplied control tokens in state, questions, and options', () => {
  const tok = tokenizer();
  packKevRequest(
    tok,
    {state: '<|fim_suffix|>', questions: [{instructions: '<|fim_middle|>', options: ['<|box_end|>', 'safe']}]},
    limits
  );
  assert.deepEqual(tok.texts, ['<¦fim_suffix¦>', '<¦fim_middle¦>', '<¦box_end¦>', 'safe']);
});

test('rejects distinct options that encode to the same Kev tokens', () => {
  const tok = tokenizer();
  const request = {state: '', questions: [{instructions: 'choose', options: ['<|box_end|>', '<¦box_end¦>']}]};
  assert.throws(() => packKevRequest(tok, request, limits), {code: 'INVALID_INPUT', message: /same tokens/i});
  const collapsingTokenizer = {
    encode(text) {
      return {ids: text === 'one' || text === 'uno' ? [99] : [42]};
    }
  };
  assert.throws(
    () =>
      packKevRequest(
        collapsingTokenizer,
        {state: '', questions: [{instructions: 'choose', options: ['one', 'uno']}]},
        limits
      ),
    {code: 'INVALID_INPUT', message: /same tokens/i}
  );
  assert.doesNotThrow(() =>
    packKevRequest(
      tok,
      {
        state: '',
        questions: [
          {instructions: 'first', options: ['one', 'two']},
          {instructions: 'second', options: ['one', 'two']}
        ]
      },
      limits
    )
  );
});

test('rejects unpaired UTF-16 surrogates in any text field', () => {
  const tok = tokenizer();
  const base = {state: 'valid 😀 �', questions: [{instructions: 'choose 😀', options: ['yes 😀', 'no �']}]};
  assert.doesNotThrow(() => packKevRequest(tok, base, limits));
  const invalids = [
    {...base, state: 'a\ud800b'},
    {...base, state: 'a\udc00b'},
    {...base, state: '\ud800\ud800'},
    {...base, state: '\udc00\ud800'},
    {...base, questions: [{...base.questions[0], instructions: 'choose \ud800'}]},
    {...base, questions: [{...base.questions[0], options: ['yes', 'no\udc00']}]}
  ];
  for (const request of invalids) {
    assert.throws(() => packKevRequest(tok, request, limits), {code: 'INVALID_INPUT', message: /surrogate/i});
  }
});

test('rejects invalid, duplicate, overlong and non-finite inputs instead of silently changing them', () => {
  const tok = tokenizer();
  assert.throws(() => packKevRequest(tok, {state: '', questions: []}, limits), {code: 'INVALID_INPUT'});
  assert.throws(
    () => packKevRequest(tok, {state: '', questions: [{instructions: 'x', options: ['same', 'same']}]}, limits),
    {code: 'INVALID_INPUT'}
  );
  assert.throws(
    () => packKevRequest(tok, {state: 'x'.repeat(8192), questions: [{instructions: 'q', options: ['a', 'b']}]}, limits),
    {code: 'INVALID_INPUT', message: /state exceeds 8191 tokens/i}
  );
  assert.throws(
    () =>
      packKevRequest(tok, {state: '', questions: [{instructions: 'x'.repeat(8189), options: ['']}]}, limits, {
        allowEmptyOptions: true
      }),
    {code: 'INVALID_INPUT', message: /state and branch exceed 8192/i}
  );
  const packed = packKevRequest(tok, {state: '', questions: [{instructions: 'q', options: ['a', 'b']}]}, limits);
  assert.throws(() => unpackKevScores(new Array(packed.ids.length).fill(NaN), packed), WebDecisionError);
});

test('decodes float16 normal, subnormal and special values', () => {
  assert.equal(decodeFloat16(0x3c00), 1);
  assert.equal(decodeFloat16(0xc000), -2);
  assert.equal(decodeFloat16(1), 2 ** -24);
  assert.equal(decodeFloat16(0x7c00), Infinity);
  assert.ok(Number.isNaN(decodeFloat16(0x7e00)));
});
