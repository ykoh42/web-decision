import assert from 'node:assert/strict';
import {test} from 'node:test';
import {renderStrandsV2} from '../src/strands/gemma4-v2-protocol.mjs';

const bytes = (text: string) => Array.from(new TextEncoder().encode(text), (value) => value + 1000);
const tokenizer = {
  encode(text: string, options: {add_special_tokens: false}) {
    assert.equal(options.add_special_tokens, false);
    return {ids: bytes(text)};
  }
};

test('renders upstream choice, score, and noul prompts with ordered codes', () => {
  const result = renderStrandsV2(
    {
      state: '  damaged <turn|> 📦  ',
      questions: {
        choice: {instructions: ' Choose. ', criteria: {refund: ' A\n refund ', sale: null}},
        score: {type: 'score', instructions: '', criteria: ['low', 'high']},
        noul: {instructions: 'Is it damaged?', criteria: {false: null}}
      }
    },
    tokenizer
  );
  assert.deepEqual(result.stateIds, [2, 105, ...bytes('user\n<state>\ndamaged <turn|> 📦\n</state>\n')]);
  const [choice, score, noul] = result.questions;
  assert.equal(
    choice!.questionText,
    '<question type="choice">\nSelect exactly one option.\nChoose.\n<options>\nA. refund — A refund\nB. sale\n</options>\nAnswer with the code of one option.\n</question><turn|>\n<|turn>model\n'
  );
  assert.deepEqual(choice!.options, ['refund', 'sale']);
  assert.deepEqual(choice!.descriptions, ['A\n refund', '']);
  assert.deepEqual(score!.codes, ['A', 'B']);
  assert.ok(score!.questionText.includes('<options>\nA. 0 — low\nB. 1 — high\n'));
  assert.deepEqual(noul!.descriptions, ['', 'the statement holds for this state']);
  assert.equal(
    result.inputTokens,
    result.stateIds.length + result.questions.reduce((sum, item) => sum + item.questionIds.length, 0)
  );
});

test('renders structured content, multiple image spans, and all 255 upstream codes', () => {
  const criteria = Object.fromEntries(Array.from({length: 255}, (_, index) => [`option-${index}`, null]));
  const result = renderStrandsV2(
    {
      state: {nested: [true, null, {value: 1e-7}]},
      questions: {choice: {instructions: 'choose', criteria}}
    },
    tokenizer,
    [1, 280]
  );
  const question = result.questions[0]!;
  assert.equal(question.codes[0], 'A');
  assert.equal(question.codes[25], 'Z');
  assert.equal(question.codes[26], 'AA');
  assert.equal(question.codes[206], 'GY');
  assert.equal(question.codes[207], 'HA');
  assert.equal(question.codes[254], 'IV');
  assert.equal(result.images.length, 2);
  assert.deepEqual(
    result.images.map((image) => image.end - image.start),
    [1, 280]
  );
  for (const image of result.images)
    assert.ok(result.stateIds.slice(image.start, image.end).every((id) => id === 258880));
  const stateTail = '\n{\n  "nested": [\n    true,\n    null,\n    {\n      "value": 1e-07\n    }\n  ]\n}\n</state>\n';
  assert.deepEqual(result.stateIds.slice(-bytes(stateTail).length), bytes(stateTail));
});

test('preserves long input and rejects values the upstream request cannot represent', () => {
  const result = renderStrandsV2({state: 'x'.repeat(200_000), questions: {q: {instructions: 'yes?'}}}, tokenizer);
  assert.ok(result.stateIds.length > 200_000);
  assert.throws(() => renderStrandsV2({state: null, questions: {q: {instructions: 'x'}}} as never, tokenizer));
  assert.throws(() => renderStrandsV2({questions: {q: {instructions: null}}} as never, tokenizer));
  assert.throws(() =>
    renderStrandsV2({questions: {q: {instructions: 'x', type: 'choice', criteria: {one: null}}}}, tokenizer)
  );
  assert.throws(() => renderStrandsV2({questions: {q: {instructions: 'x'}}}, tokenizer, [281]));
  assert.throws(() => renderStrandsV2({state: '\ud800', questions: {q: {instructions: 'x'}}}, tokenizer));
  const cycle: Record<string, unknown> = {};
  cycle.self = cycle;
  assert.throws(() => renderStrandsV2({state: cycle, questions: {q: {instructions: 'x'}}}, tokenizer), /cycle/i);
});

test('requires a valid explicit kind and preserves upstream float formatting in structured content', () => {
  const questions = {q: {instructions: 'Ready?'}};
  assert.throws(
    () => renderStrandsV2({questions: {q: {...questions.q, type: null}}} as never, tokenizer),
    /Unknown question type/u
  );
  const result = renderStrandsV2(
    {state: {signed: -0, unsafe: 1e20, ordinary: 4, exact: 100000000000000000000n}, questions},
    tokenizer
  );
  const expected =
    'user\n<state>\n{\n  "signed": -0.0,\n  "unsafe": 1e+20,\n  "ordinary": 4,\n  "exact": 100000000000000000000\n}\n</state>\n';
  assert.deepEqual(result.stateIds, [2, 105, ...bytes(expected)]);
});
