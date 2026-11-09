import assert from 'node:assert/strict';
import test from 'node:test';
import {KANANA_MODEL_MAX_LENGTH} from '../src/kanana/contract.mjs';
import {protocol, truncation} from './kanana-core-modules.mjs';

const {
  answerKananaQuestion,
  prepareKananaSystemOne,
  prepareKananaTextSystemOne,
  renderKananaQueryInput,
  renderKananaQueryMessages
} = protocol;
const {KANANA_MAX_LENGTH, truncateKananaQuery} = truncation;

function choice(
  criteria = new Map([
    ['billing', 'Billing'],
    ['support', 'Support']
  ])
) {
  return {type: 'choice', instructions: 'Select a route', criteria};
}

function codePointTokenizer() {
  return {
    encode(value) {
      return {ids: Array.from(value, (character) => character.codePointAt(0) + 1)};
    },
    decode(ids) {
      return Array.from(ids, (id) => String.fromCodePoint(id - 1)).join('');
    }
  };
}

function queryInput(state) {
  const prepared = prepareKananaTextSystemOne({state, questions: {route: choice()}});
  return renderKananaQueryInput(prepared, 'route');
}

function truncateQuery(input, tokenizer, options = {}) {
  return truncateKananaQuery(tokenizer, input.messages, [], {
    editableUserTextIndexes: input.editableUserTextIndexes,
    ...options
  });
}

test('default query length matches the pinned 20,480-token limit at the exact boundary', () => {
  assert.equal(KANANA_MODEL_MAX_LENGTH, 20_480);
  assert.equal(KANANA_MAX_LENGTH, KANANA_MODEL_MAX_LENGTH);

  const tokenizer = codePointTokenizer();
  const overhead = truncateQuery(queryInput(''), tokenizer).tokenCount;
  const stateAtLimit = 'x'.repeat(KANANA_MAX_LENGTH - overhead);
  const exact = truncateQuery(queryInput(stateAtLimit), tokenizer);
  const over = queryInput(`${stateAtLimit}x`);
  const truncated = truncateQuery(over, tokenizer);

  assert.equal(exact.tokenCount, KANANA_MODEL_MAX_LENGTH);
  assert.equal(exact.truncated, false);
  assert.equal(truncated.tokenCount, KANANA_MODEL_MAX_LENGTH);
  assert.equal(truncated.truncated, true);
  assert.equal(truncated.messages[1].content[over.editableUserTextIndexes[0]].text, stateAtLimit);
  assert.throws(() => truncateQuery(over, tokenizer, {truncateInputs: false}), {
    code: 'input_too_long',
    maxLength: KANANA_MODEL_MAX_LENGTH,
    tokenCount: KANANA_MODEL_MAX_LENGTH + 1
  });
});

test('choice options and response entries preserve criteria order', () => {
  const prepared = prepareKananaSystemOne({
    state: '',
    questions: new Map([
      [
        'route',
        choice(
          new Map([
            ['10', 'ten'],
            ['2', 'two'],
            ['1', 'one']
          ])
        )
      ]
    ])
  });
  const entry = prepared.questionEntries[0];
  const userText = renderKananaQueryMessages(prepared, 'route')[1]
    .content.map(({text}) => text ?? '')
    .join('');
  const answer = answerKananaQuestion(entry.question, entry.candidateKeys, [0, 0, 1]);

  assert.match(userText, /Options:\nA: 10: ten\nB: 2: two\nC: 1: one/);
  assert.deepEqual(entry.candidateKeys, ['10', '2', '1']);
  assert.equal(answer.choice, '1');
  assert.deepEqual(
    answer.probabilityEntries.map(([key]) => key),
    ['10', '2', '1']
  );
});

test('Noul reports P(true) without adding confidence or an abstention field', () => {
  const answer = answerKananaQuestion({type: 'noul'}, ['false', 'true'], [Math.log(3), 0]);

  assert.equal(answer.type, 'noul');
  assert.ok(Math.abs(answer.noul - 0.25) < 1e-12);
  assert.deepEqual(Object.keys(answer), ['type', 'noul']);
  assert.equal(Object.hasOwn(answer, 'abstain'), false);
});

test('score answers keep ordered levels in their probabilities and legend', () => {
  const question = {type: 'score', criteria: ['low', 'middle', 'high']};
  const answer = answerKananaQuestion(question, ['0', '1', '2'], [0, 0, 0]);

  assert.equal(answer.score, 1);
  assert.deepEqual(answer.legend, {0: 'low', 1: 'middle', 2: 'high'});
  assert.deepEqual(
    answer.probabilityEntries.map(([key]) => key),
    ['0', '1', '2']
  );
});

test('empty state is valid, while empty questions and empty choice criteria are rejected', () => {
  const prepared = prepareKananaSystemOne({state: '', questions: {route: choice()}});

  assert.equal(prepared.stateFormat, 'text');
  assert.equal(prepared.state, '');
  assert.throws(() => prepareKananaSystemOne({state: '', questions: {}}), {code: 'questions'});
  assert.throws(() => prepareKananaSystemOne({state: '', questions: {route: choice(new Map())}}), {
    code: 'choice_count'
  });
});

test('text-only requests reject image attachments but accept image-shaped JSON state data', () => {
  const dataState = {attachment: {type: 'image', image: 'application data'}};
  const prepared = prepareKananaTextSystemOne({state: dataState, questions: {route: choice()}});

  assert.equal(prepared.stateFormat, 'json');
  assert.deepEqual(prepared.state, dataState);
  assert.throws(() => prepareKananaTextSystemOne({state: '', images: [], questions: {route: choice()}}), {
    code: 'images'
  });
  assert.throws(
    () =>
      prepareKananaTextSystemOne({
        state: [{role: 'user', content: [{type: 'image', image: 'photo.png'}]}],
        stateFormat: 'messages',
        questions: {route: choice()}
      }),
    {code: 'images'}
  );
  assert.equal(
    prepareKananaSystemOne({state: '', images: ['photo.png'], questions: {route: choice()}}).images.length,
    1
  );
});
