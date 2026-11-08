import test from 'node:test';
import assert from 'node:assert/strict';
import {packChoiceRequest, unpackChoiceScores, validatePacked} from '../src/clef/protocol.mjs';
import {buildChoiceInputs} from '../src/clef/graph-inputs.mjs';
import {approximateSoftmax} from '../src/clef/softmax.mjs';

const tokenizer = {encode: (text) => ({ids: Array.from({length: Math.ceil(text.length / 16)}, () => 1)})};
const request = (options) => ({state: 'state', questions: [{instructions: 'choose', options}]});

test('lexical option ID order preserves input answer order and original-order ties', () => {
  const options = Array.from({length: 11}, (_, i) => `option ${i}`);
  const packed = packChoiceRequest(tokenizer, request(options), [2]);
  assert.deepEqual(packed.questions[0].option_ids.slice(0, 4), ['0', '1', '10', '2']);
  const scores = new Float32Array(11);
  scores[2] = scores[3] = 5;
  const answer = unpackChoiceScores(packed, [scores], approximateSoftmax([scores])).answers[0];
  assert.equal(answer.selectedIndex, 2);
  assert.deepEqual(
    answer.probabilities.map((item) => item.option),
    options
  );
  assert.equal(answer.probabilities[10].logit, 5);
  assert.equal(answer.probabilities[2].logit, 5);
});

test('4 and 512 tokens and nonoverlapping option spans pass; 513 and invalid shapes fail', () => {
  const packed = {
    ids: [1, 1, 1, 1],
    maxInputTokens: 512,
    questions: [
      {
        question_id: 'q0',
        question_type: 1,
        question_span: [0, 1],
        option_spans: [
          [1, 2],
          [2, 3]
        ],
        option_ids: ['0', '1'],
        options: ['a', 'b']
      }
    ]
  };
  validatePacked(packed);
  const graph = buildChoiceInputs(packed);
  assert.deepEqual([...graph.inputs[4].data], [1]);
  assert.deepEqual([...graph.inputs[1].data], [0, 1, 2, 3]);
  assert.deepEqual([...graph.inputs[7].data], [0, 1, 0, 0, 0, 0, 1, 0]);
  const long = structuredClone(packed);
  long.ids = Array(512).fill(1);
  validatePacked(long);
  assert.equal(buildChoiceInputs(long).inputs[2].data.length, 512 * 512);
  long.ids.push(1);
  assert.throws(() => validatePacked(long));
  const overlap = structuredClone(packed);
  overlap.questions[0].option_spans[1] = [1, 2];
  assert.throws(() => validatePacked(overlap));
});

test('the text packer accepts exactly 512 tokens and rejects 513', () => {
  const characterTokenizer = {encode: (text) => ({ids: Array(text.length).fill(1)})};
  const base = packChoiceRequest(
    characterTokenizer,
    {state: '', questions: [{instructions: 'q', options: ['a', 'b']}]},
    [2]
  );
  const stateLength = 512 - base.ids.length;
  assert.ok(stateLength > 0);

  const exact = packChoiceRequest(
    characterTokenizer,
    {state: 's'.repeat(stateLength), questions: [{instructions: 'q', options: ['a', 'b']}]},
    [2]
  );
  assert.equal(exact.ids.length, 512);
  assert.throws(
    () =>
      packChoiceRequest(
        characterTokenizer,
        {state: 's'.repeat(stateLength + 1), questions: [{instructions: 'q', options: ['a', 'b']}]},
        [2]
      ),
    /token limit/
  );
});

test('joint count limits and unsupported user data reject without truncation', () => {
  const options = ['a', 'b'];
  assert.throws(() => packChoiceRequest(tokenizer, {...request(options), images: []}, [2]));
  assert.throws(() => packChoiceRequest(tokenizer, request(['same', 'same']), [2]));
  assert.throws(() =>
    packChoiceRequest(tokenizer, {...request(options), questions: Array(9).fill({instructions: 'q', options})}, [2])
  );
  const tooMany = Array.from({length: 33}, (_, i) => `choice ${i}`);
  assert.throws(() =>
    packChoiceRequest(
      tokenizer,
      {...request(options), questions: Array(2).fill({instructions: 'q', options: tooMany})},
      [2]
    )
  );
  assert.throws(
    () => packChoiceRequest(tokenizer, {...request(options), state: 'x'.repeat(10000)}, [2]),
    /token limit/
  );
});
