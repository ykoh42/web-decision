import assert from 'node:assert/strict';
import test from 'node:test';
import {createGemma4Session} from '../src/strands/gemma4-e2b-session.mjs';

function fixture() {
  const rows: {type: string; options: string[]}[] = [];
  const tokenizer = {
    encode(text: string) {
      return {ids: Array.from(text, (character) => character.codePointAt(0)!)};
    },
    decode(ids: number[]) {
      return String.fromCodePoint(...ids);
    }
  };
  const runner = {
    async run(row: {type: string; options: string[]}) {
      rows.push(row);
      const logits = new Float32Array(row.options.length);
      logits[logits.length - 1] = 1;
      return logits;
    },
    async releaseCachedMemory() {},
    async dispose() {}
  };
  return {session: createGemma4Session({tokenizer, runner}), rows};
}

test('Gemma 4 session preserves typed choice order and the native score and Noul outputs', async () => {
  const {session, rows} = fixture();
  const result = await session.systemOne('A small state.', {
    decision: {type: 'choice', instructions: 'Choose one.', criteria: ['first', 'second']},
    rating: {type: 'score', instructions: 'Rate it.', criteria: ['low', 'middle', 'high']},
    claim: {type: 'noul', instructions: 'Is this true?'}
  });

  assert.equal(result.model, 'experimental-strands-gemma4-e2b-fp16');
  assert.equal(result.answers.decision.choice, 'second');
  assert.deepEqual(Object.keys(result.answers.decision.probabilities), ['first', 'second']);
  assert.equal(result.answers.rating.type, 'score');
  assert.equal(result.answers.claim.type, 'noul');
  assert.deepEqual(
    rows.map((row) => row.type),
    ['choice', 'score', 'noul']
  );
});

test('Gemma 4 generic choice accepts 260 options and returns logits in caller order', async () => {
  const {session, rows} = fixture();
  const options = Array.from({length: 260}, (_, index) => `o${index.toString(36)}`);
  const result = await session.decide({
    state: 'A small state.',
    questions: [{instructions: 'Choose one.', options}]
  });

  assert.equal(rows[0]!.options.length, 260);
  assert.equal(result.answers[0].selected, options.at(-1));
  assert.equal(result.answers[0].selectedIndex, 259);
  assert.deepEqual(
    result.answers[0].probabilities.map(({option}: {option: string}) => option),
    options
  );
  assert.equal(result.answers[0].probabilities.at(-1).logit, 1);
});

test('Gemma 4 session rejects rows over the pinned context without token truncation', async () => {
  const {session, rows} = fixture();
  await assert.rejects(
    session.systemOne('x'.repeat(4096), {
      choice: {type: 'choice', instructions: '', criteria: ['a', 'b']}
    }),
    {code: 'INVALID_INPUT'}
  );
  assert.equal(rows.length, 0);
  await session.dispose();
});
