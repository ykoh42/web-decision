import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {resolve} from 'node:path';
import {renderStrandsV2} from '../../../packages/web-decision/src/strands/gemma4-v2-protocol.mts';
const require = createRequire(resolve('packages/web-decision/package.json'));
const {Tokenizer} = require('@huggingface/tokenizers');
const modelDir = resolve('../strands-e2b-v2');
const model = JSON.parse(await readFile(resolve(modelDir, 'tokenizer.json'), 'utf8'));
const config = JSON.parse(await readFile(resolve(modelDir, 'tokenizer_config.json'), 'utf8'));
const tokenizer = new Tokenizer(
  {...model, added_tokens: model.added_tokens.filter((x: {special: boolean}) => !x.special)},
  config
);
const fixtures = [
  ...JSON.parse(await readFile('../e2b-v2-prompt-reference.json', 'utf8')),
  ...JSON.parse(await readFile('../e2b-v2-protocol-boundary-reference.json', 'utf8'))
];
for (const fixture of fixtures) {
  const imageTokens = (fixture.images ?? []).map((span: {start: number; end: number}) => span.end - span.start);
  const actual = renderStrandsV2(fixture.request, tokenizer, imageTokens);
  assert.deepEqual(actual.images, fixture.images ?? []);
  assert.deepEqual(actual.stateIds, fixture.state_ids);
  assert.equal(actual.inputTokens, fixture.input_tokens);
  for (const q of actual.questions) {
    const expected = fixture.questions[q.name];
    assert.deepEqual(q.options, expected.options);
    assert.deepEqual(q.codes, expected.codes);
    assert.equal(q.questionText, expected.question_text);
    assert.deepEqual(q.questionIds, expected.question_ids);
  }
}
const vision = JSON.parse(await readFile('../e2b-v2-vision-reference/tokens.json', 'utf8'));
const imageRequest = {
  state: 'The image contains two color regions.',
  questions: {
    color: {type: 'choice' as const, instructions: 'Which color is on the left?', criteria: {red: 'red', blue: 'blue'}}
  }
};
const imageResult = renderStrandsV2(imageRequest, tokenizer, [256]);
assert.deepEqual(imageResult.stateIds, vision.state);
assert.deepEqual(imageResult.images, vision.images);
assert.deepEqual(imageResult.questions[0]!.questionIds, vision.questions[0]);
console.log('Image prompt and placeholder span match the pinned official implementation');
console.log(`Rendered ${fixtures.length} requests identically to the pinned official implementation`);
