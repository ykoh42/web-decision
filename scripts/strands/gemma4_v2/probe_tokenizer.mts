import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {resolve} from 'node:path';

const require = createRequire(resolve('packages/web-decision/package.json'));
const {Tokenizer} = require('@huggingface/tokenizers') as {
  Tokenizer: new (
    model: unknown,
    config: unknown
  ) => {
    encode(text: string, options: {add_special_tokens: boolean}): {ids: number[]};
  };
};
const modelDir = resolve('../strands-e2b-v2');
const model = JSON.parse(await readFile(resolve(modelDir, 'tokenizer.json'), 'utf8'));
const config = JSON.parse(await readFile(resolve(modelDir, 'tokenizer_config.json'), 'utf8'));
const tokenizer = new Tokenizer(
  {...model, added_tokens: model.added_tokens.filter((token: {special: boolean}) => !token.special)},
  config
);
const fixtures = JSON.parse(await readFile('../e2b-v2-prompt-reference.json', 'utf8')) as Array<{
  state_text: string;
  state_ids: number[];
  questions: Record<string, {question_text: string; question_ids: number[]}>;
}>;
const encode = (text: string) => tokenizer.encode(text, {add_special_tokens: false}).ids;
for (const fixture of fixtures) {
  const prefix = '<bos><|turn>';
  assert.ok(fixture.state_text.startsWith(prefix));
  assert.deepEqual([2, 105, ...encode(fixture.state_text.slice(prefix.length))], fixture.state_ids);
  for (const question of Object.values(fixture.questions)) {
    const suffix = '<turn|>\n<|turn>model\n';
    assert.ok(question.question_text.endsWith(suffix));
    const text = question.question_text.slice(0, -suffix.length);
    assert.deepEqual([...encode(text), 106, ...encode('\n'), 105, ...encode('model\n')], question.question_ids);
  }
}
console.log(`Official token IDs matched for ${fixtures.length} text fixtures`);
