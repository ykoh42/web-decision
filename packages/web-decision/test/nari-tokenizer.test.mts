import assert from 'node:assert/strict';
import test from 'node:test';
import {createNariTokenizer} from '../src/nari/tokenizer.mjs';

const names = Array.from({length: 33}, (_, i) => `token-${i}`);
names[1] = '<|im_start|>';
names[2] = '<|im_end|>';
names[24] = '<think>';
names[25] = '</think>';
const data = () => ({added_tokens: names.map((content, i) => ({id: 248044 + i, content}))});
const bytes = (value: unknown) => new TextEncoder().encode(JSON.stringify(value));
const config = bytes({tokenizer_class: 'Qwen3_5Tokenizer'});
class Tokenizer {
  declare tokens: ReturnType<typeof data>['added_tokens'];

  constructor(value: ReturnType<typeof data>) {
    this.tokens = value.added_tokens;
  }
  encode(text: string) {
    const token = this.tokens.find((entry) => text.includes(entry.content));
    return {ids: token ? [token.id] : []};
  }
}

test('binds all 33 control IDs and verifies prompt marker encodings', () => {
  const result = createNariTokenizer(Tokenizer, bytes(data()), config);
  assert.equal(result.controlIds.length, 33);
  assert.deepEqual(result.tokenizer.encode('<|im_start|>').ids, [248045]);
});

test('rejects wrong tokenizer class, duplicate IDs and marker encoding', () => {
  assert.throws(() => createNariTokenizer(Tokenizer, bytes(data()), bytes({tokenizer_class: 'other'})), {
    code: 'INVALID_ASSET'
  });
  const duplicate = data();
  duplicate.added_tokens[0]!.id = duplicate.added_tokens[1]!.id;
  assert.throws(() => createNariTokenizer(Tokenizer, bytes(duplicate), config), {code: 'INVALID_ASSET'});
  class Broken extends Tokenizer {
    override encode() {
      return {ids: [0]};
    }
  }
  assert.throws(() => createNariTokenizer(Broken, bytes(data()), config), {code: 'INVALID_ASSET'});
});
