import test from 'node:test';
import assert from 'node:assert/strict';
import {createClefTokenizer} from '../src/clef/tokenizer.mjs';

const data = () => ({
  normalizer: {type: 'NFC'},
  added_tokens: Array.from({length: 33}, (_, i) => ({id: i + 10, content: `<${i}>`}))
});
const config = {tokenizer_class: 'Qwen2Tokenizer'};
class FakeTokenizer {
  declare items: ReturnType<typeof data>['added_tokens'];

  constructor(value: ReturnType<typeof data>) {
    this.items = value.added_tokens;
  }
  encode(text: string) {
    return {ids: [this.items.find((item) => item.content === text)?.id]};
  }
}

test('control IDs are checked against the actual tokenizer encoding', () => {
  const value = data(),
    result = createClefTokenizer(FakeTokenizer, value, config);
  assert.deepEqual(
    result.controlIds,
    value.added_tokens.map((item) => item.id)
  );
  class WrongTokenizer extends FakeTokenizer {
    override encode() {
      return {ids: [0, 1]};
    }
  }
  assert.throws(() => createClefTokenizer(WrongTokenizer, value, config), {code: 'INVALID_ASSET'});
});

test('wrong tokenizer class, normalization, duplicate and invalid control tokens fail', () => {
  for (const mutate of [
    (v: ReturnType<typeof data>) => {
      v.normalizer.type = 'NFKC';
    },
    (v: ReturnType<typeof data>) => {
      v.added_tokens.pop();
    },
    (v: ReturnType<typeof data>) => {
      v.added_tokens[1]!.id = v.added_tokens[0]!.id;
    },
    (v: ReturnType<typeof data>) => {
      v.added_tokens[1]!.content = v.added_tokens[0]!.content;
    },
    (v: ReturnType<typeof data>) => {
      v.added_tokens[0]!.id = 248320;
    }
  ]) {
    const value = data();
    mutate(value);
    assert.throws(() => createClefTokenizer(FakeTokenizer, value, config), {code: 'INVALID_ASSET'});
  }
  assert.throws(() => createClefTokenizer(FakeTokenizer, data(), {tokenizer_class: 'Other'}), {code: 'INVALID_ASSET'});
});
