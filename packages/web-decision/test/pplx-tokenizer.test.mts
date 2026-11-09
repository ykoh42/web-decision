import assert from 'node:assert/strict';
import test from 'node:test';
import {createFlashTokenizer} from '../src/pplx/tokenizer.mjs';

const markers = new Map([
  ['<|im_start|>', 248045],
  ['<|im_end|>', 248046],
  ['<|fim_suffix|>', 248062]
]);
const special = [
  '<|im_start|>',
  '<|im_end|>',
  '<|fim_suffix|>',
  '<|endoftext|>',
  '<|audio_start|>',
  '<|audio_end|>',
  '<|audio_pad|>',
  '<|image_pad|>',
  '<|video_pad|>',
  '<|vision_start|>',
  '<|vision_end|>'
];

function data() {
  return {
    added_tokens: Array.from({length: 33}, (_, index) => {
      const content = special[index] ?? `token-${index}`;
      return {content, id: markers.get(content) ?? index};
    })
  };
}

class Tokenizer {
  encode(text: string): {ids: unknown} {
    for (const [marker, id] of markers) {
      if (text === marker) return {ids: [id]};
      if (text === `x${marker}y`) return {ids: [1, id, 2]};
    }
    return {ids: [1, 2]};
  }
}

test('accepts a tokenizer with the pinned marker behavior and snapshots output IDs', () => {
  const result = createFlashTokenizer(Tokenizer, data(), {});
  assert.deepEqual(result.tokenizer.encode('hello').ids, [1, 2]);
  assert.equal(result.config.codes.length, 255);
  assert.equal(result.config.markerId, 248062);
  assert.equal(result.config.maxInputTokens, 512);
});

test('rejects duplicate, missing or incorrectly encoded control tokens', () => {
  const duplicate = data();
  duplicate.added_tokens[1]!.id = duplicate.added_tokens[0]!.id;
  const missing = data();
  missing.added_tokens[3]!.content = 'replacement';
  class BrokenTokenizer extends Tokenizer {
    override encode() {
      return {ids: [0]};
    }
  }
  for (const [impl, tokens] of [
    [Tokenizer, duplicate],
    [Tokenizer, missing],
    [BrokenTokenizer, data()]
  ]) {
    assert.throws(() => createFlashTokenizer(impl, tokens, {}), {code: 'INVALID_ASSET'});
  }
});

test('checks runtime token IDs and returns an independent snapshot', () => {
  let output: number[] | null = [5, 6];
  class MutableTokenizer extends Tokenizer {
    override encode(text: string) {
      return text === 'input' ? {ids: output} : super.encode(text);
    }
  }
  const {tokenizer} = createFlashTokenizer(MutableTokenizer, data(), {});
  const result = tokenizer.encode('input');
  output[0] = 7;
  assert.deepEqual(result.ids, [5, 6]);
  for (const ids of [[-1], [248320], [1.5], null]) {
    output = ids;
    assert.throws(() => tokenizer.encode('input'), {code: 'INVALID_ASSET'});
  }
});

test('preserves the cause when the tokenizer throws', () => {
  const cause = new Error('encoding failed');
  class ThrowingTokenizer extends Tokenizer {
    override encode(text: string) {
      if (text === 'input') throw cause;
      return super.encode(text);
    }
  }
  const {tokenizer} = createFlashTokenizer(ThrowingTokenizer, data(), {});
  assert.throws(() => tokenizer.encode('input'), {code: 'INVALID_ASSET', cause});
});
