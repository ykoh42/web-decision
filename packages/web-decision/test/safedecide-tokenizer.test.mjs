import test from 'node:test';
import assert from 'node:assert/strict';
import {createSafeDecideTokenizer} from '../src/safedecide/tokenizer.mjs';

const data = {
  model: {type: 'BPE', vocab: Object.fromEntries(Array.from({length: 8}, (_, index) => [`token-${index}`, index]))},
  truncation: {direction: 'Right', max_length: 640, strategy: 'LongestFirst'},
  post_processor: {type: 'TemplateProcessing'}
};
const config = {
  bos_token: '<bos>',
  eos_token: '<eos>',
  pad_token: '<pad>',
  unk_token: '<unk>',
  mask_token: '<mask>',
  padding_side: 'right'
};
const modelConfig = {
  vocabSize: 8,
  specialTokenIds: {bosId: 2, eosId: 1, padId: 0, maskId: 4}
};

class FakeTokenizer {
  token_to_id(token) {
    return (
      {'<pad>': 0, '<eos>': 1, '<bos>': 2, '<unk>': 3, '<mask>': 4}[token] ??
      {a: 5, b: 6, c: 7, x: 5, y: 6, z: 7}[token]
    );
  }

  tokenize(text) {
    return text ? text.split(' ') : [];
  }

  get post_processor() {
    return {
      post_process(first, second) {
        return {tokens: ['<bos>', ...first, '<eos>', ...second, '<eos>']};
      }
    };
  }
}

test('SafeDecide tokenizer adds the checkpoint pair template and maps special token IDs', () => {
  const tokenizer = createSafeDecideTokenizer(FakeTokenizer, data, config, modelConfig, 8);
  assert.deepEqual(tokenizer.encodePair('a b', 'x'), {inputIds: [2, 5, 6, 1, 5, 1], truncated: false});
  assert.deepEqual(tokenizer.special, {bosId: 2, eosId: 1, padId: 0, maskId: 4});
});

test('SafeDecide truncates the longer pair from the right while reserving pair special tokens', () => {
  const tokenizer = createSafeDecideTokenizer(FakeTokenizer, data, config, modelConfig, 6);
  assert.deepEqual(tokenizer.encodePair('a b c', 'x'), {inputIds: [2, 5, 6, 1, 5, 1], truncated: true});
  assert.deepEqual(tokenizer.encodePair('a b c', 'x y z'), {inputIds: [2, 5, 6, 1, 5, 1], truncated: true});
});

test('SafeDecide tokenizer rejects unexpected vocabulary and special-token mappings', () => {
  const wrong = {...config, pad_token: '<missing>'};
  assert.throws(() => createSafeDecideTokenizer(FakeTokenizer, data, wrong, modelConfig, 8), {code: 'INVALID_ASSET'});
  assert.throws(
    () => createSafeDecideTokenizer(FakeTokenizer, {...data, truncation: {max_length: 384}}, config, modelConfig, 8),
    {code: 'INVALID_ASSET'}
  );
});
