import {WebDecisionError} from '../kev.js';
import {D1_OMNI_MODEL_MAX_TOKENS} from './contract.mjs';

const RESERVED_IDS = Object.freeze({
  '<|startoftext|>': 1,
  '<|pad|>': 0,
  '<|mask|>': 16,
  '<|reserved_7|>': 17,
  '<|reserved_8|>': 18,
  '<|reserved_9|>': 19,
  '<|reserved_10|>': 20,
  '<|reserved_11|>': 21
});

function invalidAsset(message) {
  return new WebDecisionError('INVALID_ASSET', message);
}

function checkedIds(encoding) {
  const ids = encoding?.ids;
  if (!Array.isArray(ids) && !ArrayBuffer.isView(ids))
    throw invalidAsset('d1-omni tokenizer returned invalid token IDs.');
  const result = Array.from(ids);
  if (result.some((id) => !Number.isSafeInteger(id) || id < 0))
    throw invalidAsset('d1-omni tokenizer returned invalid token IDs.');
  return result;
}

export function createD1OmniTokenizer(Tokenizer, data, config) {
  if (
    typeof Tokenizer !== 'function' ||
    !data?.model?.vocab ||
    config?.bos_token !== '<|startoftext|>' ||
    config?.pad_token !== '<|pad|>' ||
    config?.mask_token !== '<|mask|>' ||
    config?.model_max_length !== D1_OMNI_MODEL_MAX_TOKENS
  )
    throw invalidAsset('d1-omni tokenizer files do not match the pinned checkpoint.');

  const tokenizer = new Tokenizer(data, config);
  const ids = Object.fromEntries(
    Object.entries(RESERVED_IDS).map(([token, expected]) => {
      const actual = tokenizer.token_to_id(token);
      if (actual !== expected) throw invalidAsset(`d1-omni tokenizer has an unexpected ID for ${token}.`);
      return [token, actual];
    })
  );

  return Object.freeze({
    ids: Object.freeze(ids),
    encode(text) {
      if (typeof text !== 'string') throw new TypeError('d1-omni tokenizer input must be a string.');
      return checkedIds(tokenizer.encode(text, {add_special_tokens: false}));
    }
  });
}
