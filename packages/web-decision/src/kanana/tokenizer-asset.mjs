import {Tokenizer} from '@huggingface/tokenizers';
import {WebDecisionError} from '../kev.js';
import {KANANA_MODEL_MAX_LENGTH, KANANA_VOCAB_SIZE} from './contract.mjs';

const SPECIAL_TOKEN_IDS = Object.freeze({
  '<|begin_of_text|>': 128000,
  '<|end_of_text|>': 128001,
  '<|im_start|>': 128009,
  '<|im_end|>': 128010,
  '<|vision_start|>': 128017,
  '<|vision_end|>': 128018,
  '<|image_pad|>': 128020,
  '<|video_pad|>': 128021,
  '<think>': 128032,
  '</think>': 128033
});

function invalidAsset(message) {
  return new WebDecisionError('INVALID_ASSET', message);
}

function checkedIds(encoding) {
  const ids = encoding?.ids;
  if ((!Array.isArray(ids) && !ArrayBuffer.isView(ids)) || ids instanceof DataView)
    throw invalidAsset('Kanana tokenizer returned invalid token IDs.');
  const result = Array.from(ids);
  if (result.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= KANANA_VOCAB_SIZE))
    throw invalidAsset('Kanana tokenizer returned an out-of-range token ID.');
  return result;
}

function validateMetadata(tokenizerJson, tokenizerConfig, modelConfig) {
  if (
    tokenizerJson?.model?.type !== 'BPE' ||
    tokenizerConfig?.backend !== 'tokenizers' ||
    tokenizerConfig?.tokenizer_class !== 'TokenizersBackend' ||
    tokenizerConfig?.bos_token !== '<|begin_of_text|>' ||
    tokenizerConfig?.eos_token !== '<|im_end|>' ||
    tokenizerConfig?.pad_token !== '<|end_of_text|>' ||
    tokenizerConfig?.padding_side !== 'left' ||
    tokenizerConfig?.model_max_length !== KANANA_MODEL_MAX_LENGTH ||
    modelConfig?.model_type !== 'kanana_v_decide' ||
    modelConfig?.max_length !== KANANA_MODEL_MAX_LENGTH ||
    modelConfig?.image_token_id !== SPECIAL_TOKEN_IDS['<|image_pad|>'] ||
    modelConfig?.text_config?.vocab_size !== KANANA_VOCAB_SIZE
  )
    throw invalidAsset('Kanana tokenizer and configuration do not match the pinned checkpoint.');
}

/** Construct the pinned tokenizer with the same defaults used by the Python tokenizer. */
export function createKananaTokenizer(tokenizerJson, tokenizerConfig, modelConfig, TokenizerClass = Tokenizer) {
  validateMetadata(tokenizerJson, tokenizerConfig, modelConfig);
  if (typeof TokenizerClass !== 'function') throw invalidAsset('Kanana tokenizer runtime is unavailable.');

  const tokenizer = new TokenizerClass(tokenizerJson, tokenizerConfig);
  if (
    typeof tokenizer?.get_vocab !== 'function' ||
    typeof tokenizer?.token_to_id !== 'function' ||
    typeof tokenizer?.encode !== 'function' ||
    typeof tokenizer?.decode !== 'function'
  )
    throw invalidAsset('Kanana tokenizer runtime is incomplete.');
  if (tokenizer.get_vocab(true).size !== KANANA_VOCAB_SIZE)
    throw invalidAsset('Kanana tokenizer vocabulary size does not match the pinned checkpoint.');

  const ids = Object.fromEntries(
    Object.entries(SPECIAL_TOKEN_IDS).map(([token, expected]) => {
      const actual = tokenizer.token_to_id(token);
      if (actual !== expected) throw invalidAsset(`Kanana tokenizer has an unexpected ID for ${token}.`);
      return [token, actual];
    })
  );

  return Object.freeze({
    ids: Object.freeze(ids),
    encode(text, options) {
      if (typeof text !== 'string') throw new TypeError('Kanana tokenizer input must be a string.');
      const encoded = tokenizer.encode(text, options);
      checkedIds(encoded);
      return encoded;
    },
    decode(tokenIds, options) {
      if ((!Array.isArray(tokenIds) && !ArrayBuffer.isView(tokenIds)) || tokenIds instanceof DataView)
        throw new TypeError('Kanana tokenizer IDs must be an array.');
      const idsToDecode = Array.from(tokenIds);
      if (idsToDecode.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= KANANA_VOCAB_SIZE))
        throw new TypeError('Kanana tokenizer IDs are out of range.');
      const decoded = tokenizer.decode(idsToDecode, options);
      if (typeof decoded !== 'string') throw invalidAsset('Kanana tokenizer returned invalid decoded text.');
      return decoded;
    }
  });
}
