import {WebDecisionError} from '../kev.js';

const invalidAsset = (message) => new WebDecisionError('INVALID_ASSET', message);
const WORD_PATTERN =
  /(?:https?:\/\/[^\s]+|www\.[^\s]+)|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|@[a-z0-9_]+|[\p{L}\p{N}_]+(?:[-_][\p{L}\p{N}_]+)*|\S/giu;
const SPECIAL_TOKENS = ['[SEP_STRUCT]', '[SEP_TEXT]', '[P]', '[L]', '[DESCRIPTION]', '[PAD]'];

function tokenIds(encoding, vocabularySize) {
  const ids = encoding?.ids;
  if (!Array.isArray(ids) && !ArrayBuffer.isView(ids))
    throw invalidAsset('GLiNER tokenizer returned invalid token IDs.');
  const result = Array.from(ids);
  if (result.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= vocabularySize))
    throw invalidAsset('GLiNER tokenizer returned invalid token IDs.');
  return result;
}

function specialIds(tokenizer) {
  const ids = Object.fromEntries(
    SPECIAL_TOKENS.map((token) => {
      const id = tokenizer.token_to_id(token);
      if (!Number.isSafeInteger(id) || id < 0) throw invalidAsset(`GLiNER tokenizer is missing ${token}.`);
      return [token, id];
    })
  );
  if (
    ids['[PAD]'] !== 0 ||
    ids['[SEP_STRUCT]'] !== 128001 ||
    ids['[SEP_TEXT]'] !== 128002 ||
    ids['[P]'] !== 128003 ||
    ids['[L]'] !== 128007 ||
    ids['[DESCRIPTION]'] !== 128010
  )
    throw invalidAsset('GLiNER special token IDs do not match the pinned model.');
  return Object.freeze(ids);
}

export function createGlinerMobileTokenizer(Tokenizer, data, tokenizerConfig, specialTokenMap, modelConfig) {
  if (
    typeof Tokenizer !== 'function' ||
    data?.model?.type !== 'Unigram' ||
    tokenizerConfig?.tokenizer_class !== 'DebertaV2TokenizerFast' ||
    tokenizerConfig?.backend !== 'tokenizers' ||
    tokenizerConfig?.pad_token !== '[PAD]' ||
    specialTokenMap?.pad_token?.content !== '[PAD]' ||
    modelConfig?.vocab_size !== 128011 ||
    modelConfig?.gliner2?.max_len !== 512 ||
    !Array.isArray(modelConfig.gliner2.special_tokens) ||
    !SPECIAL_TOKENS.slice(0, -1).every((token) => modelConfig.gliner2.special_tokens.includes(token))
  )
    throw invalidAsset('GLiNER tokenizer files do not match the pinned model.');

  const tokenizer = new Tokenizer(data, tokenizerConfig);
  const special = specialIds(tokenizer);
  const vocabularySize = modelConfig.vocab_size;

  return Object.freeze({
    special,
    encode(text) {
      return tokenIds(tokenizer.encode(text, {add_special_tokens: false}), vocabularySize);
    },
    encodeState(text) {
      const trimmed = text.trim();
      const normalized = /[.!?]$/.test(trimmed) ? trimmed : `${trimmed}.`;
      const words = normalized.match(WORD_PATTERN) ?? [];
      return words.flatMap((word) =>
        tokenIds(tokenizer.encode(word.toLowerCase(), {add_special_tokens: false}), vocabularySize)
      );
    }
  });
}
