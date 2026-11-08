import {WebDecisionError} from '../kev.js';

export function createClefTokenizer(Tokenizer, data, config) {
  if (
    typeof Tokenizer !== 'function' ||
    config?.tokenizer_class !== 'Qwen2Tokenizer' ||
    data?.normalizer?.type !== 'NFC' ||
    !Array.isArray(data.added_tokens) ||
    data.added_tokens.length !== 33
  )
    throw new WebDecisionError('INVALID_ASSET', 'Invalid Clef tokenizer.');
  const ids = data.added_tokens.map((token) => token?.id),
    texts = data.added_tokens.map((token) => token?.content);
  if (
    ids.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= 248320) ||
    new Set(ids).size !== ids.length ||
    texts.some((text) => typeof text !== 'string' || !text) ||
    new Set(texts).size !== texts.length
  )
    throw new WebDecisionError('INVALID_ASSET', 'Invalid Clef control tokens.');
  try {
    const tokenizer = new Tokenizer(data, config);
    for (const [index, text] of texts.entries()) {
      const encoded = tokenizer.encode(text, {add_special_tokens: false})?.ids;
      if (!encoded || encoded.length !== 1 || encoded[0] !== ids[index]) throw new Error('Control token mismatch');
    }
    return {tokenizer, controlIds: ids};
  } catch (cause) {
    throw new WebDecisionError('INVALID_ASSET', 'Could not initialize the Clef tokenizer.', {cause});
  }
}
