import {WebDecisionError} from '../kev.js';
import type {ClefTokenizer} from './types.mjs';

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function createClefTokenizer(
  Tokenizer: unknown,
  data: unknown,
  config: unknown
): {
  tokenizer: ClefTokenizer;
  controlIds: number[];
} {
  const tokenizerData = isRecord(data) ? data : null;
  const tokenizerConfig = isRecord(config) ? config : null;
  const addedTokens = tokenizerData?.added_tokens;
  if (
    typeof Tokenizer !== 'function' ||
    tokenizerConfig?.tokenizer_class !== 'Qwen2Tokenizer' ||
    !isRecord(tokenizerData?.normalizer) ||
    tokenizerData.normalizer.type !== 'NFC' ||
    !Array.isArray(addedTokens) ||
    addedTokens.length !== 33
  )
    throw new WebDecisionError('INVALID_ASSET', 'Invalid Clef tokenizer.');
  const ids = addedTokens.map((token) => (isRecord(token) ? token.id : undefined));
  const texts = addedTokens.map((token) => (isRecord(token) ? token.content : undefined));
  if (
    ids.some((id) => !Number.isSafeInteger(id) || (id as number) < 0 || (id as number) >= 248320) ||
    new Set(ids).size !== ids.length ||
    texts.some((text) => typeof text !== 'string' || !text) ||
    new Set(texts).size !== texts.length
  )
    throw new WebDecisionError('INVALID_ASSET', 'Invalid Clef control tokens.');
  try {
    const tokenizer = Reflect.construct(Tokenizer, [data, config]) as ClefTokenizer;
    const controlIds = ids as number[];
    const controlTexts = texts as string[];
    for (const [index, text] of controlTexts.entries()) {
      const encoded = tokenizer.encode(text, {add_special_tokens: false})?.ids;
      if (!encoded || encoded.length !== 1 || encoded[0] !== controlIds[index])
        throw new Error('Control token mismatch');
    }
    return {tokenizer, controlIds};
  } catch (cause) {
    throw new WebDecisionError('INVALID_ASSET', 'Could not initialize the Clef tokenizer.', {cause});
  }
}
