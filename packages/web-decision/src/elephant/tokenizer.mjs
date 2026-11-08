import {sha256} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {verifySlots} from './protocol.mjs';

const SLOTS_SHA256 = '35c8f2e0a581bd39341f0e58a6d0aee317e27436daf052d614d777b57b4e374e';

export async function createElephantTokenizer(Tokenizer, data, config) {
  if (
    typeof Tokenizer !== 'function' ||
    config?.tokenizer_class !== 'Qwen3_5Tokenizer' ||
    data?.normalizer?.type !== 'NFC' ||
    !Array.isArray(data.added_tokens)
  )
    throw new WebDecisionError('INVALID_ASSET', 'Invalid Elephant tokenizer.');
  const addedTokens = data.added_tokens.map((item) => item?.content);
  if (
    addedTokens.some((token) => typeof token !== 'string' || !token) ||
    new Set(addedTokens).size !== addedTokens.length
  )
    throw new WebDecisionError('INVALID_ASSET', 'Invalid Elephant control tokens.');
  let tokenizer, slots;
  try {
    tokenizer = new Tokenizer(data, config);
    slots = verifySlots(tokenizer);
  } catch (cause) {
    throw new WebDecisionError('INVALID_ASSET', 'Invalid Elephant answer slots.', {cause});
  }
  if ((await sha256(new TextEncoder().encode(JSON.stringify(slots)))) !== SLOTS_SHA256)
    throw new WebDecisionError('INVALID_ASSET', 'Elephant answer slots do not match the checkpoint.');
  return {tokenizer, addedTokens};
}
