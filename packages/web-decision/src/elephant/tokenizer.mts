import {sha256} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {verifySlots} from './protocol.mjs';
import type {ElephantSlotTokenizer} from './protocol.mjs';

const SLOTS_SHA256 = '35c8f2e0a581bd39341f0e58a6d0aee317e27436daf052d614d777b57b4e374e';

interface ElephantTokenizerConstructor {
  new (data: unknown, config: unknown): ElephantSlotTokenizer;
}

export async function createElephantTokenizer(
  Tokenizer: unknown,
  data: unknown,
  config: unknown
): Promise<{readonly tokenizer: ElephantSlotTokenizer; readonly addedTokens: string[]}> {
  if (
    typeof Tokenizer !== 'function' ||
    (config as {readonly tokenizer_class?: unknown} | null | undefined)?.tokenizer_class !== 'Qwen3_5Tokenizer' ||
    (data as {readonly normalizer?: {readonly type?: unknown} | null} | null | undefined)?.normalizer?.type !== 'NFC' ||
    !Array.isArray((data as {readonly added_tokens?: unknown}).added_tokens)
  )
    throw new WebDecisionError('INVALID_ASSET', 'Invalid Elephant tokenizer.');
  const addedTokens = (data as {readonly added_tokens: unknown[]}).added_tokens.map(
    (item) => (item as {readonly content?: unknown} | null | undefined)?.content
  );
  if (
    addedTokens.some((token) => typeof token !== 'string' || !token) ||
    new Set(addedTokens).size !== addedTokens.length
  )
    throw new WebDecisionError('INVALID_ASSET', 'Invalid Elephant control tokens.');
  let tokenizer: ElephantSlotTokenizer, slots: number[];
  try {
    tokenizer = new (Tokenizer as ElephantTokenizerConstructor)(data, config);
    slots = verifySlots(tokenizer);
  } catch (cause) {
    throw new WebDecisionError('INVALID_ASSET', 'Invalid Elephant answer slots.', {cause});
  }
  if ((await sha256(new TextEncoder().encode(JSON.stringify(slots)))) !== SLOTS_SHA256)
    throw new WebDecisionError('INVALID_ASSET', 'Elephant answer slots do not match the checkpoint.');
  return {tokenizer, addedTokens: addedTokens as string[]};
}
