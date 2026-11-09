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

interface D1OmniTokenizerInstance {
  token_to_id(token: string): unknown;
  encode(text: string, options: {readonly add_special_tokens: false}): unknown;
}

interface D1OmniTokenizerConstructor {
  new (data: unknown, config: unknown): D1OmniTokenizerInstance;
}

export interface D1OmniTokenizer {
  readonly ids: Readonly<Record<string, number>>;
  encode(text: string): number[];
}

function invalidAsset(message: string): WebDecisionError {
  return new WebDecisionError('INVALID_ASSET', message);
}

function checkedIds(encoding: unknown): number[] {
  const ids = (encoding as {readonly ids?: unknown} | null | undefined)?.ids;
  if (!Array.isArray(ids) && !ArrayBuffer.isView(ids))
    throw invalidAsset('d1-omni tokenizer returned invalid token IDs.');
  const result = Array.from(ids as unknown as ArrayLike<number>);
  if (result.some((id) => !Number.isSafeInteger(id) || id < 0))
    throw invalidAsset('d1-omni tokenizer returned invalid token IDs.');
  return result;
}

export function createD1OmniTokenizer(Tokenizer: unknown, data: unknown, config: unknown): D1OmniTokenizer {
  if (
    typeof Tokenizer !== 'function' ||
    !(data as {readonly model?: {readonly vocab?: unknown} | null} | null | undefined)?.model?.vocab ||
    (config as {readonly bos_token?: unknown} | null | undefined)?.bos_token !== '<|startoftext|>' ||
    (config as {readonly pad_token?: unknown} | null | undefined)?.pad_token !== '<|pad|>' ||
    (config as {readonly mask_token?: unknown} | null | undefined)?.mask_token !== '<|mask|>' ||
    (config as {readonly model_max_length?: unknown} | null | undefined)?.model_max_length !== D1_OMNI_MODEL_MAX_TOKENS
  )
    throw invalidAsset('d1-omni tokenizer files do not match the pinned checkpoint.');

  const tokenizer = new (Tokenizer as D1OmniTokenizerConstructor)(data, config);
  const ids = Object.fromEntries(
    Object.entries(RESERVED_IDS).map(([token, expected]) => {
      const actual = tokenizer.token_to_id(token);
      if (actual !== expected) throw invalidAsset(`d1-omni tokenizer has an unexpected ID for ${token}.`);
      return [token, actual];
    })
  ) as Record<string, number>;

  return Object.freeze({
    ids: Object.freeze(ids),
    encode(text: string) {
      if (typeof text !== 'string') throw new TypeError('d1-omni tokenizer input must be a string.');
      return checkedIds(tokenizer.encode(text, {add_special_tokens: false}));
    }
  });
}
