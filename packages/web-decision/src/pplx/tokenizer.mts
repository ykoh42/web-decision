import {WebDecisionError} from '../kev.js';
import decision from './decision-config.json' with {type: 'json'};
import type {FlashDecisionConfig, FlashTokenizer} from './types.js';

const SPECIAL = Object.freeze([
  '<|im_end|>',
  '<|endoftext|>',
  '<|audio_start|>',
  '<|audio_end|>',
  '<|audio_pad|>',
  '<|image_pad|>',
  '<|video_pad|>',
  '<|vision_start|>',
  '<|vision_end|>',
  '<|fim_suffix|>'
]);
const invalid = () => new WebDecisionError('INVALID_ASSET', 'Perplexity tokenizer does not match the pinned protocol.');

interface FlashRawTokenizer {
  encode(text: string, options?: {add_special_tokens?: boolean}): {ids?: unknown} | null;
}

interface FlashAddedToken {
  content: string;
  id: number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function checkedEncode(
  tokenizer: FlashRawTokenizer,
  text: string,
  options?: {add_special_tokens?: boolean}
): {ids: number[]} {
  let ids;
  try {
    ids = tokenizer.encode(text, options)?.ids;
  } catch (cause) {
    throw new WebDecisionError('INVALID_ASSET', 'Perplexity tokenizer failed.', {cause});
  }
  if (!Array.isArray(ids)) throw invalid();
  const snapshot = Array.from(ids);
  if (snapshot.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= 248320)) throw invalid();
  return {ids: snapshot};
}

export function createFlashTokenizer(
  Tokenizer: unknown,
  data: unknown,
  config: unknown
): {tokenizer: FlashTokenizer; config: FlashDecisionConfig} {
  const dataRecord = isRecord(data) ? data : null;
  const addedTokens = dataRecord?.added_tokens;
  if (
    typeof Tokenizer !== 'function' ||
    !Array.isArray(addedTokens) ||
    addedTokens.length !== 33 ||
    !config ||
    typeof config !== 'object'
  )
    throw invalid();
  const tokens = addedTokens as unknown[];
  if (
    tokens.some((item: unknown) => {
      const token = isRecord(item) ? item : null;
      return (
        typeof token?.content !== 'string' ||
        !token.content ||
        !Number.isSafeInteger(token.id) ||
        (token.id as number) < 0 ||
        (token.id as number) >= 248320
      );
    }) ||
    new Set(tokens.map((item) => (isRecord(item) ? item.content : undefined))).size !== tokens.length ||
    new Set(tokens.map((item) => (isRecord(item) ? item.id : undefined))).size !== tokens.length ||
    !tokens.some((item) => isRecord(item) && item.content === '<|fim_suffix|>' && item.id === 248062) ||
    SPECIAL.some((value) => !tokens.some((item) => isRecord(item) && item.content === value))
  )
    throw invalid();
  const validTokens = tokens as FlashAddedToken[];
  const inner = Reflect.construct(Tokenizer, [data, config]) as FlashRawTokenizer;
  const tokenizer: FlashTokenizer = Object.freeze({
    encode: (text: string, options?: {add_special_tokens?: boolean}) => checkedEncode(inner, text, options)
  });
  for (const [marker, id] of [
    ['<|im_start|>', 248045],
    ['<|im_end|>', 248046],
    ['<|fim_suffix|>', 248062]
  ] as const) {
    const direct = tokenizer.encode(marker, {add_special_tokens: false}).ids;
    const embedded = tokenizer.encode(`x${marker}y`, {add_special_tokens: false}).ids;
    if (direct.length !== 1 || direct[0] !== id || embedded.filter((item) => item === id).length !== 1) throw invalid();
  }
  if (!Array.isArray(decision.codes) || decision.codes.length !== 255 || new Set(decision.codes).size !== 255)
    throw invalid();
  return {
    tokenizer,
    config: {
      codes: [...decision.codes],
      markerId: 248062,
      escapedTokens: [...SPECIAL],
      addedTokens: validTokens.map((item) => item.content),
      addedTokenIds: validTokens.map((item) => item.id),
      maxInputTokens: 512
    }
  };
}
