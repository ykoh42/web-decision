import {WebDecisionError} from '../kev.js';
import type {Tokenizer as TokenizerImplementation} from '@huggingface/tokenizers';

type GlinerSpecialToken = '[SEP_STRUCT]' | '[SEP_TEXT]' | '[P]' | '[L]' | '[DESCRIPTION]' | '[PAD]';

interface GlinerSpecialIds extends Record<GlinerSpecialToken, number> {}

interface GlinerTokenizer {
  readonly special: Readonly<GlinerSpecialIds>;
  encode(text: string): number[];
  encodeState(text: string): number[];
}

interface GlinerTokenizerData {
  readonly model?: {readonly type?: unknown} | null;
}

interface GlinerTokenizerConfig {
  readonly tokenizer_class?: unknown;
  readonly backend?: unknown;
  readonly pad_token?: unknown;
}

interface GlinerSpecialTokenMap {
  readonly pad_token?: {readonly content?: unknown} | null;
}

interface GlinerModelConfig {
  readonly vocab_size?: unknown;
  readonly gliner2?: {readonly max_len?: unknown; readonly special_tokens?: unknown[]};
}

type GlinerTokenizerConstructor = new (data: object, config: object) => TokenizerImplementation;

const invalidAsset = (message: string): WebDecisionError => new WebDecisionError('INVALID_ASSET', message);
const WORD_PATTERN =
  /(?:https?:\/\/[^\s]+|www\.[^\s]+)|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|@[a-z0-9_]+|[\p{L}\p{N}_]+(?:[-_][\p{L}\p{N}_]+)*|\S/giu;
const SPECIAL_TOKENS = ['[SEP_STRUCT]', '[SEP_TEXT]', '[P]', '[L]', '[DESCRIPTION]', '[PAD]'];

function tokenIds(encoding: unknown, vocabularySize: number): number[] {
  const ids = (encoding as {readonly ids?: unknown} | null | undefined)?.ids;
  if (!Array.isArray(ids) && !ArrayBuffer.isView(ids))
    throw invalidAsset('GLiNER tokenizer returned invalid token IDs.');
  const result = Array.from(ids as ArrayLike<unknown>);
  if (result.some((id) => !Number.isSafeInteger(id) || (id as number) < 0 || (id as number) >= vocabularySize))
    throw invalidAsset('GLiNER tokenizer returned invalid token IDs.');
  return result as number[];
}

function specialIds(tokenizer: TokenizerImplementation): Readonly<GlinerSpecialIds> {
  const ids = Object.fromEntries(
    SPECIAL_TOKENS.map((token) => {
      const id = tokenizer.token_to_id(token);
      if (!Number.isSafeInteger(id) || (id as number) < 0) throw invalidAsset(`GLiNER tokenizer is missing ${token}.`);
      return [token, id];
    })
  ) as GlinerSpecialIds;
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

export function createGlinerMobileTokenizer(
  Tokenizer: GlinerTokenizerConstructor | null | undefined,
  data: unknown,
  tokenizerConfig: unknown,
  specialTokenMap: unknown,
  modelConfig: unknown
): GlinerTokenizer {
  if (
    typeof Tokenizer !== 'function' ||
    (data as GlinerTokenizerData | null | undefined)?.model?.type !== 'Unigram' ||
    (tokenizerConfig as GlinerTokenizerConfig | null | undefined)?.tokenizer_class !== 'DebertaV2TokenizerFast' ||
    (tokenizerConfig as GlinerTokenizerConfig | null | undefined)?.backend !== 'tokenizers' ||
    (tokenizerConfig as GlinerTokenizerConfig | null | undefined)?.pad_token !== '[PAD]' ||
    (specialTokenMap as GlinerSpecialTokenMap | null | undefined)?.pad_token?.content !== '[PAD]' ||
    (modelConfig as GlinerModelConfig | null | undefined)?.vocab_size !== 128011 ||
    (modelConfig as GlinerModelConfig | null | undefined)?.gliner2?.max_len !== 512 ||
    !Array.isArray((modelConfig as GlinerModelConfig).gliner2!.special_tokens) ||
    !SPECIAL_TOKENS.slice(0, -1).every((token) =>
      ((modelConfig as GlinerModelConfig).gliner2!.special_tokens as unknown[]).includes(token)
    )
  )
    throw invalidAsset('GLiNER tokenizer files do not match the pinned model.');

  const tokenizer = new Tokenizer(data as object, tokenizerConfig as object);
  const special = specialIds(tokenizer);
  const vocabularySize = (modelConfig as GlinerModelConfig).vocab_size as number;

  return Object.freeze({
    special,
    encode(text: string): number[] {
      return tokenIds(tokenizer.encode(text, {add_special_tokens: false}), vocabularySize);
    },
    encodeState(text: string): number[] {
      const trimmed = text.trim();
      const normalized = /[.!?]$/.test(trimmed) ? trimmed : `${trimmed}.`;
      const words = normalized.match(WORD_PATTERN) ?? [];
      return words.flatMap((word) =>
        tokenIds(tokenizer.encode(word.toLowerCase(), {add_special_tokens: false}), vocabularySize)
      );
    }
  });
}
