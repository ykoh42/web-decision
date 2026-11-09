import {WebDecisionError} from '../kev.js';
import {SKEJJA_MAX_TOKENS, SKEJJA_MIN_BODY_TOKENS} from './contract.mjs';
import type {Tokenizer as TokenizerImplementation} from '@huggingface/tokenizers';

type TokenizerConstructor = new (data: object, config: object) => TokenizerImplementation;
type TokenizerData = {
  readonly model?: {readonly type?: unknown; readonly vocab?: {readonly [token: string]: unknown} | null} | null;
};
type TokenizerConfig = {
  readonly tokenizer_class?: unknown;
  readonly add_bos_token?: unknown;
  readonly bos_token?: unknown;
  readonly eos_token?: unknown;
  readonly max_length?: unknown;
};

export interface SkejjaTokenizer {
  encode(text: string): number[];
  decode(ids: number[]): string;
}

interface ShortenedText {
  readonly text: string;
  readonly truncated: boolean;
}

interface SkejjaPrompt {
  readonly inputIds: number[];
  readonly tokenCount: number;
  readonly truncated: boolean;
}

export const SKEJJA_TRUNCATION_MARKER = '\n[…]\n';
const LABEL_TOKEN = '<<LABEL>>';
const SEPARATOR_TOKEN = '<<SEP>>';

const invalidAsset = () => new WebDecisionError('INVALID_ASSET', 'Skejja tokenizer does not match the pinned model.');
const invalidInput = (message: string): WebDecisionError => new WebDecisionError('INVALID_INPUT', message);

function tokenIds(tokenizer: TokenizerImplementation, text: string): number[] {
  const ids = tokenizer.encode(text)?.ids;
  if (!Array.isArray(ids) || ids.some((id) => !Number.isSafeInteger(id) || id < 0)) throw invalidAsset();
  return ids as number[];
}

export function createSkejjaTokenizer(
  Tokenizer: TokenizerConstructor | null | undefined,
  data: unknown,
  config: unknown
): SkejjaTokenizer {
  if (
    typeof Tokenizer !== 'function' ||
    (data as TokenizerData | null | undefined)?.model?.type !== 'BPE' ||
    (config as TokenizerConfig | null | undefined)?.tokenizer_class !== 'PreTrainedTokenizerFast' ||
    (config as TokenizerConfig | null | undefined)?.add_bos_token !== true ||
    (config as TokenizerConfig | null | undefined)?.bos_token !== '<bos>' ||
    (config as TokenizerConfig | null | undefined)?.eos_token !== '<eos>' ||
    (config as TokenizerConfig | null | undefined)?.max_length !== 4096 ||
    (data as TokenizerData | null | undefined)?.model?.vocab?.['<bos>'] !== 2 ||
    (data as TokenizerData | null | undefined)?.model?.vocab?.['<eos>'] !== 1
  )
    throw invalidAsset();

  const tokenizer = new Tokenizer(data as object, config as object);
  if (tokenizer.token_to_id(LABEL_TOKEN) !== 32765 || tokenizer.token_to_id(SEPARATOR_TOKEN) !== 32766)
    throw invalidAsset();

  const probe = tokenIds(tokenizer, `${LABEL_TOKEN}probe${SEPARATOR_TOKEN}state`);
  if (probe[0] !== 2 || probe.at(-1) !== 1) throw invalidAsset();

  return Object.freeze({
    encode(text: string) {
      return tokenIds(tokenizer, text);
    },
    decode(ids: number[]) {
      const text = tokenizer.decode(ids, {skip_special_tokens: true});
      if (typeof text !== 'string') throw invalidAsset();
      return text;
    }
  });
}

export function createSkejjaPrompt(tokenizer: SkejjaTokenizer, labels: string[], body: string): SkejjaPrompt {
  const prefix = `${labels.map((label) => `${LABEL_TOKEN}${label}`).join('')}${SEPARATOR_TOKEN}`;
  const prefixTokens = tokenizer.encode(prefix);
  const textBudget = SKEJJA_MAX_TOKENS - prefixTokens.length;
  if (textBudget < SKEJJA_MIN_BODY_TOKENS)
    throw invalidInput(`Skejja label text exceeds ${SKEJJA_MAX_TOKENS - SKEJJA_MIN_BODY_TOKENS} tokens.`);

  const shortened = shortenSkejjaText(tokenizer, body, textBudget);
  const inputIds = tokenizer.encode(prefix + shortened.text);
  if (inputIds.length > SKEJJA_MAX_TOKENS) throw invalidInput('Skejja input exceeds 512 tokens after truncation.');
  return {inputIds, tokenCount: inputIds.length, truncated: shortened.truncated};
}

function shortenSkejjaText(tokenizer: SkejjaTokenizer, text: string, budget: number): ShortenedText {
  const ids = tokenizer.encode(text);
  if (ids.length <= budget) return {text, truncated: false};

  let keep = budget - tokenizer.encode(SKEJJA_TRUNCATION_MARKER).length;
  while (keep > 8) {
    const headLength = Math.floor((keep * 3) / 5);
    const tailLength = keep - headLength;
    const head = tokenizer.decode(ids.slice(0, headLength));
    const tail = tokenizer.decode(ids.slice(ids.length - tailLength));
    const shortened = `${head}${SKEJJA_TRUNCATION_MARKER}${tail}`;
    const excess = tokenizer.encode(shortened).length - budget;
    if (excess <= 0) return {text: shortened, truncated: true};
    keep -= excess;
  }
  throw invalidInput('Skejja could not shorten this text within the 512-token input limit.');
}
