import type {Tokenizer as HuggingFaceTokenizer} from '@huggingface/tokenizers';
import {WebDecisionError} from '../kev.js';
import {VERDICT_MAX_TOKENS} from './contract.mjs';

const TOKEN_IDS = Object.freeze({
  cls: {token: '[CLS]', id: 50281},
  sep: {token: '[SEP]', id: 50282},
  pad: {token: '[PAD]', id: 50283},
  label: {token: '<<LABEL>>', id: 50368},
  query: {token: '<<SEP>>', id: 50369}
} as const);

type VerdictSpecialName = keyof typeof TOKEN_IDS;
type VerdictSpecialTokens = Record<VerdictSpecialName, number>;

interface TokenizerDataCandidate {
  model?: {type?: unknown; vocab?: unknown} | null;
}

interface TokenizerConfigCandidate {
  cls_token?: unknown;
  sep_token?: unknown;
  pad_token?: unknown;
  model_input_names?: {join(separator: string): string} | null;
}

interface VerdictTokenizedText {
  ids: number[];
  truncated: boolean;
}

interface VerdictTokenizer {
  readonly special: Readonly<VerdictSpecialTokens>;
  encode(text: string): VerdictTokenizedText;
}

type TokenizerConstructor = typeof HuggingFaceTokenizer;

const invalidAsset = (message: string): WebDecisionError => new WebDecisionError('INVALID_ASSET', message);

function isVerdictTokenId(value: unknown): value is number {
  return Number.isSafeInteger(value as number) && (value as number) >= 0 && (value as number) < 50370;
}

function isVerdictTokenIdList(values: unknown[]): values is number[] {
  return values.every(isVerdictTokenId);
}

function checkedIds(encoding: unknown, special: VerdictSpecialTokens): VerdictTokenizedText {
  const ids = (encoding as {ids?: unknown} | null | undefined)?.ids;
  if (!Array.isArray(ids) && !ArrayBuffer.isView(ids))
    throw invalidAsset('Verdict tokenizer returned invalid token IDs.');
  const result = Array.from(ids as ArrayLike<unknown>);
  if (result.length < 2 || !isVerdictTokenIdList(result) || result[0] !== special.cls || result.at(-1) !== special.sep)
    throw invalidAsset('Verdict tokenizer returned an invalid special-token sequence.');
  if (result.length <= VERDICT_MAX_TOKENS) return {ids: result, truncated: false};
  return {
    ids: [result[0]!, ...result.slice(1, VERDICT_MAX_TOKENS - 1), result.at(-1)!],
    truncated: true
  };
}

export function createVerdictTokenizer(
  Tokenizer: TokenizerConstructor,
  data: object,
  config: object
): Readonly<VerdictTokenizer> {
  const tokenizerData = data as TokenizerDataCandidate;
  const tokenizerConfig = config as TokenizerConfigCandidate;
  if (
    typeof Tokenizer !== 'function' ||
    tokenizerData?.model?.type !== 'BPE' ||
    !tokenizerData.model.vocab ||
    tokenizerConfig?.cls_token !== '[CLS]' ||
    tokenizerConfig?.sep_token !== '[SEP]' ||
    tokenizerConfig?.pad_token !== '[PAD]' ||
    tokenizerConfig?.model_input_names?.join(',') !== 'input_ids,attention_mask'
  )
    throw invalidAsset('Verdict tokenizer files do not match the pinned checkpoint.');

  const tokenizer = new Tokenizer(data, config);
  const special = Object.fromEntries(
    (Object.entries(TOKEN_IDS) as [VerdictSpecialName, {token: string; id: number}][]).map(([name, {token, id}]) => {
      if (tokenizer.token_to_id(token) !== id)
        throw invalidAsset(`Verdict tokenizer has an unexpected ID for ${token}.`);
      return [name, id] as const;
    })
  ) as VerdictSpecialTokens;

  return Object.freeze({
    special: Object.freeze(special),
    encode(text: string): VerdictTokenizedText {
      if (typeof text !== 'string') throw new TypeError('Verdict tokenizer input must be a string.');
      return checkedIds(tokenizer.encode(text, {add_special_tokens: true}), special);
    }
  });
}
