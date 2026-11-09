import {WebDecisionError} from '../kev.js';
import {TAMEV_MICRO_PROFILE} from './contract.mjs';
import type {TAMEV_PROFILES} from './contract.mjs';

type TamevProfile = (typeof TAMEV_PROFILES)[keyof typeof TAMEV_PROFILES];
interface TokenizerDataCandidate {
  readonly model?: {readonly type?: unknown} | null;
}
interface TokenizerConfigCandidate {
  readonly tokenizer_class?: unknown;
  readonly cls_token?: unknown;
  readonly sep_token?: unknown;
  readonly pad_token?: unknown;
  readonly max_length?: unknown;
  readonly padding_side?: unknown;
  readonly truncation_side?: unknown;
}
interface RawTokenizer {
  encode(text: string, options: {add_special_tokens: true}): {ids?: unknown} | null;
  token_to_id(token: string): unknown;
}
type TokenizerConstructor = new (data: unknown, tokenizerConfig: unknown) => RawTokenizer;

const invalidAsset = () =>
  new WebDecisionError('INVALID_ASSET', 'TAMEV Micro tokenizer does not match the pinned model.');

function readIds(tokenizer: RawTokenizer, text: string): number[] {
  let ids;
  try {
    ids = tokenizer.encode(text, {add_special_tokens: true})?.ids;
  } catch (cause) {
    throw new WebDecisionError('INVALID_ASSET', 'TAMEV Micro tokenizer failed.', {cause});
  }
  if (!Array.isArray(ids) || ids.length < 2 || ids.some((id) => !Number.isSafeInteger(id) || id < 0))
    throw invalidAsset();
  return ids as number[];
}

export function createTamevTokenizer(
  Tokenizer: unknown,
  data: unknown,
  tokenizerConfig: unknown,
  profile: TamevProfile = TAMEV_MICRO_PROFILE
) {
  if (
    typeof Tokenizer !== 'function' ||
    (data as TokenizerDataCandidate | null | undefined)?.model?.type !== 'WordPiece' ||
    (tokenizerConfig as TokenizerConfigCandidate | null | undefined)?.tokenizer_class !== 'BertTokenizer' ||
    (tokenizerConfig as TokenizerConfigCandidate | null | undefined)?.cls_token !== '[CLS]' ||
    (tokenizerConfig as TokenizerConfigCandidate | null | undefined)?.sep_token !== '[SEP]' ||
    (tokenizerConfig as TokenizerConfigCandidate | null | undefined)?.pad_token !== '[PAD]' ||
    ((tokenizerConfig as TokenizerConfigCandidate | null | undefined)?.max_length !== undefined &&
      (tokenizerConfig as TokenizerConfigCandidate).max_length !== profile.maxContextTokens) ||
    ((tokenizerConfig as TokenizerConfigCandidate | null | undefined)?.padding_side !== undefined &&
      (tokenizerConfig as TokenizerConfigCandidate).padding_side !== 'right') ||
    ((tokenizerConfig as TokenizerConfigCandidate | null | undefined)?.truncation_side !== undefined &&
      (tokenizerConfig as TokenizerConfigCandidate).truncation_side !== 'right')
  )
    throw invalidAsset();

  const tokenizer = new (Tokenizer as TokenizerConstructor)(data, tokenizerConfig);
  const clsId = tokenizer.token_to_id('[CLS]');
  const sepId = tokenizer.token_to_id('[SEP]');
  const padId = tokenizer.token_to_id('[PAD]');
  if (clsId !== 101 || sepId !== 102 || padId !== 0) throw invalidAsset();

  const probe = readIds(tokenizer, 'Hello world');
  if (probe[0] !== clsId || probe.at(-1) !== sepId) throw invalidAsset();

  function encode(text: string, maximum: number) {
    const ids = readIds(tokenizer, text);
    if (ids[0] !== clsId || ids.at(-1) !== sepId) throw invalidAsset();
    if (ids.length <= maximum) return {ids, truncated: false};
    return {ids: [...ids.slice(0, maximum - 1), sepId], truncated: true};
  }

  return Object.freeze({
    special: Object.freeze({clsId, sepId, padId}),
    encodeContext(text: string) {
      return encode(text, profile.maxContextTokens);
    },
    encodeOption(text: string) {
      return encode(text, profile.maxOptionTokens);
    }
  });
}
