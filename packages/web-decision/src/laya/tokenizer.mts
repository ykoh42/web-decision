import {WebDecisionError} from '../kev.js';
import type {Tokenizer as TokenizerImplementation} from '@huggingface/tokenizers';

type TokenizerConstructor = new (data: object, config: object) => TokenizerImplementation;

interface LayaTokenizerConfig {
  readonly cls_token?: unknown;
  readonly sep_token?: unknown;
  readonly mask_token?: unknown;
  readonly pad_token?: unknown;
}

interface LayaModelConfig {
  readonly sep_token_id: number;
  readonly mask_token_id: number;
  readonly pad_token_id: number;
}

interface ExpectedSpecialTokenIds {
  readonly clsId?: number;
  readonly sepId: number;
  readonly maskId: number;
  readonly padId: number;
}

interface LayaSpecialTokens {
  readonly clsId: number;
  readonly sepId: number;
  readonly maskId: number;
  readonly padId: number;
  readonly maskToken: string;
}

export interface LayaTokenizer {
  readonly special: LayaSpecialTokens;
  encode(text: string, limit?: number): number[];
}

const invalidAsset = (message: string): WebDecisionError => new WebDecisionError('INVALID_ASSET', message);

function requireTokenId(tokenizer: TokenizerImplementation, token: string): number {
  const id = tokenizer.token_to_id(token);
  if (!Number.isSafeInteger(id) || (id as number) < 0) throw invalidAsset(`Laya tokenizer is missing ${token}.`);
  return id as number;
}

function tokenIds(encoding: unknown): number[] {
  const ids = (encoding as {readonly ids?: unknown} | null | undefined)?.ids;
  if (!Array.isArray(ids) && !ArrayBuffer.isView(ids)) throw invalidAsset('Laya tokenizer returned invalid token IDs.');
  const result = Array.from(ids as ArrayLike<unknown>);
  if (result.some((id) => !Number.isSafeInteger(id) || (id as number) < 0))
    throw invalidAsset('Laya tokenizer returned invalid token IDs.');
  return result as number[];
}

export function createLayaTokenizer(
  Tokenizer: TokenizerConstructor | null | undefined,
  data: unknown,
  tokenizerConfig: unknown,
  modelConfig: LayaModelConfig,
  expectedSpecialTokenIds: ExpectedSpecialTokenIds | null | undefined
): LayaTokenizer {
  if (
    typeof Tokenizer !== 'function' ||
    !(data as {readonly model?: {readonly vocab?: unknown} | null} | null | undefined)?.model?.vocab ||
    typeof (tokenizerConfig as LayaTokenizerConfig | null | undefined)?.cls_token !== 'string' ||
    typeof (tokenizerConfig as LayaTokenizerConfig | null | undefined)?.sep_token !== 'string' ||
    typeof (tokenizerConfig as LayaTokenizerConfig | null | undefined)?.mask_token !== 'string' ||
    typeof (tokenizerConfig as LayaTokenizerConfig | null | undefined)?.pad_token !== 'string'
  )
    throw invalidAsset('Laya tokenizer files are incomplete.');

  const tokenizer = new Tokenizer(data as object, tokenizerConfig as object);
  const special = Object.freeze({
    clsId: requireTokenId(tokenizer, (tokenizerConfig as LayaTokenizerConfig).cls_token as string),
    sepId: requireTokenId(tokenizer, (tokenizerConfig as LayaTokenizerConfig).sep_token as string),
    maskId: requireTokenId(tokenizer, (tokenizerConfig as LayaTokenizerConfig).mask_token as string),
    padId: requireTokenId(tokenizer, (tokenizerConfig as LayaTokenizerConfig).pad_token as string),
    maskToken: (tokenizerConfig as LayaTokenizerConfig).mask_token as string
  });
  const expected = expectedSpecialTokenIds ?? {
    sepId: modelConfig.sep_token_id,
    maskId: modelConfig.mask_token_id,
    padId: modelConfig.pad_token_id
  };
  if (
    (expected.clsId !== undefined && special.clsId !== expected.clsId) ||
    special.sepId !== expected.sepId ||
    special.maskId !== expected.maskId ||
    special.padId !== expected.padId
  )
    throw invalidAsset('Laya tokenizer special token IDs do not match the model config.');

  return Object.freeze({
    special,
    encode(text: string, limit: number = Number.POSITIVE_INFINITY): number[] {
      return tokenIds(tokenizer.encode(text, {add_special_tokens: false})).slice(0, limit);
    }
  });
}
