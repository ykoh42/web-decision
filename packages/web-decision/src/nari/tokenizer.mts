import {WebDecisionError} from '../kev.js';
import type {Tokenizer as TokenizerImplementation} from '@huggingface/tokenizers';

type TokenizerConstructor = new (data: object, config: object) => TokenizerImplementation;
type ParsedNariData = {readonly added_tokens: unknown[]};
type NariTokenCandidate = {readonly id?: unknown; readonly content?: unknown};
type NariConfig = {readonly tokenizer_class?: unknown};

export interface NariTokenizerResult {
  readonly tokenizer: TokenizerImplementation;
  readonly controlIds: number[];
}

const invalid = (): WebDecisionError => new WebDecisionError('INVALID_ASSET', 'Unexpected Nari tokenizer.');
const markers = {'<|im_start|>': 248045, '<|im_end|>': 248046, '<think>': 248068, '</think>': 248069};

// Input bytes must already have been verified by the pinned asset source.
export function createNariTokenizer(
  Tokenizer: unknown,
  tokenizerBytes: Uint8Array,
  configBytes: Uint8Array
): NariTokenizerResult {
  if (
    typeof Tokenizer !== 'function' ||
    !(tokenizerBytes instanceof Uint8Array) ||
    !(configBytes instanceof Uint8Array)
  )
    throw invalid();
  let data: unknown, config: unknown;
  try {
    const decoder = new TextDecoder('utf-8', {fatal: true});
    data = JSON.parse(decoder.decode(tokenizerBytes));
    config = JSON.parse(decoder.decode(configBytes));
  } catch {
    throw invalid();
  }
  if (
    (config as NariConfig | null | undefined)?.tokenizer_class !== 'Qwen3_5Tokenizer' ||
    !Array.isArray((data as {readonly added_tokens?: unknown} | null | undefined)?.added_tokens) ||
    (data as ParsedNariData).added_tokens.length !== 33
  )
    throw invalid();
  const ids = new Set<number>(),
    contents = new Set<string>();
  for (const item of (data as ParsedNariData).added_tokens) {
    if (
      !Number.isSafeInteger((item as NariTokenCandidate | null | undefined)?.id) ||
      ((item as NariTokenCandidate).id as number) < 248044 ||
      ((item as NariTokenCandidate).id as number) > 248076 ||
      typeof (item as NariTokenCandidate).content !== 'string' ||
      !(item as NariTokenCandidate).content ||
      ids.has((item as NariTokenCandidate).id as number) ||
      contents.has((item as NariTokenCandidate).content as string)
    )
      throw invalid();
    ids.add((item as NariTokenCandidate).id as number);
    contents.add((item as NariTokenCandidate).content as string);
  }
  const tokenizer = new (Tokenizer as TokenizerConstructor)(data as object, config as object);
  if (typeof tokenizer?.encode !== 'function') throw invalid();
  for (const [marker, id] of Object.entries(markers)) {
    if (
      !(data as ParsedNariData).added_tokens.some(
        (item) => (item as NariTokenCandidate).content === marker && (item as NariTokenCandidate).id === id
      )
    )
      throw invalid();
    const direct = tokenizer.encode(marker, {add_special_tokens: false})?.ids;
    const embedded = tokenizer.encode(`x${marker}y`, {add_special_tokens: false})?.ids;
    if (
      !Array.isArray(direct) ||
      direct.length !== 1 ||
      direct[0] !== id ||
      !Array.isArray(embedded) ||
      embedded.filter((value) => value === id).length !== 1
    )
      throw invalid();
  }
  return {tokenizer, controlIds: [...ids]};
}
