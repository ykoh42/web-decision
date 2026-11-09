import {WebDecisionError} from '../kev.js';
import {createH2oTokenizerWithEvidence} from './tokenizer.mjs';
import type {H2oTokenizerConstructor} from './tokenizer.mjs';

const MARKERS = Object.freeze({
  '<|im_start|>': 248045,
  '<|im_end|>': 248046,
  '<think>': 248068,
  '</think>': 248069
});

type TokenizerLike = {encode(text: string, options: {add_special_tokens: false}): unknown};
type AddedToken = {
  [key: string]: unknown;
  content?: unknown;
  id?: unknown;
  lstrip?: unknown;
  normalized?: unknown;
  rstrip?: unknown;
  single_word?: unknown;
  special?: unknown;
};
type TokenizerData = {added_tokens?: Array<AddedToken | null | undefined>};
type TokenizerConfig = {added_tokens_decoder?: unknown; tokenizer_class?: unknown};
type TokenizerProbe = {text: string; token_ids: unknown};
type CheckedTokenizer = {encode(text: string, options: unknown): {ids: number[]}};
type CheckedTokenizerBinding = {
  tokenizer: CheckedTokenizer;
  addedTokens: readonly string[];
  tokenizerAddedTokenCount: number;
};
type IdArrayLike = {length: number; readonly [index: number]: unknown};

function invalidAsset(message: string): WebDecisionError {
  return new WebDecisionError('INVALID_ASSET', message);
}

function checkedIds(result: unknown): number[] {
  const ids = (result as {ids?: unknown} | null | undefined)?.ids;
  const view = ids as IdArrayLike;
  if (!Array.isArray(ids) && !(ArrayBuffer.isView(ids) && Number.isSafeInteger(view.length)))
    throw invalidAsset('Tokenizer did not return an ID array.');
  for (let i = 0; i < view.length; i++) {
    if (
      !Object.hasOwn(ids as object, i) ||
      !Number.isSafeInteger(view[i] as number) ||
      (view[i] as number) < 0 ||
      (view[i] as number) >= 248320
    )
      throw invalidAsset('Tokenizer returned an invalid token ID.');
  }
  return Array.from(ids as ArrayLike<number>);
}

function encodeChecked(tokenizer: TokenizerLike, text: string, options: {add_special_tokens: false}): number[] {
  try {
    return checkedIds(tokenizer.encode(text, options));
  } catch (cause) {
    if (cause instanceof WebDecisionError) throw cause;
    throw new WebDecisionError('INVALID_ASSET', 'Tokenizer encoding failed.', {cause});
  }
}

// Separate for synthetic tests; production callers must use prepareCheckedH2oTokenizer.
export function bindCheckedTokenizer(tokenizer: unknown, data: unknown, config: unknown): CheckedTokenizerBinding {
  const tokenizerValue = tokenizer as TokenizerLike | null | undefined;
  const dataValue = data as TokenizerData | null | undefined;
  const configValue = config as TokenizerConfig | null | undefined;
  if (
    typeof tokenizerValue?.encode !== 'function' ||
    configValue?.tokenizer_class !== 'Qwen2Tokenizer' ||
    !Array.isArray(dataValue?.added_tokens) ||
    dataValue.added_tokens.length !== 26 ||
    !configValue.added_tokens_decoder ||
    Object.keys(configValue.added_tokens_decoder as object).length !== 33
  )
    throw invalidAsset('Unexpected tokenizer added-token metadata.');
  const seen = new Set<number>();
  const addedTokens: string[] = [];
  for (let id = 248044; id <= 248076; id++) {
    const token = (configValue!.added_tokens_decoder as Record<string, AddedToken | null | undefined>)[String(id)];
    if (
      !token ||
      typeof token.content !== 'string' ||
      !token.content ||
      token.normalized !== false ||
      ['special', 'single_word', 'lstrip', 'rstrip'].some((key) => typeof token[key] !== 'boolean') ||
      addedTokens.includes(token.content)
    )
      throw invalidAsset('Invalid tokenizer added token.');
    addedTokens.push(token.content);
  }
  for (const token of dataValue!.added_tokens!) {
    if (
      !token ||
      !Number.isSafeInteger(token.id) ||
      (token.id as number) < 248044 ||
      (token.id as number) > 248069 ||
      typeof token.content !== 'string' ||
      !token.content ||
      token.normalized !== false ||
      seen.has(token.id as number)
    )
      throw invalidAsset('Invalid tokenizer added token.');
    const expected = (configValue!.added_tokens_decoder as Record<string, AddedToken | null | undefined>)[
      String(token.id as number)
    ];
    if (
      !expected ||
      ['content', 'normalized', 'special', 'single_word', 'lstrip', 'rstrip'].some(
        (key) => token[key] !== expected[key]
      )
    )
      throw invalidAsset('Tokenizer and config added tokens disagree.');
    seen.add(token.id as number);
  }
  for (const [marker, id] of Object.entries(MARKERS)) {
    if (!(dataValue!.added_tokens as AddedToken[]).some((token) => token.content === marker && token.id === id))
      throw invalidAsset('Tokenizer prompt marker does not match its pinned ID.');
    const direct = encodeChecked(tokenizerValue!, marker, {add_special_tokens: false});
    const embedded = encodeChecked(tokenizerValue!, `x${marker}y`, {add_special_tokens: false});
    if (direct.length !== 1 || direct[0] !== id || embedded.filter((value) => value === id).length !== 1)
      throw invalidAsset('Tokenizer prompt marker encoding differs from the contract.');
  }
  return {
    tokenizer: Object.freeze({
      encode(text: unknown, options: unknown): {ids: number[]} {
        const keys = options && typeof options === 'object' ? Reflect.ownKeys(options) : [];
        const descriptor =
          options && typeof options === 'object'
            ? Object.getOwnPropertyDescriptor(options, 'add_special_tokens')
            : null;
        if (
          typeof text !== 'string' ||
          keys.length !== 1 ||
          keys[0] !== 'add_special_tokens' ||
          !descriptor ||
          !Object.hasOwn(descriptor, 'value') ||
          descriptor.value !== false
        )
          throw invalidAsset('Tokenizer requires text and explicit add_special_tokens: false.');
        return {ids: encodeChecked(tokenizerValue!, text, {add_special_tokens: false})};
      }
    }),
    addedTokens: Object.freeze(addedTokens),
    tokenizerAddedTokenCount: seen.size
  };
}

/** Verify all pinned source bytes and vendor probes before binding the adapter. */
export async function prepareCheckedH2oTokenizer(
  Tokenizer: H2oTokenizerConstructor,
  tokenizerBytes: Uint8Array | ArrayBuffer,
  configBytes: Uint8Array | ArrayBuffer,
  contractBytes: Uint8Array | ArrayBuffer,
  signal?: AbortSignal
): Promise<CheckedTokenizerBinding> {
  try {
    signal?.throwIfAborted();
    const snapshot = (value: Uint8Array | ArrayBuffer, max: number): Uint8Array => {
      const view = value instanceof Uint8Array ? value : value instanceof ArrayBuffer ? new Uint8Array(value) : null;
      if (!view || !view.byteLength || view.byteLength > max) throw invalidAsset('Invalid tokenizer input bytes.');
      return Uint8Array.from(view);
    };
    // Verify and subsequently parse the same private bytes even if callers
    // mutate their original buffers while the asynchronous hashes run.
    const inputs: [Uint8Array, Uint8Array, Uint8Array] = [
      snapshot(tokenizerBytes, 24_000_000),
      snapshot(configBytes, 1_000_000),
      snapshot(contractBytes, 1_000_000)
    ];
    const {tokenizer, evidence} = await createH2oTokenizerWithEvidence(Tokenizer, ...inputs, signal);
    signal?.throwIfAborted();
    const bound = bindCheckedTokenizer(tokenizer, evidence.data, evidence.config);
    // The original builder's Array.every comparison skips sparse holes. Repeat
    // the pinned probes through the adapter's dense, bounded-ID validation.
    const {contract} = evidence;
    if (!Array.isArray(contract.probes) || contract.probes.length !== 24)
      throw invalidAsset('Invalid tokenizer probes.');
    for (const probe of contract.probes as TokenizerProbe[]) {
      signal?.throwIfAborted();
      const actual = bound.tokenizer.encode(probe.text, {add_special_tokens: false}).ids;
      const expected = probe.token_ids;
      if (
        !Array.isArray(expected) ||
        actual.length !== expected.length ||
        actual.some((value, i) => !Object.hasOwn(expected, i) || value !== expected[i])
      )
        throw invalidAsset('Tokenizer probe mismatch.');
    }
    signal?.throwIfAborted();
    return bound;
  } catch (cause) {
    signal?.throwIfAborted();
    throw new WebDecisionError('INVALID_ASSET', 'Pinned H2O tokenizer adapter verification failed.', {cause});
  }
}
