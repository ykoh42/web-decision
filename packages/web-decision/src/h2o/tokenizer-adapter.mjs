import {WebDecisionError} from '../kev.js';
import {createH2oTokenizerWithEvidence} from './tokenizer.mjs';

const MARKERS = Object.freeze({
  '<|im_start|>': 248045,
  '<|im_end|>': 248046,
  '<think>': 248068,
  '</think>': 248069
});

function invalidAsset(message) {
  return new WebDecisionError('INVALID_ASSET', message);
}

function checkedIds(result) {
  const ids = result?.ids;
  if (!Array.isArray(ids) && !(ArrayBuffer.isView(ids) && Number.isSafeInteger(ids.length)))
    throw invalidAsset('Tokenizer did not return an ID array.');
  for (let i = 0; i < ids.length; i++) {
    if (!Object.hasOwn(ids, i) || !Number.isSafeInteger(ids[i]) || ids[i] < 0 || ids[i] >= 248320)
      throw invalidAsset('Tokenizer returned an invalid token ID.');
  }
  return Array.from(ids);
}

function encodeChecked(tokenizer, text, options) {
  try {
    return checkedIds(tokenizer.encode(text, options));
  } catch (cause) {
    if (cause instanceof WebDecisionError) throw cause;
    throw new WebDecisionError('INVALID_ASSET', 'Tokenizer encoding failed.', {cause});
  }
}

// Separate for synthetic tests; production callers must use prepareCheckedH2oTokenizer.
export function bindCheckedTokenizer(tokenizer, data, config) {
  if (
    typeof tokenizer?.encode !== 'function' ||
    config?.tokenizer_class !== 'Qwen2Tokenizer' ||
    !Array.isArray(data?.added_tokens) ||
    data.added_tokens.length !== 26 ||
    !config.added_tokens_decoder ||
    Object.keys(config.added_tokens_decoder).length !== 33
  )
    throw invalidAsset('Unexpected tokenizer added-token metadata.');
  const seen = new Set();
  const addedTokens = [];
  for (let id = 248044; id <= 248076; id++) {
    const token = config.added_tokens_decoder[String(id)];
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
  for (const token of data.added_tokens) {
    if (
      !token ||
      !Number.isSafeInteger(token.id) ||
      token.id < 248044 ||
      token.id > 248069 ||
      typeof token.content !== 'string' ||
      !token.content ||
      token.normalized !== false ||
      seen.has(token.id)
    )
      throw invalidAsset('Invalid tokenizer added token.');
    const expected = config.added_tokens_decoder[String(token.id)];
    if (
      !expected ||
      ['content', 'normalized', 'special', 'single_word', 'lstrip', 'rstrip'].some(
        (key) => token[key] !== expected[key]
      )
    )
      throw invalidAsset('Tokenizer and config added tokens disagree.');
    seen.add(token.id);
  }
  for (const [marker, id] of Object.entries(MARKERS)) {
    if (!data.added_tokens.some((token) => token.content === marker && token.id === id))
      throw invalidAsset('Tokenizer prompt marker does not match its pinned ID.');
    const direct = encodeChecked(tokenizer, marker, {add_special_tokens: false});
    const embedded = encodeChecked(tokenizer, `x${marker}y`, {add_special_tokens: false});
    if (direct.length !== 1 || direct[0] !== id || embedded.filter((value) => value === id).length !== 1)
      throw invalidAsset('Tokenizer prompt marker encoding differs from the contract.');
  }
  return {
    tokenizer: Object.freeze({
      encode(text, options) {
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
        return {ids: encodeChecked(tokenizer, text, {add_special_tokens: false})};
      }
    }),
    addedTokens: Object.freeze(addedTokens),
    tokenizerAddedTokenCount: seen.size
  };
}

/** Verify all pinned source bytes and vendor probes before binding the adapter. */
export async function prepareCheckedH2oTokenizer(Tokenizer, tokenizerBytes, configBytes, contractBytes, signal) {
  try {
    signal?.throwIfAborted();
    const snapshot = (value, max) => {
      const view = value instanceof Uint8Array ? value : value instanceof ArrayBuffer ? new Uint8Array(value) : null;
      if (!view || !view.byteLength || view.byteLength > max) throw invalidAsset('Invalid tokenizer input bytes.');
      return Uint8Array.from(view);
    };
    // Verify and subsequently parse the same private bytes even if callers
    // mutate their original buffers while the asynchronous hashes run.
    const inputs = [
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
    for (const probe of contract.probes) {
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
