import {WebDecisionError} from '../kev.js';
import {ETTIN_MARKER_IDS, ETTIN_PROFILE} from './contract.mjs';
import type {EttinProfile} from './contract.mjs';
import type {EttinTokenizer} from './protocol.mjs';

interface EttinTokenizerInstance {
  token_to_id(token: string): number | undefined;
  encode(text: string, options: {readonly add_special_tokens: false}): {readonly ids?: unknown};
}

interface EttinTokenizerConstructor {
  new (data: object, config: object): EttinTokenizerInstance;
}

const markerNames: readonly (readonly [string, number])[] = [
  ['[DECISION]', ETTIN_MARKER_IDS.decision],
  ['[STATE]', ETTIN_MARKER_IDS.state],
  ['[QUESTION]', ETTIN_MARKER_IDS.question],
  ...ETTIN_MARKER_IDS.options.map((id, index): [string, number] => [`[OPT_${index}]`, id])
];

function invalidAsset(cause?: unknown) {
  return new WebDecisionError(
    'INVALID_ASSET',
    'Decider Ettin tokenizer does not match the pinned model.',
    cause as ErrorOptions | undefined
  );
}

export function createEttinTokenizer(
  Tokenizer: unknown,
  data: unknown,
  config: unknown,
  profile: EttinProfile = ETTIN_PROFILE
): EttinTokenizer {
  if (
    typeof Tokenizer !== 'function' ||
    (data as {readonly model?: {readonly type?: unknown} | null} | null | undefined)?.model?.type !== 'BPE' ||
    (config as {readonly backend?: unknown} | null | undefined)?.backend !== 'tokenizers'
  )
    throw invalidAsset();

  let tokenizer: EttinTokenizerInstance;
  try {
    tokenizer = new (Tokenizer as EttinTokenizerConstructor)(data as object, config as object);
  } catch (cause) {
    throw invalidAsset(cause);
  }

  if (typeof tokenizer.token_to_id !== 'function' || typeof tokenizer.encode !== 'function') throw invalidAsset();
  for (const [token, id] of markerNames) {
    if (tokenizer.token_to_id(token) !== id) throw invalidAsset();
  }

  return Object.freeze({
    special: profile.markerIds,
    encode(text: string): number[] {
      let ids: unknown;
      try {
        ids = tokenizer.encode(text, {add_special_tokens: false})?.ids;
      } catch (cause) {
        throw new WebDecisionError('INFERENCE_ERROR', 'Decider Ettin tokenization failed.', {cause});
      }
      if (!Array.isArray(ids) && !ArrayBuffer.isView(ids))
        throw new WebDecisionError('INVALID_ASSET', 'Decider Ettin tokenizer returned invalid token IDs.');
      const result = Array.from(ids as ArrayLike<number>);
      if (result.some((id) => !Number.isSafeInteger(id) || id < 0))
        throw new WebDecisionError('INVALID_ASSET', 'Decider Ettin tokenizer returned invalid token IDs.');
      return result;
    }
  });
}
