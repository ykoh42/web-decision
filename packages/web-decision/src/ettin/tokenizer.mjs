import {WebDecisionError} from '../kev.js';
import {ETTIN_MARKER_IDS, ETTIN_PROFILE} from './contract.mjs';

const markerNames = [
  ['[DECISION]', ETTIN_MARKER_IDS.decision],
  ['[STATE]', ETTIN_MARKER_IDS.state],
  ['[QUESTION]', ETTIN_MARKER_IDS.question],
  ...ETTIN_MARKER_IDS.options.map((id, index) => [`[OPT_${index}]`, id])
];

function invalidAsset(cause) {
  return new WebDecisionError('INVALID_ASSET', 'Decider Ettin tokenizer does not match the pinned model.', cause);
}

export function createEttinTokenizer(Tokenizer, data, config, profile = ETTIN_PROFILE) {
  if (typeof Tokenizer !== 'function' || data?.model?.type !== 'BPE' || config?.backend !== 'tokenizers')
    throw invalidAsset();

  let tokenizer;
  try {
    tokenizer = new Tokenizer(data, config);
  } catch (cause) {
    throw invalidAsset(cause);
  }

  if (typeof tokenizer.token_to_id !== 'function' || typeof tokenizer.encode !== 'function') throw invalidAsset();
  for (const [token, id] of markerNames) {
    if (tokenizer.token_to_id(token) !== id) throw invalidAsset();
  }

  return Object.freeze({
    special: profile.markerIds,
    encode(text) {
      let ids;
      try {
        ids = tokenizer.encode(text, {add_special_tokens: false})?.ids;
      } catch (cause) {
        throw new WebDecisionError('INFERENCE_ERROR', 'Decider Ettin tokenization failed.', {cause});
      }
      if (!Array.isArray(ids) && !ArrayBuffer.isView(ids))
        throw new WebDecisionError('INVALID_ASSET', 'Decider Ettin tokenizer returned invalid token IDs.');
      const result = Array.from(ids);
      if (result.some((id) => !Number.isSafeInteger(id) || id < 0))
        throw new WebDecisionError('INVALID_ASSET', 'Decider Ettin tokenizer returned invalid token IDs.');
      return result;
    }
  });
}
