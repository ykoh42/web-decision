import {WebDecisionError} from '../kev.js';
import decision from './decision-config.json' with {type: 'json'};

const SPECIAL = Object.freeze([
  '<|im_end|>',
  '<|endoftext|>',
  '<|audio_start|>',
  '<|audio_end|>',
  '<|audio_pad|>',
  '<|image_pad|>',
  '<|video_pad|>',
  '<|vision_start|>',
  '<|vision_end|>',
  '<|fim_suffix|>'
]);
const invalid = () => new WebDecisionError('INVALID_ASSET', 'Perplexity tokenizer does not match the pinned protocol.');

function checkedEncode(tokenizer, text, options) {
  let ids;
  try {
    ids = tokenizer.encode(text, options)?.ids;
  } catch (cause) {
    throw new WebDecisionError('INVALID_ASSET', 'Perplexity tokenizer failed.', {cause});
  }
  if (!Array.isArray(ids)) throw invalid();
  const snapshot = Array.from(ids);
  if (snapshot.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= 248320)) throw invalid();
  return {ids: snapshot};
}

export function createFlashTokenizer(Tokenizer, data, config) {
  if (
    typeof Tokenizer !== 'function' ||
    !Array.isArray(data?.added_tokens) ||
    data.added_tokens.length !== 33 ||
    !config ||
    typeof config !== 'object'
  )
    throw invalid();
  const tokens = data.added_tokens;
  if (
    tokens.some(
      (item) =>
        typeof item?.content !== 'string' ||
        !item.content ||
        !Number.isSafeInteger(item.id) ||
        item.id < 0 ||
        item.id >= 248320
    ) ||
    new Set(tokens.map((item) => item.content)).size !== tokens.length ||
    new Set(tokens.map((item) => item.id)).size !== tokens.length ||
    !tokens.some((item) => item.content === '<|fim_suffix|>' && item.id === 248062) ||
    SPECIAL.some((value) => !tokens.some((item) => item.content === value))
  )
    throw invalid();
  const inner = new Tokenizer(data, config);
  const tokenizer = Object.freeze({encode: (text, options) => checkedEncode(inner, text, options)});
  for (const [marker, id] of [
    ['<|im_start|>', 248045],
    ['<|im_end|>', 248046],
    ['<|fim_suffix|>', 248062]
  ]) {
    const direct = tokenizer.encode(marker, {add_special_tokens: false}).ids;
    const embedded = tokenizer.encode(`x${marker}y`, {add_special_tokens: false}).ids;
    if (direct.length !== 1 || direct[0] !== id || embedded.filter((item) => item === id).length !== 1) throw invalid();
  }
  if (!Array.isArray(decision.codes) || decision.codes.length !== 255 || new Set(decision.codes).size !== 255)
    throw invalid();
  return {
    tokenizer,
    config: {
      codes: [...decision.codes],
      markerId: 248062,
      escapedTokens: [...SPECIAL],
      addedTokens: tokens.map((item) => item.content),
      addedTokenIds: tokens.map((item) => item.id),
      maxInputTokens: 512
    }
  };
}
