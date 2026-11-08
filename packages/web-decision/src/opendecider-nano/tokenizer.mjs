import {WebDecisionError} from '../kev.js';

export const OPENDECIDER_NANO_TOKENS = Object.freeze({cls: 50281, sep: 50282, pad: 50283, mask: 50284});
const VOCAB_SIZE = 50368;
const invalid = () =>
  new WebDecisionError('INVALID_ASSET', 'OpenDecider-nano tokenizer does not match the pinned model.');

function textIds(tokenizer, text) {
  let ids;
  try {
    ids = tokenizer.encode(text, {add_special_tokens: false, split_special_tokens: true})?.ids;
  } catch (cause) {
    throw new WebDecisionError('INVALID_ASSET', 'OpenDecider-nano tokenizer failed.', {cause});
  }
  if (!Array.isArray(ids) || ids.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= VOCAB_SIZE)) throw invalid();
  if (ids.some((id) => Object.values(OPENDECIDER_NANO_TOKENS).includes(id))) throw invalid();
  return [...ids];
}

export function createOpenDeciderNanoTokenizer(Tokenizer, data, config) {
  if (
    typeof Tokenizer !== 'function' ||
    !data ||
    !Array.isArray(data.added_tokens) ||
    !data.model?.vocab ||
    typeof data.model.vocab !== 'object' ||
    !config ||
    typeof config !== 'object' ||
    Array.isArray(config)
  )
    throw invalid();
  const ids = [...Object.values(data.model.vocab), ...data.added_tokens.map((item) => item?.id)];
  if (
    ids.length < 1 ||
    ids.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= VOCAB_SIZE) ||
    Math.max(...ids) !== VOCAB_SIZE - 1
  )
    throw invalid();
  const tokenizer = new Tokenizer(data, config);
  for (const [text, expected] of [
    ['question: Which team should handle this?', [19751, 27, 6758, 2285, 943, 6016, 436, 32]],
    ['input: I was charged twice for order 1182.', [5423, 27, 309, 369, 6636, 7019, 323, 1340, 12643, 19, 15]]
  ]) {
    const actual = textIds(tokenizer, text);
    if (actual.length !== expected.length || actual.some((id, index) => id !== expected[index])) throw invalid();
  }
  return Object.freeze({encode: (text) => textIds(tokenizer, text)});
}
