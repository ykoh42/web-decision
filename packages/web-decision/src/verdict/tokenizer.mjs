import {WebDecisionError} from '../kev.js';
import {VERDICT_MAX_TOKENS} from './contract.mjs';

const TOKEN_IDS = Object.freeze({
  cls: {token: '[CLS]', id: 50281},
  sep: {token: '[SEP]', id: 50282},
  pad: {token: '[PAD]', id: 50283},
  label: {token: '<<LABEL>>', id: 50368},
  query: {token: '<<SEP>>', id: 50369}
});

const invalidAsset = (message) => new WebDecisionError('INVALID_ASSET', message);

function checkedIds(encoding, special) {
  const ids = encoding?.ids;
  if (!Array.isArray(ids) && !ArrayBuffer.isView(ids))
    throw invalidAsset('Verdict tokenizer returned invalid token IDs.');
  const result = Array.from(ids);
  if (
    result.length < 2 ||
    result.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= 50370) ||
    result[0] !== special.cls ||
    result.at(-1) !== special.sep
  )
    throw invalidAsset('Verdict tokenizer returned an invalid special-token sequence.');
  if (result.length <= VERDICT_MAX_TOKENS) return {ids: result, truncated: false};
  return {
    ids: [result[0], ...result.slice(1, VERDICT_MAX_TOKENS - 1), result.at(-1)],
    truncated: true
  };
}

export function createVerdictTokenizer(Tokenizer, data, config) {
  if (
    typeof Tokenizer !== 'function' ||
    data?.model?.type !== 'BPE' ||
    !data.model.vocab ||
    config?.cls_token !== '[CLS]' ||
    config?.sep_token !== '[SEP]' ||
    config?.pad_token !== '[PAD]' ||
    config?.model_input_names?.join(',') !== 'input_ids,attention_mask'
  )
    throw invalidAsset('Verdict tokenizer files do not match the pinned checkpoint.');

  const tokenizer = new Tokenizer(data, config);
  const special = Object.fromEntries(
    Object.entries(TOKEN_IDS).map(([name, {token, id}]) => {
      if (tokenizer.token_to_id(token) !== id)
        throw invalidAsset(`Verdict tokenizer has an unexpected ID for ${token}.`);
      return [name, id];
    })
  );

  return Object.freeze({
    special: Object.freeze(special),
    encode(text) {
      if (typeof text !== 'string') throw new TypeError('Verdict tokenizer input must be a string.');
      return checkedIds(tokenizer.encode(text, {add_special_tokens: true}), special);
    }
  });
}
