import {WebDecisionError} from '../kev.js';
import {TAMEV_NANO_PROFILE} from './contract.mjs';

const invalidAsset = () =>
  new WebDecisionError('INVALID_ASSET', 'TAMEV tokenizer does not match the pinned model.');

function readIds(tokenizer, text) {
  let ids;
  try {
    ids = tokenizer.encode(text, {add_special_tokens: true})?.ids;
  } catch (cause) {
    throw new WebDecisionError('INVALID_ASSET', 'TAMEV tokenizer failed.', {cause});
  }
  if (!Array.isArray(ids) || ids.length < 2 || ids.some((id) => !Number.isSafeInteger(id) || id < 0))
    throw invalidAsset();
  return ids;
}

export function createTamevTokenizer(Tokenizer, data, tokenizerConfig, profile = TAMEV_NANO_PROFILE) {
  if (
    typeof Tokenizer !== 'function' ||
    data?.model?.type !== 'WordPiece' ||
    tokenizerConfig?.tokenizer_class !== 'BertTokenizer' ||
    tokenizerConfig?.cls_token !== '[CLS]' ||
    tokenizerConfig?.sep_token !== '[SEP]' ||
    tokenizerConfig?.pad_token !== '[PAD]' ||
    (tokenizerConfig?.max_length !== undefined && tokenizerConfig.max_length !== profile.maxContextTokens) ||
    (tokenizerConfig?.padding_side !== undefined && tokenizerConfig.padding_side !== 'right') ||
    (tokenizerConfig?.truncation_side !== undefined && tokenizerConfig.truncation_side !== 'right')
  )
    throw invalidAsset();

  const tokenizer = new Tokenizer(data, tokenizerConfig);
  const clsId = tokenizer.token_to_id('[CLS]');
  const sepId = tokenizer.token_to_id('[SEP]');
  const padId = tokenizer.token_to_id('[PAD]');
  if (clsId !== 101 || sepId !== 102 || padId !== 0) throw invalidAsset();

  const probe = readIds(tokenizer, 'Hello world');
  if (probe[0] !== clsId || probe.at(-1) !== sepId) throw invalidAsset();

  function encode(text, maximum) {
    const ids = readIds(tokenizer, text);
    if (ids[0] !== clsId || ids.at(-1) !== sepId) throw invalidAsset();
    if (ids.length <= maximum) return {ids, truncated: false};
    return {ids: [...ids.slice(0, maximum - 1), sepId], truncated: true};
  }

  return Object.freeze({
    special: Object.freeze({clsId, sepId, padId}),
    encodeContext(text) {
      return encode(text, profile.maxContextTokens);
    },
    encodeOption(text) {
      return encode(text, profile.maxOptionTokens);
    }
  });
}
