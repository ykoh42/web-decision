import {WebDecisionError} from '../kev.js';

const invalidAsset = (message) => new WebDecisionError('INVALID_ASSET', message);

function requireTokenId(tokenizer, token) {
  const id = tokenizer.token_to_id(token);
  if (!Number.isSafeInteger(id) || id < 0) throw invalidAsset(`Laya tokenizer is missing ${token}.`);
  return id;
}

function tokenIds(encoding) {
  const ids = encoding?.ids;
  if (!Array.isArray(ids) && !ArrayBuffer.isView(ids)) throw invalidAsset('Laya tokenizer returned invalid token IDs.');
  const result = Array.from(ids);
  if (result.some((id) => !Number.isSafeInteger(id) || id < 0))
    throw invalidAsset('Laya tokenizer returned invalid token IDs.');
  return result;
}

export function createLayaTokenizer(Tokenizer, data, tokenizerConfig, modelConfig, expectedSpecialTokenIds) {
  if (
    typeof Tokenizer !== 'function' ||
    !data?.model?.vocab ||
    typeof tokenizerConfig?.cls_token !== 'string' ||
    typeof tokenizerConfig?.sep_token !== 'string' ||
    typeof tokenizerConfig?.mask_token !== 'string' ||
    typeof tokenizerConfig?.pad_token !== 'string'
  )
    throw invalidAsset('Laya tokenizer files are incomplete.');

  const tokenizer = new Tokenizer(data, tokenizerConfig);
  const special = Object.freeze({
    clsId: requireTokenId(tokenizer, tokenizerConfig.cls_token),
    sepId: requireTokenId(tokenizer, tokenizerConfig.sep_token),
    maskId: requireTokenId(tokenizer, tokenizerConfig.mask_token),
    padId: requireTokenId(tokenizer, tokenizerConfig.pad_token),
    maskToken: tokenizerConfig.mask_token
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
    encode(text, limit = Number.POSITIVE_INFINITY) {
      return tokenIds(tokenizer.encode(text, {add_special_tokens: false})).slice(0, limit);
    }
  });
}
