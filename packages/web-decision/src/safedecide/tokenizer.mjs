import {WebDecisionError} from '../kev.js';
import {SAFEDECIDE_MAX_SEQUENCE_TOKENS, SAFEDECIDE_MODEL_CONFIG} from './contract.mjs';

const invalidAsset = (message) => new WebDecisionError('INVALID_ASSET', message);

function idsFor(tokenizer, tokens) {
  return tokens.map((token) => {
    const id = tokenizer.token_to_id(token);
    if (!Number.isSafeInteger(id) || id < 0) throw invalidAsset('SafeDecide tokenizer returned an unknown token.');
    return id;
  });
}

function validateTokenizerFiles(data, tokenizerConfig, modelConfig = SAFEDECIDE_MODEL_CONFIG) {
  const actualModel = data?.model;
  const processor = data?.post_processor;
  if (
    actualModel?.type !== 'BPE' ||
    !actualModel.vocab ||
    typeof actualModel.vocab !== 'object' ||
    Object.keys(actualModel.vocab).length !== modelConfig.vocabSize ||
    tokenizerConfig?.bos_token !== '<bos>' ||
    tokenizerConfig?.eos_token !== '<eos>' ||
    tokenizerConfig?.pad_token !== '<pad>' ||
    tokenizerConfig?.unk_token !== '<unk>' ||
    tokenizerConfig?.mask_token !== '<mask>' ||
    tokenizerConfig?.padding_side !== 'right' ||
    data?.truncation?.max_length !== SAFEDECIDE_MAX_SEQUENCE_TOKENS ||
    data?.truncation?.direction !== 'Right' ||
    data?.truncation?.strategy !== 'LongestFirst' ||
    processor?.type !== 'TemplateProcessing'
  )
    throw invalidAsset('SafeDecide tokenizer files do not match the pinned checkpoint.');
}

function validateSpecialTokens(tokenizer, tokenizerConfig, modelConfig) {
  const expected = modelConfig.specialTokenIds;
  const actual = {
    bosId: tokenizer.token_to_id(tokenizerConfig.bos_token),
    eosId: tokenizer.token_to_id(tokenizerConfig.eos_token),
    padId: tokenizer.token_to_id(tokenizerConfig.pad_token),
    maskId: tokenizer.token_to_id(tokenizerConfig.mask_token)
  };
  if (Object.entries(expected).some(([key, id]) => actual[key] !== id))
    throw invalidAsset('SafeDecide tokenizer special tokens do not match the model config.');
  return Object.freeze(actual);
}

export function createSafeDecideTokenizer(
  Tokenizer,
  data,
  tokenizerConfig,
  modelConfig = SAFEDECIDE_MODEL_CONFIG,
  maxSequenceTokens = SAFEDECIDE_MAX_SEQUENCE_TOKENS
) {
  if (typeof Tokenizer !== 'function' || !Number.isSafeInteger(maxSequenceTokens) || maxSequenceTokens < 4)
    throw new TypeError('SafeDecide tokenizer bindings are required.');
  validateTokenizerFiles(data, tokenizerConfig, modelConfig);
  const tokenizer = new Tokenizer(data, tokenizerConfig);
  const special = validateSpecialTokens(tokenizer, tokenizerConfig, modelConfig);
  const postProcessor = tokenizer.post_processor;
  if (typeof postProcessor?.post_process !== 'function')
    throw invalidAsset('SafeDecide tokenizer is missing its post-processor.');
  const pairSpecialTokens = postProcessor.post_process([], [], true).tokens;
  if (
    !Array.isArray(pairSpecialTokens) ||
    pairSpecialTokens.length !== 3 ||
    idsFor(tokenizer, pairSpecialTokens).join(',') !== `${special.bosId},${special.eosId},${special.eosId}`
  )
    throw invalidAsset('SafeDecide tokenizer pair template does not match the pinned checkpoint.');

  return Object.freeze({
    special,
    encodePair(first, second, limit = maxSequenceTokens) {
      if (typeof first !== 'string' || typeof second !== 'string')
        throw new TypeError('SafeDecide pair inputs must be strings.');
      if (!Number.isSafeInteger(limit) || limit < pairSpecialTokens.length)
        throw new RangeError('Invalid SafeDecide sequence limit.');
      const firstTokens = tokenizer.tokenize(first, {add_special_tokens: false});
      const secondTokens = tokenizer.tokenize(second, {add_special_tokens: false});
      const contentLimit = limit - pairSpecialTokens.length;
      let truncated = false;
      while (firstTokens.length + secondTokens.length > contentLimit) {
        if (firstTokens.length > secondTokens.length) firstTokens.pop();
        else secondTokens.pop();
        truncated = true;
      }
      const processed = postProcessor.post_process(firstTokens, secondTokens, true);
      if (!Array.isArray(processed?.tokens) || processed.tokens.length > limit)
        throw invalidAsset('SafeDecide tokenizer returned an invalid pair sequence.');
      return {inputIds: idsFor(tokenizer, processed.tokens), truncated};
    }
  });
}
