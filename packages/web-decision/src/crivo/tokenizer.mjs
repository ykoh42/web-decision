import {WebDecisionError} from '../kev.js';
import {CRIVO_OPT_TOKEN_ID} from './contract.mjs';

const SPECIAL_IDS = Object.freeze({cls: 50281, sep: 50282, pad: 50283, opt: CRIVO_OPT_TOKEN_ID});
const fail = (message) => new WebDecisionError('INVALID_ASSET', message);

export function validateCrivoConfig(config, tokenizerConfig) {
  const decision = config?.decision;
  if (
    config?.model_type !== 'custom' ||
    config?.vocab_size !== 50369 ||
    config?.max_position_embeddings !== 7999 ||
    config?.cls_token_id !== SPECIAL_IDS.cls ||
    config?.sep_token_id !== SPECIAL_IDS.sep ||
    config?.pad_token_id !== SPECIAL_IDS.pad ||
    decision?.opt_token !== '[OPT]' ||
    decision?.opt_token_id !== SPECIAL_IDS.opt ||
    decision?.question_types?.noul !== 0 ||
    decision?.question_types?.choice !== 1 ||
    decision?.question_types?.score !== 2 ||
    tokenizerConfig?.backend !== 'tokenizers' ||
    tokenizerConfig?.cls_token !== '[CLS]' ||
    tokenizerConfig?.sep_token !== '[SEP]' ||
    tokenizerConfig?.pad_token !== '[PAD]' ||
    tokenizerConfig?.max_length !== 7999 ||
    tokenizerConfig?.model_max_length !== 7999
  )
    throw fail('Crivo configuration does not match the pinned tokenizer and ONNX graph.');
}

export function createCrivoTokenizer(TokenizerClass, tokenizerJson, tokenizerConfig) {
  if (typeof TokenizerClass !== 'function' || tokenizerJson?.model?.type !== 'BPE')
    throw fail('Crivo tokenizer data is invalid.');
  const tokenizer = new TokenizerClass(tokenizerJson, tokenizerConfig);
  const actual = {
    cls: tokenizer.token_to_id('[CLS]'),
    sep: tokenizer.token_to_id('[SEP]'),
    pad: tokenizer.token_to_id('[PAD]'),
    opt: tokenizer.token_to_id('[OPT]')
  };
  if (Object.keys(SPECIAL_IDS).some((name) => actual[name] !== SPECIAL_IDS[name]))
    throw fail('Crivo special token IDs do not match the pinned checkpoint.');

  return Object.freeze({
    special: SPECIAL_IDS,
    encode(text) {
      let result;
      try {
        result = tokenizer.encode(text, {add_special_tokens: false})?.ids;
      } catch (cause) {
        throw new WebDecisionError('INVALID_INPUT', 'Crivo could not tokenize the input.', {cause});
      }
      if (!Array.isArray(result) && !ArrayBuffer.isView(result))
        throw fail('Crivo tokenizer returned invalid token IDs.');
      const ids = Array.from(result);
      if (ids.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= 50369))
        throw fail('Crivo tokenizer returned an out-of-range token ID.');
      return ids;
    }
  });
}
