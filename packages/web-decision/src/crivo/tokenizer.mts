import {WebDecisionError} from '../kev.js';
import {CRIVO_OPT_TOKEN_ID} from './contract.mjs';

const SPECIAL_IDS = Object.freeze({cls: 50281, sep: 50282, pad: 50283, opt: CRIVO_OPT_TOKEN_ID});
const fail = (message: string) => new WebDecisionError('INVALID_ASSET', message);

interface CrivoDecisionConfig {
  readonly opt_token?: unknown;
  readonly opt_token_id?: unknown;
  readonly question_types?: {
    readonly noul?: unknown;
    readonly choice?: unknown;
    readonly score?: unknown;
  } | null;
}

interface CrivoConfig {
  readonly model_type?: unknown;
  readonly vocab_size?: unknown;
  readonly max_position_embeddings?: unknown;
  readonly cls_token_id?: unknown;
  readonly sep_token_id?: unknown;
  readonly pad_token_id?: unknown;
  readonly decision?: CrivoDecisionConfig | null;
}

interface CrivoTokenizerInstance {
  token_to_id(token: string): unknown;
  encode(text: string, options: {readonly add_special_tokens: false}): unknown;
}

interface CrivoTokenizerConstructor {
  new (json: unknown, config: unknown): CrivoTokenizerInstance;
}

interface CrivoTokenizer {
  readonly special: typeof SPECIAL_IDS;
  encode(text: string): number[];
}

export function validateCrivoConfig(config: unknown, tokenizerConfig: unknown): void {
  const checkedConfig = config as CrivoConfig | null | undefined;
  const checkedTokenizerConfig = tokenizerConfig as Record<string, unknown> | null | undefined;
  const decision = checkedConfig?.decision;
  if (
    checkedConfig?.model_type !== 'custom' ||
    checkedConfig?.vocab_size !== 50369 ||
    checkedConfig?.max_position_embeddings !== 7999 ||
    checkedConfig?.cls_token_id !== SPECIAL_IDS.cls ||
    checkedConfig?.sep_token_id !== SPECIAL_IDS.sep ||
    checkedConfig?.pad_token_id !== SPECIAL_IDS.pad ||
    decision?.opt_token !== '[OPT]' ||
    decision?.opt_token_id !== SPECIAL_IDS.opt ||
    decision?.question_types?.noul !== 0 ||
    decision?.question_types?.choice !== 1 ||
    decision?.question_types?.score !== 2 ||
    checkedTokenizerConfig?.backend !== 'tokenizers' ||
    checkedTokenizerConfig?.cls_token !== '[CLS]' ||
    checkedTokenizerConfig?.sep_token !== '[SEP]' ||
    checkedTokenizerConfig?.pad_token !== '[PAD]' ||
    checkedTokenizerConfig?.max_length !== 7999 ||
    checkedTokenizerConfig?.model_max_length !== 7999
  )
    throw fail('Crivo configuration does not match the pinned tokenizer and ONNX graph.');
}

export function createCrivoTokenizer(
  TokenizerClass: unknown,
  tokenizerJson: unknown,
  tokenizerConfig: unknown
): CrivoTokenizer {
  const checkedTokenizerJson = tokenizerJson as {readonly model?: {readonly type?: unknown} | null} | null | undefined;
  if (typeof TokenizerClass !== 'function' || checkedTokenizerJson?.model?.type !== 'BPE')
    throw fail('Crivo tokenizer data is invalid.');
  const tokenizer = new (TokenizerClass as CrivoTokenizerConstructor)(tokenizerJson, tokenizerConfig);
  const actual: Record<keyof typeof SPECIAL_IDS, unknown> = {
    cls: tokenizer.token_to_id('[CLS]'),
    sep: tokenizer.token_to_id('[SEP]'),
    pad: tokenizer.token_to_id('[PAD]'),
    opt: tokenizer.token_to_id('[OPT]')
  };
  if (
    Object.keys(SPECIAL_IDS).some(
      (name) => actual[name as keyof typeof SPECIAL_IDS] !== SPECIAL_IDS[name as keyof typeof SPECIAL_IDS]
    )
  )
    throw fail('Crivo special token IDs do not match the pinned checkpoint.');

  return Object.freeze({
    special: SPECIAL_IDS,
    encode(text: string) {
      let result: unknown;
      try {
        const encoded = tokenizer.encode(text, {add_special_tokens: false});
        result = (encoded as {readonly ids?: unknown} | null | undefined)?.ids;
      } catch (cause) {
        throw new WebDecisionError('INVALID_INPUT', 'Crivo could not tokenize the input.', {cause});
      }
      if (!Array.isArray(result) && !ArrayBuffer.isView(result))
        throw fail('Crivo tokenizer returned invalid token IDs.');
      const ids = Array.from(result as ArrayLike<number>);
      if (ids.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= 50369))
        throw fail('Crivo tokenizer returned an out-of-range token ID.');
      return ids;
    }
  });
}
