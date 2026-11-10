import {Tokenizer} from '@huggingface/tokenizers';
import {WebDecisionError} from '../kev.js';
import {KANANA_IMAGE_TOKEN_ID, KANANA_MODEL_MAX_LENGTH, KANANA_VOCAB_SIZE} from './contract.mjs';

const SPECIAL_TOKEN_IDS = Object.freeze({
  '<|begin_of_text|>': 128000,
  '<|end_of_text|>': 128001,
  '<|im_start|>': 128009,
  '<|im_end|>': 128010,
  '<|vision_start|>': 128017,
  '<|vision_end|>': 128018,
  '<|image_pad|>': KANANA_IMAGE_TOKEN_ID,
  '<|video_pad|>': 128021,
  '<think>': 128032,
  '</think>': 128033
});

type TokenizerRuntime = Pick<InstanceType<typeof Tokenizer>, 'get_vocab' | 'token_to_id' | 'encode' | 'decode'>;
type TokenizerConstructor = new (tokenizerJson: object, tokenizerConfig: object) => TokenizerRuntime;
type EncodeOptions = Parameters<TokenizerRuntime['encode']>[1];
type DecodeOptions = Parameters<TokenizerRuntime['decode']>[1];
type KananaTokenizer = Readonly<{
  ids: Readonly<Record<string, number>>;
  encode(text: string, options?: EncodeOptions): ReturnType<TokenizerRuntime['encode']>;
  decode(tokenIds: unknown, options?: DecodeOptions): string;
}>;

function invalidAsset(message: string) {
  return new WebDecisionError('INVALID_ASSET', message);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function numberArray(value: unknown, message: string): number[] {
  if (!Array.isArray(value) && (!ArrayBuffer.isView(value) || value instanceof DataView)) throw invalidAsset(message);
  const values = Array.from(value as ArrayLike<unknown>);
  if (values.some((id) => !Number.isSafeInteger(id) || typeof id !== 'number' || id < 0 || id >= KANANA_VOCAB_SIZE))
    throw invalidAsset('Kanana tokenizer returned an out-of-range token ID.');
  return values as number[];
}

function checkedIds(encoding: unknown): number[] {
  const ids = isRecord(encoding) ? encoding.ids : undefined;
  if ((!Array.isArray(ids) && !ArrayBuffer.isView(ids)) || ids instanceof DataView)
    throw invalidAsset('Kanana tokenizer returned invalid token IDs.');
  return numberArray(ids, 'Kanana tokenizer returned invalid token IDs.');
}

function validateMetadata(tokenizerJson: unknown, tokenizerConfig: unknown, modelConfig: unknown): void {
  const tokenizerModel = isRecord(tokenizerJson) && isRecord(tokenizerJson.model) ? tokenizerJson.model : undefined;
  const tokenizerSettings = isRecord(tokenizerConfig) ? tokenizerConfig : undefined;
  const modelSettings = isRecord(modelConfig) ? modelConfig : undefined;
  const textSettings = modelSettings && isRecord(modelSettings.text_config) ? modelSettings.text_config : undefined;
  if (
    tokenizerModel?.type !== 'BPE' ||
    tokenizerSettings?.backend !== 'tokenizers' ||
    tokenizerSettings?.tokenizer_class !== 'TokenizersBackend' ||
    tokenizerSettings?.bos_token !== '<|begin_of_text|>' ||
    tokenizerSettings?.eos_token !== '<|im_end|>' ||
    tokenizerSettings?.pad_token !== '<|end_of_text|>' ||
    tokenizerSettings?.padding_side !== 'left' ||
    tokenizerSettings?.model_max_length !== KANANA_MODEL_MAX_LENGTH ||
    modelSettings?.model_type !== 'kanana_v_decide' ||
    modelSettings?.max_length !== KANANA_MODEL_MAX_LENGTH ||
    modelSettings?.image_token_id !== SPECIAL_TOKEN_IDS['<|image_pad|>'] ||
    textSettings?.vocab_size !== KANANA_VOCAB_SIZE
  )
    throw invalidAsset('Kanana tokenizer and configuration do not match the pinned checkpoint.');
}

/** Construct the pinned tokenizer with the same defaults used by the Python tokenizer. */
export function createKananaTokenizer(
  tokenizerJson: object,
  tokenizerConfig: object,
  modelConfig: object,
  TokenizerClass: TokenizerConstructor = Tokenizer
): KananaTokenizer {
  validateMetadata(tokenizerJson, tokenizerConfig, modelConfig);
  if (typeof TokenizerClass !== 'function') throw invalidAsset('Kanana tokenizer runtime is unavailable.');

  const tokenizer = new TokenizerClass(tokenizerJson, tokenizerConfig);
  if (
    typeof tokenizer?.get_vocab !== 'function' ||
    typeof tokenizer?.token_to_id !== 'function' ||
    typeof tokenizer?.encode !== 'function' ||
    typeof tokenizer?.decode !== 'function'
  )
    throw invalidAsset('Kanana tokenizer runtime is incomplete.');
  if (tokenizer.get_vocab(true).size !== KANANA_VOCAB_SIZE)
    throw invalidAsset('Kanana tokenizer vocabulary size does not match the pinned checkpoint.');

  const ids: Record<string, number> = Object.fromEntries(
    Object.entries(SPECIAL_TOKEN_IDS).map(([token, expected]) => {
      const actual = tokenizer.token_to_id(token);
      if (actual !== expected) throw invalidAsset(`Kanana tokenizer has an unexpected ID for ${token}.`);
      return [token, actual];
    })
  );

  return Object.freeze({
    ids: Object.freeze(ids),
    encode(text: string, options?: EncodeOptions) {
      if (typeof text !== 'string') throw new TypeError('Kanana tokenizer input must be a string.');
      const encoded = tokenizer.encode(text, options);
      checkedIds(encoded);
      return encoded;
    },
    decode(tokenIds: unknown, options?: DecodeOptions) {
      let idsToDecode: number[];
      try {
        idsToDecode = numberArray(tokenIds, 'Kanana tokenizer IDs must be an array.');
      } catch {
        throw new TypeError('Kanana tokenizer IDs are out of range.');
      }
      const decoded = tokenizer.decode(idsToDecode, options);
      if (typeof decoded !== 'string') throw invalidAsset('Kanana tokenizer returned invalid decoded text.');
      return decoded;
    }
  });
}
