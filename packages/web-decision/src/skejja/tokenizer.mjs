import {WebDecisionError} from '../kev.js';
import {SKEJJA_MAX_TOKENS, SKEJJA_MIN_BODY_TOKENS} from './contract.mjs';

export const SKEJJA_TRUNCATION_MARKER = '\n[…]\n';
const LABEL_TOKEN = '<<LABEL>>';
const SEPARATOR_TOKEN = '<<SEP>>';

const invalidAsset = () => new WebDecisionError('INVALID_ASSET', 'Skejja tokenizer does not match the pinned model.');
const invalidInput = (message) => new WebDecisionError('INVALID_INPUT', message);

function tokenIds(tokenizer, text) {
  const ids = tokenizer.encode(text)?.ids;
  if (!Array.isArray(ids) || ids.some((id) => !Number.isSafeInteger(id) || id < 0)) throw invalidAsset();
  return ids;
}

export function createSkejjaTokenizer(Tokenizer, data, config) {
  if (
    typeof Tokenizer !== 'function' ||
    data?.model?.type !== 'BPE' ||
    config?.tokenizer_class !== 'PreTrainedTokenizerFast' ||
    config?.add_bos_token !== true ||
    config?.bos_token !== '<bos>' ||
    config?.eos_token !== '<eos>' ||
    config?.max_length !== 4096 ||
    data?.model?.vocab?.['<bos>'] !== 2 ||
    data?.model?.vocab?.['<eos>'] !== 1
  )
    throw invalidAsset();

  const tokenizer = new Tokenizer(data, config);
  if (tokenizer.token_to_id(LABEL_TOKEN) !== 32765 || tokenizer.token_to_id(SEPARATOR_TOKEN) !== 32766)
    throw invalidAsset();

  const probe = tokenIds(tokenizer, `${LABEL_TOKEN}probe${SEPARATOR_TOKEN}state`);
  if (probe[0] !== 2 || probe.at(-1) !== 1) throw invalidAsset();

  return Object.freeze({
    encode(text) {
      return tokenIds(tokenizer, text);
    },
    decode(ids) {
      const text = tokenizer.decode(ids, {skip_special_tokens: true});
      if (typeof text !== 'string') throw invalidAsset();
      return text;
    }
  });
}

export function createSkejjaPrompt(tokenizer, labels, body) {
  const prefix = `${labels.map((label) => `${LABEL_TOKEN}${label}`).join('')}${SEPARATOR_TOKEN}`;
  const prefixTokens = tokenizer.encode(prefix);
  const textBudget = SKEJJA_MAX_TOKENS - prefixTokens.length;
  if (textBudget < SKEJJA_MIN_BODY_TOKENS)
    throw invalidInput(`Skejja label text exceeds ${SKEJJA_MAX_TOKENS - SKEJJA_MIN_BODY_TOKENS} tokens.`);

  const shortened = shortenSkejjaText(tokenizer, body, textBudget);
  const inputIds = tokenizer.encode(prefix + shortened.text);
  if (inputIds.length > SKEJJA_MAX_TOKENS) throw invalidInput('Skejja input exceeds 512 tokens after truncation.');
  return {inputIds, tokenCount: inputIds.length, truncated: shortened.truncated};
}

function shortenSkejjaText(tokenizer, text, budget) {
  const ids = tokenizer.encode(text);
  if (ids.length <= budget) return {text, truncated: false};

  let keep = budget - tokenizer.encode(SKEJJA_TRUNCATION_MARKER).length;
  while (keep > 8) {
    const headLength = Math.floor((keep * 3) / 5);
    const tailLength = keep - headLength;
    const head = tokenizer.decode(ids.slice(0, headLength));
    const tail = tokenizer.decode(ids.slice(ids.length - tailLength));
    const shortened = `${head}${SKEJJA_TRUNCATION_MARKER}${tail}`;
    const excess = tokenizer.encode(shortened).length - budget;
    if (excess <= 0) return {text: shortened, truncated: true};
    keep -= excess;
  }
  throw invalidInput('Skejja could not shorten this text within the 512-token input limit.');
}
