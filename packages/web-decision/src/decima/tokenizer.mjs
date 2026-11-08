import {WebDecisionError} from '../kev.js';
import {DECIMA_SMALL_MAX_CHOICE_TOKENS, DECIMA_SMALL_MAX_STATE_TOKENS} from './contract.mjs';
import {normalizeDecimaText} from './normalize.mjs';

const fail = (message) => new WebDecisionError('INVALID_INPUT', message);

function vocabularyId(vocabulary, addedTokens, token) {
  const added = Array.isArray(addedTokens) ? addedTokens.find((item) => item?.content === token) : undefined;
  if (added && Number.isSafeInteger(added.id)) {
    if (Array.isArray(vocabulary) && vocabulary[added.id]?.[0] === token) return added.id;
    if (!Array.isArray(vocabulary) && vocabulary[token] === added.id) return added.id;
  }
  if (Array.isArray(vocabulary)) return vocabulary.findIndex((item) => Array.isArray(item) && item[0] === token);
  return vocabulary[token];
}

function tokenIds(tokenizer, text) {
  const ids = tokenizer.encode(text, {add_special_tokens: false})?.ids;
  if (!Array.isArray(ids) && !ArrayBuffer.isView(ids))
    throw new WebDecisionError('INVALID_ASSET', 'Decima tokenizer returned invalid token IDs.');
  const result = Array.from(ids);
  if (result.some((id) => !Number.isSafeInteger(id) || id < 0))
    throw new WebDecisionError('INVALID_ASSET', 'Decima tokenizer returned invalid token IDs.');
  return result;
}

export function createDecimaTokenizer(Tokenizer, data, config) {
  if (typeof Tokenizer !== 'function' || !data?.model?.vocab || !config || typeof config !== 'object')
    throw new WebDecisionError('INVALID_ASSET', 'Decima tokenizer files are incomplete.');
  const vocabulary = data.model.vocab;
  const clsId = vocabularyId(vocabulary, data.added_tokens, config.cls_token);
  const sepId = vocabularyId(vocabulary, data.added_tokens, config.sep_token);
  const padId = vocabularyId(vocabulary, data.added_tokens, config.pad_token);
  if ([clsId, sepId, padId].some((id) => !Number.isSafeInteger(id) || id < 0))
    throw new WebDecisionError('INVALID_ASSET', 'Decima tokenizer is missing its required special tokens.');
  const tokenizer = new Tokenizer(data, config);
  const special = Object.freeze({clsId, sepId, padId});
  return Object.freeze({
    special,
    encode(text) {
      return tokenIds(tokenizer, text);
    }
  });
}

function addSpecialTokens(ids, special, maxTokens) {
  const maxContentTokens = maxTokens - 2;
  return [special.clsId, ...ids.slice(0, maxContentTokens), special.sepId];
}

function fullChoiceText(question, choice, config, language) {
  const text = `${question} ${choice}`.trim();
  return `${config.choice_prefix}${normalizeDecimaText(text, language)}`;
}

function choiceTextWithFit(question, choice, config, tokenizer, language) {
  const maxTokens = config.max_choice_tokens ?? DECIMA_SMALL_MAX_CHOICE_TOKENS;
  const fullText = fullChoiceText(question, choice, config, language);
  if (tokenizer.encode(fullText).length + 2 <= maxTokens) return fullText;
  const words = normalizeDecimaText(question, language).split(/\s+/u).filter(Boolean);
  const normalizedChoice = normalizeDecimaText(choice, language);
  let low = 0;
  let high = words.length;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    const text = `${config.choice_prefix}${[...words.slice(0, middle), '…', normalizedChoice].join(' ')}`;
    if (tokenizer.encode(text).length + 2 <= maxTokens) low = middle;
    else high = middle - 1;
  }
  const questionHead = words.slice(0, low).join(' ');
  const body = questionHead ? `${questionHead} … ${normalizedChoice}` : normalizedChoice;
  return `${config.choice_prefix}${body}`;
}

function buildChoiceIds(question, choices, config, tokenizer, language, allowDuplicates) {
  const maxTokens = config.max_choice_tokens ?? DECIMA_SMALL_MAX_CHOICE_TOKENS;
  let texts = choices.map((choice) => fullChoiceText(question, choice, config, language));
  let contentRows = texts.map((text) => tokenizer.encode(text));
  let rows = contentRows.map((ids) => addSpecialTokens(ids, tokenizer.special, maxTokens));
  const distinct = new Set(rows.map((row) => row.join(',')));
  const fitted = config.choice_fit === true || distinct.size !== rows.length;
  if (fitted) {
    texts = choices.map((choice) => choiceTextWithFit(question, choice, config, tokenizer, language));
    contentRows = texts.map((text) => tokenizer.encode(text));
    rows = contentRows.map((ids) => addSpecialTokens(ids, tokenizer.special, maxTokens));
  }
  if (!allowDuplicates && new Set(rows.map((row) => row.join(','))).size !== rows.length)
    throw fail('Choice texts encode to identical Decima inputs.');
  const adapted = texts.some((text, index) => text !== fullChoiceText(question, choices[index], config, language));
  return {
    rows,
    truncated: adapted || contentRows.some((ids) => ids.length + 2 > maxTokens)
  };
}

export function prepareDecimaQuestion(
  state,
  instructions,
  choices,
  config,
  tokenizer,
  language = 'en',
  allowDuplicateChoices = false
) {
  if (typeof state !== 'string' || typeof instructions !== 'string' || !instructions.trim())
    throw fail('State and non-empty instructions are required.');
  const maxStateTokens = config.max_state_tokens ?? DECIMA_SMALL_MAX_STATE_TOKENS;
  if (!Array.isArray(choices) || choices.length < 2) throw fail('A Decima choice needs at least 2 options.');
  if (choices.some((choice) => typeof choice !== 'string' || !choice.trim()))
    throw fail('Decima options must be non-empty strings.');
  if (!allowDuplicateChoices && new Set(choices).size !== choices.length)
    throw fail('Decima options must be distinct.');
  if (typeof language !== 'string' || !language.trim()) throw fail('Decima language must be a non-empty string.');

  const question = normalizeDecimaText(instructions, language);
  const stateText = `${config.state_prefix}${normalizeDecimaText(`${question}\n${state}`, language)}`;
  const stateContent = tokenizer.encode(stateText);
  const stateOverflow = stateContent.length + 2 > maxStateTokens;
  if (stateOverflow && config.state_overflow === 'reject')
    throw fail(`Decima state exceeds the ${maxStateTokens}-token limit.`);
  const stateIds = addSpecialTokens(stateContent, tokenizer.special, maxStateTokens);
  const {rows: choiceIds, truncated} = buildChoiceIds(
    question,
    choices,
    config,
    tokenizer,
    language,
    allowDuplicateChoices
  );
  return {
    stateIds,
    choiceIds,
    choices: [...choices],
    tokenCount: stateIds.length + choiceIds.reduce((sum, row) => sum + row.length, 0),
    truncated: truncated || (stateOverflow && config.state_overflow !== 'reject')
  };
}
