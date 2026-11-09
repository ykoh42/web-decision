import {WebDecisionError} from '../kev.js';
import {DECIMA_SMALL_MAX_CHOICE_TOKENS, DECIMA_SMALL_MAX_STATE_TOKENS} from './contract.mjs';
import {normalizeDecimaText} from './normalize.mjs';

interface AddedToken {
  readonly content?: unknown;
  readonly id?: unknown;
}

interface TokenizerArtifact {
  readonly model: {readonly vocab: unknown};
  readonly added_tokens?: unknown;
}

interface TokenizerArtifactConfig {
  readonly cls_token?: unknown;
  readonly sep_token?: unknown;
  readonly pad_token?: unknown;
}

interface DecimaConfig {
  readonly choice_prefix: string;
  readonly state_prefix: string;
  readonly max_choice_tokens?: number | null | undefined;
  readonly max_state_tokens?: number | null | undefined;
  readonly choice_fit?: boolean | undefined;
  readonly state_overflow?: string | undefined;
}

export interface DecimaSpecialTokenIds {
  readonly clsId: number;
  readonly sepId: number;
  readonly padId: number;
}

export interface DecimaTokenizer {
  readonly special: DecimaSpecialTokenIds;
  encode(text: string): number[];
}

export interface PreparedDecimaQuestion {
  readonly stateIds: number[];
  readonly choiceIds: number[][];
  readonly choices: string[];
  readonly tokenCount: number;
  readonly truncated: boolean;
}

interface TokenizerEncoding {
  readonly ids?: unknown;
}

interface TokenizerImplementation {
  encode(text: string, options: {readonly add_special_tokens: false}): TokenizerEncoding | null | undefined;
}

type TokenizerConstructor = new (data: TokenizerArtifact, config: TokenizerArtifactConfig) => TokenizerImplementation;

const fail = (message: string): WebDecisionError => new WebDecisionError('INVALID_INPUT', message);

function vocabularyId(vocabulary: unknown, addedTokens: unknown, token: unknown): unknown {
  const added = Array.isArray(addedTokens)
    ? (addedTokens as unknown[]).find((item) => (item as AddedToken | null | undefined)?.content === token)
    : undefined;
  if (added && Number.isSafeInteger((added as AddedToken).id)) {
    if (
      Array.isArray(vocabulary) &&
      ((vocabulary as unknown[])[(added as AddedToken).id as number] as readonly unknown[] | null | undefined)?.[0] ===
        token
    )
      return (added as AddedToken).id;
    if (
      !Array.isArray(vocabulary) &&
      (vocabulary as Record<PropertyKey, unknown>)[token as PropertyKey] === (added as AddedToken).id
    )
      return (added as AddedToken).id;
  }
  if (Array.isArray(vocabulary))
    return (vocabulary as unknown[]).findIndex((item) => Array.isArray(item) && (item as unknown[])[0] === token);
  return (vocabulary as Record<PropertyKey, unknown>)[token as PropertyKey];
}

function tokenIds(tokenizer: TokenizerImplementation, text: string): number[] {
  const ids = tokenizer.encode(text, {add_special_tokens: false})?.ids;
  if (!Array.isArray(ids) && !ArrayBuffer.isView(ids))
    throw new WebDecisionError('INVALID_ASSET', 'Decima tokenizer returned invalid token IDs.');
  const result = Array.from(ids as ArrayLike<unknown>);
  if (result.some((id) => !Number.isSafeInteger(id) || (id as number) < 0))
    throw new WebDecisionError('INVALID_ASSET', 'Decima tokenizer returned invalid token IDs.');
  return result as number[];
}

export function createDecimaTokenizer(Tokenizer: unknown, data: unknown, config: unknown): DecimaTokenizer {
  if (
    typeof Tokenizer !== 'function' ||
    !(data as TokenizerArtifact | null | undefined)?.model?.vocab ||
    !config ||
    typeof config !== 'object'
  )
    throw new WebDecisionError('INVALID_ASSET', 'Decima tokenizer files are incomplete.');
  const vocabulary = (data as TokenizerArtifact).model.vocab;
  const clsId = vocabularyId(
    vocabulary,
    (data as TokenizerArtifact).added_tokens,
    (config as TokenizerArtifactConfig).cls_token
  );
  const sepId = vocabularyId(
    vocabulary,
    (data as TokenizerArtifact).added_tokens,
    (config as TokenizerArtifactConfig).sep_token
  );
  const padId = vocabularyId(
    vocabulary,
    (data as TokenizerArtifact).added_tokens,
    (config as TokenizerArtifactConfig).pad_token
  );
  if ([clsId, sepId, padId].some((id) => !Number.isSafeInteger(id) || (id as number) < 0))
    throw new WebDecisionError('INVALID_ASSET', 'Decima tokenizer is missing its required special tokens.');
  const tokenizer = new (Tokenizer as TokenizerConstructor)(
    data as TokenizerArtifact,
    config as TokenizerArtifactConfig
  );
  const special = Object.freeze({clsId: clsId as number, sepId: sepId as number, padId: padId as number});
  return Object.freeze({
    special,
    encode(text: string): number[] {
      return tokenIds(tokenizer, text);
    }
  });
}

function addSpecialTokens(ids: number[], special: DecimaSpecialTokenIds, maxTokens: number): number[] {
  const maxContentTokens = maxTokens - 2;
  return [special.clsId, ...ids.slice(0, maxContentTokens), special.sepId];
}

function fullChoiceText(question: string, choice: string, config: DecimaConfig, language: string): string {
  const text = `${question} ${choice}`.trim();
  return `${config.choice_prefix}${normalizeDecimaText(text, language)}`;
}

function choiceTextWithFit(
  question: string,
  choice: string,
  config: DecimaConfig,
  tokenizer: DecimaTokenizer,
  language: string
): string {
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

function buildChoiceIds(
  question: string,
  choices: readonly string[],
  config: DecimaConfig,
  tokenizer: DecimaTokenizer,
  language: string,
  allowDuplicates: boolean
): {readonly rows: number[][]; readonly truncated: boolean} {
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
  const adapted = texts.some(
    (text, index) => text !== fullChoiceText(question, choices[index] as string, config, language)
  );
  return {
    rows,
    truncated: adapted || contentRows.some((ids) => ids.length + 2 > maxTokens)
  };
}

export function prepareDecimaQuestion(
  state: string,
  instructions: string,
  choices: readonly string[],
  config: DecimaConfig,
  tokenizer: DecimaTokenizer,
  language = 'en',
  allowDuplicateChoices = false
): PreparedDecimaQuestion {
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
