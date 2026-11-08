import type {Tokenizer} from '@huggingface/tokenizers';
import {WebDecisionError} from './curated/errors.ts';

export {WebDecisionError};

export type KevTokenizer = Pick<Tokenizer, 'encode'>;

export interface KevLimits {
  readonly maxInputTokens?: number;
  readonly maxStateTokens?: number;
  readonly maxBranchTokens?: number;
  readonly maxOptions?: number;
}

export interface KevPackOptions {
  readonly allowEmptyInstructions?: boolean;
  readonly allowEmptyOptions?: boolean;
  readonly allowDuplicateOptionsFor?: ReadonlySet<number>;
}

interface KevRequest {
  readonly state: string;
  readonly questions: readonly unknown[];
}

interface KevQuestionCandidate {
  readonly instructions: string;
  readonly options?: unknown;
}

interface KevQuestionValidationOptions {
  readonly allowEmptyInstructions: boolean;
  readonly allowEmptyOptions: boolean;
  readonly allowDuplicates: boolean;
}

interface KevGroup {
  readonly indices: number[];
  readonly options: string[];
}

export interface PackedKevRequest {
  readonly ids: number[];
  readonly groups: KevGroup[];
}

export interface KevScoreProbability {
  readonly option: string;
  readonly probability: number;
  readonly logit: number;
}

export interface KevScoreAnswer {
  readonly selected: string;
  readonly selectedIndex: number;
  readonly probabilities: KevScoreProbability[];
}

export interface KevScoreResult {
  readonly tokenCount: number;
  readonly answers: KevScoreAnswer[];
}

/** Kev input packing, kept independent from the GPU runtime. */
export const KEV_DELIMITERS = Object.freeze({
  state: 151659,
  question: 151660,
  option_start: 151648,
  option_end: 151649,
  decide: 151661
});

// Native per-state and per-question row limits from the pinned Kev checkpoint.
export const KEV_MAX_STATE_TOKENS = 8192;
export const KEV_MAX_BRANCH_TOKENS = 8192;
// The pinned Kev API declares at most 255 options per question.
export const KEV_MAX_OPTIONS = 255;

function invalid(message: string): WebDecisionError {
  return new WebDecisionError('INVALID_INPUT', message);
}

function validateQuestionInstructions(
  question: unknown,
  index: number,
  allowEmpty: boolean
): asserts question is KevQuestionCandidate {
  if (
    !question ||
    typeof question !== 'object' ||
    typeof (question as KevQuestionCandidate).instructions !== 'string' ||
    (!allowEmpty && !(question as KevQuestionCandidate).instructions.trim())
  )
    throw invalid(`Question ${index} needs ${allowEmpty ? 'text' : 'non-empty instructions'}.`);
}

function validateKevRequest(request: unknown): asserts request is KevRequest {
  if (!request || typeof request !== 'object' || Array.isArray(request)) throw invalid('A request object is required.');
  const candidate = request as Record<string, unknown>;
  if (typeof candidate.state !== 'string') throw invalid('state must be a string.');
  if (!Array.isArray(candidate.questions) || candidate.questions.length === 0)
    throw invalid('Provide at least one question.');
}

function appendKevTokens(ids: number[], tokens: readonly number[], maxInputTokens: number | undefined): void {
  if (maxInputTokens !== undefined && ids.length + tokens.length > maxInputTokens)
    throw invalid(`Packed input exceeds ${maxInputTokens} tokens; no truncation was applied.`);
  for (const token of tokens) ids.push(token);
}

function validateKevOptions(
  options: unknown,
  index: number,
  allowEmpty: boolean,
  allowDuplicates: boolean,
  maxOptions: number
): string[] {
  if (!Array.isArray(options)) throw invalid(`Question ${index} needs 1 to ${maxOptions} options.`);
  const optionValues = options as unknown[];
  if (optionValues.length < 1 || optionValues.length > maxOptions)
    throw invalid(`Question ${index} needs 1 to ${maxOptions} options.`);
  const seen = new Set<string>();
  for (const option of optionValues) {
    if (typeof option !== 'string' || (!allowEmpty && !option.trim()))
      throw invalid(`Question ${index} has an empty or non-string option.`);
    if (!allowDuplicates && seen.has(option)) throw invalid(`Question ${index} has duplicate options.`);
    seen.add(option);
  }
  return [...optionValues] as string[];
}

function packKevQuestion(
  tokenizer: KevTokenizer,
  question: unknown,
  index: number,
  ids: number[],
  append: (tokens: readonly number[]) => void,
  questionOptions: KevQuestionValidationOptions,
  maxOptions: number,
  stateTokenCount: number,
  maxBranchTokens: number
): KevGroup {
  const branchStart = ids.length;
  const appendBranch = (tokens: readonly number[]): void => {
    const branchLength = ids.length - branchStart + tokens.length;
    if (stateTokenCount + branchLength > maxBranchTokens)
      throw invalid(`Question ${index} state and branch exceed ${maxBranchTokens} tokens; no truncation was applied.`);
    append(tokens);
  };
  validateQuestionInstructions(question, index, questionOptions.allowEmptyInstructions);
  const labels = validateKevOptions(
    question.options,
    index,
    questionOptions.allowEmptyOptions,
    questionOptions.allowDuplicates,
    maxOptions
  );
  appendBranch([KEV_DELIMITERS.question]);
  appendBranch(tokenizeKevText(tokenizer, question.instructions));
  const indices = [];
  const tokenizedOptions = new Set<string>();
  for (const label of labels) {
    const optionTokens = tokenizeKevText(tokenizer, label);
    const key = optionTokens.join(',');
    if (!questionOptions.allowDuplicates && tokenizedOptions.has(key))
      throw invalid(`Question ${index} has options that encode to the same tokens.`);
    tokenizedOptions.add(key);
    appendBranch([KEV_DELIMITERS.option_start]);
    appendBranch(optionTokens);
    appendBranch([KEV_DELIMITERS.option_end]);
    indices.push(ids.length - 1);
  }
  appendBranch([KEV_DELIMITERS.decide]);
  return {indices, options: labels};
}

function hasInvalidKevToken(id: number): boolean {
  return !Number.isSafeInteger(id) || id < 0 || id >= 2 ** 31;
}

export function tokenizeKevText(tokenizer: KevTokenizer, text: string): number[] {
  // JS strings can contain isolated surrogate code units. The tokenizer would
  // otherwise replace them with U+FFFD, changing the caller's input silently.
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = text.charCodeAt(++i);
      if (!(next >= 0xdc00 && next <= 0xdfff)) throw invalid('Text contains an unpaired UTF-16 surrogate.');
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      throw invalid('Text contains an unpaired UTF-16 surrogate.');
    }
  }
  // Match Kev's upstream delimiter escaping so caller text cannot introduce
  // branch markers into the model's input sequence.
  return tokenizer.encode(text.replace(/<\|([A-Za-z0-9_]+)\|>/g, '<¦$1¦>'), {add_special_tokens: false}).ids;
}

export function packKevRequest(
  tokenizer: KevTokenizer,
  request: unknown,
  limits: KevLimits,
  options: KevPackOptions = {}
): PackedKevRequest {
  validateKevRequest(request);
  const state = tokenizeKevText(tokenizer, request.state);
  const maxStateTokens = limits.maxStateTokens ?? KEV_MAX_STATE_TOKENS;
  if (state.length + 1 > maxStateTokens) throw invalid(`The state exceeds ${maxStateTokens - 1} tokens.`);
  const ids = [KEV_DELIMITERS.state, ...state];
  const stateTokenCount = ids.length;
  const groups = [];
  const {
    allowEmptyInstructions = false,
    allowEmptyOptions = false,
    allowDuplicateOptionsFor = new Set<number>()
  } = options;
  const maxInputTokens = limits.maxInputTokens;
  const maxBranchTokens = limits.maxBranchTokens ?? KEV_MAX_BRANCH_TOKENS;
  const append = (tokens: readonly number[]): void => appendKevTokens(ids, tokens, maxInputTokens);
  const maxOptions = limits.maxOptions ?? KEV_MAX_OPTIONS;
  for (let qi = 0; qi < request.questions.length; qi++) {
    groups.push(
      packKevQuestion(
        tokenizer,
        request.questions[qi],
        qi,
        ids,
        append,
        {
          allowEmptyInstructions,
          allowEmptyOptions,
          allowDuplicates: allowDuplicateOptionsFor.has(qi)
        },
        maxOptions,
        stateTokenCount,
        maxBranchTokens
      )
    );
  }
  if (ids.some(hasInvalidKevToken)) throw invalid('Tokenizer produced an unsupported token ID.');
  return {ids, groups};
}

export function unpackKevScores(logits: unknown, packed: PackedKevRequest): KevScoreResult {
  if (
    !Array.isArray(logits) ||
    logits.length !== packed.ids.length ||
    (logits as unknown[]).some((x) => !Number.isFinite(x as number))
  ) {
    throw new WebDecisionError('INFERENCE_ERROR', 'The model returned invalid scores.');
  }
  const scoreValues = logits as number[];
  return {
    tokenCount: packed.ids.length,
    answers: packed.groups.map((group) => {
      // Packing creates one valid token index and score per option.
      const scores = group.indices.map((i) => scoreValues[i]!),
        max = Math.max(...scores);
      const exps = scores.map((score) => Math.exp(score - max));
      const sum = exps.reduce((a, b) => a + b, 0);
      const probabilities = exps.map((value) => value / sum);
      const selectedIndex = probabilities.indexOf(Math.max(...probabilities));
      return {
        selected: group.options[selectedIndex]!,
        selectedIndex,
        probabilities: group.options.map((option, i) => ({option, probability: probabilities[i]!, logit: scores[i]!}))
      };
    })
  };
}

export function decodeFloat16(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1,
    exponent = (bits >> 10) & 31,
    mantissa = bits & 1023;
  return exponent === 0
    ? (sign * 2 ** -14 * mantissa) / 1024
    : exponent === 31
      ? mantissa
        ? NaN
        : sign * Infinity
      : sign * 2 ** (exponent - 15) * (1 + mantissa / 1024);
}
