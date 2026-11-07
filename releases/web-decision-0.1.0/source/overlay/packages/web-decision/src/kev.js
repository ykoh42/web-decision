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

export class WebDecisionError extends Error {
  constructor(code, message, options) {
    super(message, options);
    this.name = 'WebDecisionError';
    this.code = code;
  }
}

function invalid(message) {
  return new WebDecisionError('INVALID_INPUT', message);
}

function validateQuestionInstructions(question, index, allowEmpty) {
  if (
    !question ||
    typeof question !== 'object' ||
    typeof question.instructions !== 'string' ||
    (!allowEmpty && !question.instructions.trim())
  )
    throw invalid(`Question ${index} needs ${allowEmpty ? 'text' : 'non-empty instructions'}.`);
}

function validateKevRequest(request) {
  if (!request || typeof request !== 'object' || Array.isArray(request)) throw invalid('A request object is required.');
  if (typeof request.state !== 'string') throw invalid('state must be a string.');
  if (!Array.isArray(request.questions) || request.questions.length === 0)
    throw invalid('Provide at least one question.');
  return request;
}

function appendKevTokens(ids, tokens, maxInputTokens) {
  if (maxInputTokens !== undefined && ids.length + tokens.length > maxInputTokens)
    throw invalid(`Packed input exceeds ${maxInputTokens} tokens; no truncation was applied.`);
  for (const token of tokens) ids.push(token);
}

function validateKevOptions(options, index, allowEmpty, allowDuplicates, maxOptions) {
  if (!Array.isArray(options) || options.length < 1 || options.length > maxOptions)
    throw invalid(`Question ${index} needs 1 to ${maxOptions} options.`);
  const seen = new Set();
  for (const option of options) {
    if (typeof option !== 'string' || (!allowEmpty && !option.trim()))
      throw invalid(`Question ${index} has an empty or non-string option.`);
    if (!allowDuplicates && seen.has(option)) throw invalid(`Question ${index} has duplicate options.`);
    seen.add(option);
  }
  return [...options];
}

function packKevQuestion(
  tokenizer,
  question,
  index,
  ids,
  append,
  options,
  maxOptions,
  stateTokenCount,
  maxBranchTokens
) {
  const branchStart = ids.length;
  const appendBranch = (tokens) => {
    const branchLength = ids.length - branchStart + tokens.length;
    if (stateTokenCount + branchLength > maxBranchTokens)
      throw invalid(`Question ${index} state and branch exceed ${maxBranchTokens} tokens; no truncation was applied.`);
    append(tokens);
  };
  validateQuestionInstructions(question, index, options.allowEmptyInstructions);
  const labels = validateKevOptions(
    question.options,
    index,
    options.allowEmptyOptions,
    options.allowDuplicates,
    maxOptions
  );
  appendBranch([KEV_DELIMITERS.question]);
  appendBranch(tokenizeKevText(tokenizer, question.instructions));
  const indices = [];
  const tokenizedOptions = new Set();
  for (const label of labels) {
    const optionTokens = tokenizeKevText(tokenizer, label);
    const key = optionTokens.join(',');
    if (!options.allowDuplicates && tokenizedOptions.has(key))
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

function hasInvalidKevToken(id) {
  return !Number.isSafeInteger(id) || id < 0 || id >= 2 ** 31;
}

export function tokenizeKevText(tokenizer, text) {
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

export function packKevRequest(tokenizer, request, limits, options = {}) {
  request = validateKevRequest(request);
  const state = tokenizeKevText(tokenizer, request.state);
  const maxStateTokens = limits.maxStateTokens ?? KEV_MAX_STATE_TOKENS;
  if (state.length + 1 > maxStateTokens) throw invalid(`The state exceeds ${maxStateTokens - 1} tokens.`);
  const ids = [KEV_DELIMITERS.state, ...state];
  const stateTokenCount = ids.length;
  const groups = [];
  const {allowEmptyInstructions = false, allowEmptyOptions = false, allowDuplicateOptionsFor = new Set()} = options;
  const maxInputTokens = limits.maxInputTokens;
  const maxBranchTokens = limits.maxBranchTokens ?? KEV_MAX_BRANCH_TOKENS;
  const append = (tokens) => appendKevTokens(ids, tokens, maxInputTokens);
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

export function unpackKevScores(logits, packed) {
  if (!Array.isArray(logits) || logits.length !== packed.ids.length || logits.some((x) => !Number.isFinite(x))) {
    throw new WebDecisionError('INFERENCE_ERROR', 'The model returned invalid scores.');
  }
  return {
    tokenCount: packed.ids.length,
    answers: packed.groups.map((group) => {
      const scores = group.indices.map((i) => logits[i]),
        max = Math.max(...scores);
      const exps = scores.map((score) => Math.exp(score - max));
      const sum = exps.reduce((a, b) => a + b, 0);
      const probabilities = exps.map((value) => value / sum);
      const selectedIndex = probabilities.indexOf(Math.max(...probabilities));
      return {
        selected: group.options[selectedIndex],
        selectedIndex,
        probabilities: group.options.map((option, i) => ({option, probability: probabilities[i], logit: scores[i]}))
      };
    })
  };
}

export function decodeFloat16(bits) {
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
