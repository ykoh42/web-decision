import {WebDecisionError, tokenizeKevText} from './kev.js';

export const KEV08_MAX_OPTIONS = 255;

// Pinned Kev 0.8B uses different marker IDs and one independent causal row per question.
export const KEV08_MARKERS = Object.freeze({
  state: 248060,
  question: 248061,
  open: 248049,
  close: 248050,
  decide: 248062
});

const invalid = (message) => new WebDecisionError('INVALID_INPUT', message);

function append(target, values, maxTokens) {
  if (target.length + values.length > maxTokens)
    throw invalid(`A packed question exceeds ${maxTokens} tokens; no truncation was applied.`);
  for (const value of values) target.push(value);
}

function validateQuestion(question, index, packing) {
  const instructions = question?.instructions;
  if (typeof instructions !== 'string' || (!packing.allowEmptyInstructions && !instructions.trim()))
    throw invalid(`Question ${index} needs non-empty instructions.`);
  const choices = question.options;
  const minimum = packing.allowSingleOption ? 1 : 2;
  if (!Array.isArray(choices) || choices.length < minimum || choices.length > KEV08_MAX_OPTIONS)
    throw invalid(`Question ${index} needs ${minimum} to ${KEV08_MAX_OPTIONS} options.`);
  const options = [],
    choiceCount = choices.length;
  for (let i = 0; i < choiceCount; i++) {
    if (!Object.hasOwn(choices, i)) throw invalid(`Question ${index} has an empty or non-string option.`);
    const option = choices[i];
    if (typeof option !== 'string' || (!packing.allowEmptyOptions && !option.trim()))
      throw invalid(`Question ${index} has an empty or non-string option.`);
    options.push(option);
  }
  if (!packing.allowDuplicateOptions && new Set(options).size !== options.length)
    throw invalid(`Question ${index} has duplicate options.`);
  return {instructions, options};
}

function packQuestion(tokenizer, state, question, index, maxTokens, packing) {
  const {instructions, options} = validateQuestion(question, index, packing);
  const ids = [...state],
    optionSlots = [],
    tokenized = new Set();
  append(ids, [KEV08_MARKERS.question], maxTokens);
  append(ids, tokenizeKevText(tokenizer, instructions), maxTokens);
  for (const option of options) {
    const tokens = tokenizeKevText(tokenizer, option);
    const key = tokens.join(',');
    if (!packing.allowDuplicateOptions && tokenized.has(key))
      throw invalid(`Question ${index} has options that encode to the same tokens.`);
    tokenized.add(key);
    append(ids, [KEV08_MARKERS.open], maxTokens);
    append(ids, tokens, maxTokens);
    append(ids, [KEV08_MARKERS.close], maxTokens);
    optionSlots.push(ids.length - 1);
  }
  append(ids, [KEV08_MARKERS.decide], maxTokens);
  if (ids.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= 248320))
    throw new WebDecisionError('INVALID_ASSET', 'Tokenizer produced an unsupported Kev 0.8B token ID.');
  const row = {ids, questionSlot: ids.length - 1, optionSlots, options};
  if (packing.allowSingleOption) row.allowSingleOption = true;
  if (packing.allowEmptyOptions) row.allowEmptyOptions = true;
  if (packing.allowDuplicateOptions) row.allowDuplicateOptions = true;
  return row;
}

/** Snapshot one state-prefixed, independent row per choice question. */
export function packKev08Request(tokenizer, request, limits, packingOptions = {}) {
  if (!request || typeof request !== 'object' || Array.isArray(request)) throw invalid('A request object is required.');
  const stateText = request.state;
  if (typeof stateText !== 'string') throw invalid('state must be a string.');
  const questions = request.questions;
  if (!Array.isArray(questions) || !questions.length) throw invalid('Provide at least one question.');
  if (
    !Number.isSafeInteger(limits?.maxInputTokens) ||
    limits.maxInputTokens < 2 ||
    !Number.isSafeInteger(limits.maxStateTokens) ||
    limits.maxStateTokens < 1
  )
    throw new WebDecisionError('INVALID_ASSET', 'Invalid Kev 0.8B model limits.');
  const allowEmptyInstructions = packingOptions.allowEmptyInstructions === true;
  const allowSingleOption = packingOptions.allowSingleOption === true;
  const allowEmptyOptions = packingOptions.allowEmptyOptions === true;
  const allowDuplicateOptionsFor = packingOptions.allowDuplicateOptionsFor ?? new Set();
  if (!(allowDuplicateOptionsFor instanceof Set)) throw invalid('Invalid Kev 0.8B packing options.');
  const state = [KEV08_MARKERS.state, ...tokenizeKevText(tokenizer, stateText)];
  if (state.length > limits.maxStateTokens) throw invalid(`The state exceeds ${limits.maxStateTokens - 1} tokens.`);
  const rows = [],
    questionCount = questions.length;
  for (let index = 0; index < questionCount; index++) {
    if (!Object.hasOwn(questions, index)) throw invalid(`Question ${index} is missing.`);
    const packing = {
      allowEmptyInstructions,
      allowSingleOption,
      allowEmptyOptions,
      allowDuplicateOptions: allowDuplicateOptionsFor.has(index)
    };
    rows.push(packQuestion(tokenizer, state, questions[index], index, limits.maxInputTokens, packing));
  }
  return {rows, tokenCount: rows.reduce((sum, row) => sum + row.ids.length, 0)};
}

function snapshotScores(row, count) {
  const array = Array.isArray(row);
  if ((!array && !(ArrayBuffer.isView(row) && !(row instanceof DataView))) || row.length !== count)
    throw new WebDecisionError('INFERENCE_ERROR', 'The Kev 0.8B model returned invalid scores.');
  const logits = [];
  for (let i = 0; i < count; i++) {
    const value = row[i];
    if ((array && !Object.hasOwn(row, i)) || !Number.isFinite(value))
      throw new WebDecisionError('INFERENCE_ERROR', 'The Kev 0.8B model returned invalid scores.');
    logits.push(value);
  }
  return logits;
}

/** The pointer head already includes its pinned inference temperature. */
export function unpackKev08Scores(rows, packed) {
  if (!Array.isArray(rows) || rows.length !== packed.rows.length)
    throw new WebDecisionError('INFERENCE_ERROR', 'The Kev 0.8B model returned invalid scores.');
  const scores = [];
  for (let i = 0; i < rows.length; i++) {
    if (!Object.hasOwn(rows, i))
      throw new WebDecisionError('INFERENCE_ERROR', 'The Kev 0.8B model returned invalid scores.');
    scores.push(snapshotScores(rows[i], packed.rows[i].options.length));
  }
  return {
    tokenCount: packed.tokenCount,
    answers: packed.rows.map(({options}, i) => {
      const logits = scores[i],
        max = Math.max(...logits);
      const exponentials = logits.map((value) => Math.exp(value - max));
      const total = exponentials.reduce((a, b) => a + b, 0);
      const probabilities = exponentials.map((value) => value / total);
      const selectedIndex = probabilities.indexOf(Math.max(...probabilities));
      return {
        selected: options[selectedIndex],
        selectedIndex,
        probabilities: options.map((option, index) => ({
          option,
          probability: probabilities[index],
          logit: logits[index]
        }))
      };
    })
  };
}
