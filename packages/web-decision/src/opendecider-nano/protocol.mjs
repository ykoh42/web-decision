import {WebDecisionError} from '../kev.js';
import {OPENDECIDER_NANO_MAX_TOKENS} from './contract.mjs';
import {OPENDECIDER_NANO_TOKENS} from './tokenizer.mjs';

const invalid = (message) => new WebDecisionError('INVALID_INPUT', message);

function assertText(value) {
  if (typeof value !== 'string') throw invalid('All request text must be strings.');
  for (let index = 0; index < value.length; index++) {
    const code = value.charCodeAt(index);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(++index);
      if (!(next >= 0xdc00 && next <= 0xdfff)) throw invalid('Text contains an unpaired UTF-16 surrogate.');
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      throw invalid('Text contains an unpaired UTF-16 surrogate.');
    }
  }
}

function encode(tokenizer, text) {
  assertText(text);
  let ids;
  try {
    ids = tokenizer.encode(text);
  } catch (cause) {
    if (cause instanceof WebDecisionError && cause.code === 'INVALID_INPUT') throw cause;
    throw new WebDecisionError('INVALID_ASSET', 'OpenDecider-nano tokenizer failed.', {cause});
  }
  if (!Array.isArray(ids) || ids.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= 50368))
    throw new WebDecisionError('INVALID_ASSET', 'OpenDecider-nano tokenizer returned invalid token IDs.');
  return ids;
}

function validateQuestion(question, index) {
  if (!question || typeof question.instructions !== 'string' || !question.instructions.trim())
    throw invalid(`Question ${index} needs non-empty instructions.`);
  if (!Array.isArray(question.options) || question.options.length < 2)
    throw invalid(`Question ${index} needs at least 2 options.`);
  if (question.options.some((option) => typeof option !== 'string' || !option.trim()))
    throw invalid(`Question ${index} has an empty or non-string option.`);
  if (new Set(question.options).size !== question.options.length)
    throw invalid(`Question ${index} has duplicate options.`);
}

/** Build OpenDecider's nano_ids protocol, truncating only the state tail. */
export function packOpenDeciderNanoRequest(tokenizer, request, maxTokens = OPENDECIDER_NANO_MAX_TOKENS) {
  if (!tokenizer || typeof tokenizer.encode !== 'function')
    throw new WebDecisionError('INVALID_ASSET', 'OpenDecider-nano tokenizer is unavailable.');
  if (!Number.isSafeInteger(maxTokens) || maxTokens !== OPENDECIDER_NANO_MAX_TOKENS)
    throw new WebDecisionError('INVALID_ASSET', 'Invalid OpenDecider-nano token limit.');
  if (!request || typeof request !== 'object' || Array.isArray(request)) throw invalid('A request is required.');
  if (!Array.isArray(request.questions) || request.questions.length < 1) throw invalid('Provide at least 1 question.');
  assertText(request.state);

  const stateIds = encode(tokenizer, `input: ${request.state}`);
  const questions = request.questions.map((question, index) => {
    validateQuestion(question, index);
    const questionIds = encode(tokenizer, `question: ${question.instructions}`);
    const optionIds = question.options.flatMap((option) => [
      OPENDECIDER_NANO_TOKENS.mask,
      ...encode(tokenizer, ` ${option}`)
    ]);
    const stateRoom = maxTokens - questionIds.length - optionIds.length - 4;
    if (stateRoom < 0) throw invalid(`Question ${index} and its options exceed ${maxTokens} tokens.`);
    const inputStateIds = stateIds.slice(0, stateRoom);
    const ids = [
      OPENDECIDER_NANO_TOKENS.cls,
      ...questionIds,
      OPENDECIDER_NANO_TOKENS.sep,
      ...optionIds,
      OPENDECIDER_NANO_TOKENS.sep,
      ...inputStateIds,
      OPENDECIDER_NANO_TOKENS.sep
    ];
    const markerPositions = [];
    for (let position = 0; position < ids.length; position++)
      if (ids[position] === OPENDECIDER_NANO_TOKENS.mask) markerPositions.push(position);
    if (markerPositions.length !== question.options.length)
      throw new WebDecisionError('INVALID_INPUT', 'OpenDecider-nano option markers do not match the options.');
    return {
      ids,
      markerPositions,
      options: [...question.options],
      truncated: stateIds.length > stateRoom
    };
  });

  return {questions, tokenCount: questions.reduce((maximum, {ids}) => Math.max(maximum, ids.length), 0)};
}

/** Preserve option order and turn the option-marker logits into probabilities. */
export function finishOpenDeciderNanoRequest(packed, rows) {
  if (
    !packed ||
    !Array.isArray(packed.questions) ||
    !Array.isArray(rows) ||
    rows.length !== packed.questions.length ||
    rows.some((row, index) => !(row instanceof Float32Array) || row.length !== packed.questions[index].options.length)
  )
    throw new WebDecisionError('INFERENCE_ERROR', 'OpenDecider-nano returned invalid option scores.');
  const answers = packed.questions.map(({options}, questionIndex) => {
    const scores = Array.from(rows[questionIndex]);
    if (!scores.every(Number.isFinite))
      throw new WebDecisionError('INFERENCE_ERROR', 'OpenDecider-nano returned non-finite scores.');
    const maximum = Math.max(...scores);
    const exps = scores.map((score) => Math.exp(score - maximum));
    const sum = exps.reduce((left, right) => left + right, 0);
    const probabilities = exps.map((value) => value / sum);
    const selectedIndex = probabilities.indexOf(Math.max(...probabilities));
    return {
      selected: options[selectedIndex],
      selectedIndex,
      probabilities: options.map((option, index) => ({option, probability: probabilities[index], logit: scores[index]}))
    };
  });
  return {
    tokenCount: packed.tokenCount,
    ...(packed.questions.some(({truncated}) => truncated) ? {truncated: true} : {}),
    answers
  };
}

export function readOpenDeciderNanoMarkerScores(ids, markerPositions, logits) {
  if (
    !Array.isArray(ids) ||
    !Array.isArray(markerPositions) ||
    markerPositions.length < 2 ||
    !(logits instanceof Float32Array) ||
    logits.length !== ids.length ||
    markerPositions.some(
      (position, index) =>
        !Number.isSafeInteger(position) ||
        position < 0 ||
        position >= ids.length ||
        ids[position] !== OPENDECIDER_NANO_TOKENS.mask ||
        (index > 0 && markerPositions[index - 1] >= position)
    )
  )
    throw new WebDecisionError('INFERENCE_ERROR', 'OpenDecider-nano returned an invalid marker layout.');
  return Float32Array.from(markerPositions, (position) => logits[position]);
}
