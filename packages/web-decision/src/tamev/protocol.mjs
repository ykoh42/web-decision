import {WebDecisionError} from '../kev.js';
import {TAMEV_MICRO_PROFILE} from './contract.mjs';

const invalid = (message) => new WebDecisionError('INVALID_INPUT', message);

function sum(values) {
  return values.reduce((total, value) => total + value, 0);
}

function firstMaximumIndex(values) {
  let index = 0;
  for (let candidate = 1; candidate < values.length; candidate++) {
    if (values[candidate] > values[index]) index = candidate;
  }
  return index;
}

export function prepareTamevDecision(tokenizer, request, profile = TAMEV_MICRO_PROFILE) {
  if (typeof request?.state !== 'string' || !Array.isArray(request.questions))
    throw invalid(`${profile.name} requires a string state and a question list.`);
  if (request.questions.length < 1) throw invalid(`${profile.name} needs at least one question.`);

  const questions = request.questions.map((question, index) => {
    if (typeof question.instructions !== 'string' || !Array.isArray(question.options))
      throw invalid(`${profile.name} question ${index} requires instructions and options.`);
    if (question.options.length < 2) throw invalid(`${profile.name} question ${index} needs at least 2 options.`);
    if (question.options.some((option) => typeof option !== 'string' || !option.trim()))
      throw invalid(`${profile.name} question ${index} has an empty or non-text option.`);

    const contextText = question.instructions ? `${request.state}\n${question.instructions}`.trim() : request.state;
    const context = tokenizer.encodeContext(contextText);
    const options = question.options.map((option) => tokenizer.encodeOption(option));
    return {
      options: [...question.options],
      contextIds: context.ids,
      optionRows: options.map(({ids}) => ids),
      tokenCount: context.ids.length + options.reduce((count, option) => count + option.ids.length, 0),
      truncated: context.truncated || options.some((option) => option.truncated)
    };
  });

  return {
    questions,
    tokenCount: questions.reduce((count, question) => count + question.tokenCount, 0),
    truncated: questions.some((question) => question.truncated)
  };
}

export function calibrateTamevProbabilities(probabilities, optionCount, config) {
  const delta = config.calibration_by_k[String(optionCount)];
  if (delta === undefined || delta === 1) return [...probabilities];
  const adjusted = probabilities.map((probability) => Math.max(probability, 1e-12) ** delta);
  const total = sum(adjusted);
  return adjusted.map((probability) => probability / total);
}

export function finishTamevDecision(prepared, rows) {
  if (rows.length !== prepared.questions.length) throw new TypeError('TAMEV Micro result rows are incomplete.');
  const answers = prepared.questions.map((question, questionIndex) => {
    const row = rows[questionIndex];
    const selectedIndex = firstMaximumIndex(row.probabilities);
    return {
      selected: question.options[selectedIndex],
      selectedIndex,
      probabilities: question.options.map((option, index) => ({
        option,
        probability: row.probabilities[index],
        logit: row.logits[index]
      }))
    };
  });
  return {
    tokenCount: prepared.tokenCount,
    ...(prepared.truncated ? {truncated: true} : {}),
    answers
  };
}
