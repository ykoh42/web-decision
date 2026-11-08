import {WebDecisionError} from '../kev.js';

const SYSTEM =
  'You are a decision engine. You read a record and answer one question about it by choosing exactly one option. Reply with a single letter.';
const LOG_FLOOR = Math.log(1e-12);
export const MAX_H2O_OPTIONS = 255;
const PY_WHITESPACE = new Set([0x85, 0xa0, 0x1680, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000]);

function isPythonWhitespace(code) {
  return (
    (code >= 9 && code <= 13) ||
    (code >= 28 && code <= 32) ||
    (code >= 0x2000 && code <= 0x200a) ||
    PY_WHITESPACE.has(code)
  );
}

function pythonStrip(value, trailing = true) {
  let first = 0,
    last = value.length;
  while (first < last && isPythonWhitespace(value.charCodeAt(first))) first++;
  while (trailing && last > first && isPythonWhitespace(value.charCodeAt(last - 1))) last--;
  return value.slice(first, last);
}

function invalid(message) {
  return new WebDecisionError('INVALID_INPUT', message);
}

function safeText(value, reserved) {
  if (typeof value !== 'string') throw invalid('Text must be a string.');
  for (let i = 0; i < value.length; i++) {
    const code = value.charCodeAt(i);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(++i);
      if (!(next >= 0xdc00 && next <= 0xdfff)) throw invalid('Text contains an unpaired UTF-16 surrogate.');
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      throw invalid('Text contains an unpaired UTF-16 surrogate.');
    }
  }
  if (reserved.some((token) => value.includes(token))) throw invalid('Text contains a reserved token.');
  return value;
}

function validateQuestion(question, index) {
  if (!question || typeof question.instructions !== 'string' || !question.instructions.trim())
    throw invalid(`Question ${index} needs instructions.`);
  if (!Array.isArray(question.options) || question.options.length < 2 || question.options.length > MAX_H2O_OPTIONS)
    throw invalid(`Question ${index} needs 2 to ${MAX_H2O_OPTIONS} options.`);
  if (question.options.some((option) => typeof option !== 'string' || !option.trim()))
    throw invalid(`Question ${index} has an empty option.`);
  if (new Set(question.options).size !== question.options.length)
    throw invalid(`Question ${index} has duplicate options.`);
}

function renderQuestion(tokenizer, state, question, index, config) {
  validateQuestion(question, index);
  const instructions = pythonStrip(safeText(question.instructions, config.addedTokens), false);
  const options = question.options.map((value) => safeText(value, config.addedTokens));
  const labels = config.labels.slice(0, options.length);
  const lines = options.map((option, i) => `${labels[i]}) ${option}: ${option}`);
  const user = pythonStrip(`record: ${state}\nquestion: ${instructions}\noptions:\n${lines.join('\n')}`);
  const prompt = `<|im_start|>system\n${SYSTEM}<|im_end|>\n<|im_start|>user\n${user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:`;
  const encoded = tokenizer.encode(prompt, {add_special_tokens: false}).ids;
  if (!Array.isArray(encoded)) throw invalid('Invalid token IDs.');
  // Tokenizers may reuse their output buffer. Each question and queued request
  // must keep the tokens produced for its own prompt.
  const ids = Array.from(encoded);
  if (ids.length > config.maxInputTokens) throw invalid('Packed input exceeds the token limit.');
  if (ids.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= 2 ** 31)) throw invalid('Invalid token IDs.');
  return {prompt, ids, labels, options: [...options], labelIds: config.labelIds.slice(0, options.length)};
}

/** H2O reads each question as a separate complete prompt and forward pass. */
export function packH2oRequest(tokenizer, request, config) {
  if (!request || typeof request !== 'object' || Array.isArray(request)) throw invalid('A request is required.');
  if (!Array.isArray(request.questions) || !request.questions.length) throw invalid('Provide at least one question.');
  if (
    !Array.isArray(config.labels) ||
    config.labels.length !== 255 ||
    new Set(config.labels).size !== 255 ||
    !Array.isArray(config.labelIds) ||
    config.labelIds.length !== 255 ||
    new Set(config.labelIds).size !== 255
  )
    throw invalid('Invalid answer labels.');
  if (!Array.isArray(config.addedTokens) || config.addedTokens.some((token) => typeof token !== 'string' || !token))
    throw invalid('Invalid reserved-token list.');
  if (!Number.isSafeInteger(config.maxInputTokens) || config.maxInputTokens <= 0) throw invalid('Invalid token limit.');
  const state = safeText(request.state, config.addedTokens);
  return {questions: request.questions.map((question, i) => renderQuestion(tokenizer, state, question, i, config))};
}

/** Match the upstream pooled readout's log-probability floor and temperature. */
export function unpackH2oScores(rows, packed, temperature) {
  if (!Number.isFinite(temperature) || temperature <= 0) throw invalid('Invalid temperature.');
  if (
    !Array.isArray(rows) ||
    rows.length !== packed.questions.length ||
    rows.some((row) => !Array.isArray(row) || row.length !== 255 || row.some((value) => !Number.isFinite(value)))
  )
    throw new WebDecisionError('INFERENCE_ERROR', 'The model returned invalid scores.');
  return {
    tokenCount: packed.questions.reduce((total, question) => total + question.ids.length, 0),
    answers: packed.questions.map(({options}, i) => {
      const logits = rows[i].slice(0, options.length),
        maximum = Math.max(...logits);
      const z = maximum + Math.log(logits.reduce((total, value) => total + Math.exp(value - maximum), 0));
      const logprobs = logits.map((value) => Math.max(value - z, LOG_FLOOR));
      const peak = Math.max(...logprobs),
        exps = logprobs.map((value) => Math.exp((value - peak) / temperature));
      const sum = exps.reduce((total, value) => total + value, 0),
        probabilities = exps.map((value) => value / sum);
      const selectedIndex = probabilities.indexOf(Math.max(...probabilities));
      return {
        selected: options[selectedIndex],
        selectedIndex,
        probabilities: options.map((option, j) => ({option, probability: probabilities[j], logit: logits[j]}))
      };
    })
  };
}
