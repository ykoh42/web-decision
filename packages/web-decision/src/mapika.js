// Text-choice packing follows Mapika/decider prompt.py, Apache-2.0,
// commit 91b6fa9b94ab6c4dc2c63d3b14d413243b4fffcc. Only the joint,
// state-first text path is implemented here.
import {WebDecisionError} from './kev.js';

export const MAPIKA_MAX_OPTIONS = 255;

function inputError(message) {
  return new WebDecisionError('INVALID_INPUT', message);
}

function assetError(message) {
  return new WebDecisionError('INVALID_ASSET', message);
}

function encode(tokenizer, text) {
  const values = tokenizer.encode(text, {add_special_tokens: false})?.ids;
  if (!values || typeof values[Symbol.iterator] !== 'function') throw assetError('Invalid tokenizer output.');
  const tokens = Array.from(values);
  if (tokens.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= 2 ** 31))
    throw assetError('Tokenizer returned an unsupported token ID.');
  return tokens;
}

/** Build the first 255 single-token uppercase answer codes, as upstream does. */
export function buildMapikaLabels(tokenizer) {
  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  const candidates = [...alphabet];
  for (const first of alphabet) for (const second of alphabet) candidates.push(first + second);
  const names = [],
    ids = [];
  for (const candidate of candidates) {
    const tokens = encode(tokenizer, candidate);
    if (tokens.length === 1) {
      names.push(candidate);
      ids.push(tokens[0]);
      if (ids.length === MAPIKA_MAX_OPTIONS) break;
    }
  }
  const open = encode(tokenizer, '\n(');
  if (ids.length !== MAPIKA_MAX_OPTIONS || new Set(ids).size !== MAPIKA_MAX_OPTIONS || !open.length)
    throw assetError('Tokenizer does not provide the expected Mapika answer labels.');
  return {names, ids, open};
}

function validateText(tokenizer, value, controls) {
  if (typeof value !== 'string') throw inputError('All request text must be a string.');
  // Do not let the tokenizer silently replace an isolated UTF-16 surrogate.
  for (let i = 0; i < value.length; i++) {
    const code = value.charCodeAt(i);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(++i);
      if (!(next >= 0xdc00 && next <= 0xdfff)) throw inputError('Text contains an unpaired UTF-16 surrogate.');
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      throw inputError('Text contains an unpaired UTF-16 surrogate.');
    }
  }
  if (encode(tokenizer, value).some((id) => controls.has(id)))
    throw inputError('Request text contains a model control token.');
}

function append(ids, tokens, maxTokens) {
  if (ids.length + tokens.length > maxTokens)
    throw inputError(`Packed input exceeds ${maxTokens} tokens; no truncation was applied.`);
  for (const token of tokens) ids.push(token);
}

function validateQuestion(tokenizer, question, controls, index) {
  if (!question || typeof question.instructions !== 'string' || !question.instructions.trim())
    throw inputError(`Question ${index} needs non-empty instructions.`);
  if (!Array.isArray(question.options) || question.options.length < 2 || question.options.length > MAPIKA_MAX_OPTIONS)
    throw inputError(`Question ${index} needs 2 to 255 options.`);
  if (question.options.some((option) => typeof option !== 'string' || !option.trim()))
    throw inputError(`Question ${index} has an empty or non-string option.`);
  if (new Set(question.options).size !== question.options.length)
    throw inputError(`Question ${index} has duplicate options.`);
  validateText(tokenizer, question.instructions, controls);
  for (const option of question.options) validateText(tokenizer, option, controls);
}

function appendQuestion(ids, tokenizer, labels, question, index, count, maxTokens) {
  const number = count > 1 ? ` ${index + 1}` : '';
  const head = `\n\nQuestion${number}: ${question.instructions}\nOptions:`;
  const tail = `\nAnswer${number}: (`;
  if (question.options.length <= 10) {
    const lines = question.options.map((option, j) => `\n(${labels.names[j]}) ${option}`).join('');
    append(ids, encode(tokenizer, head + lines + tail), maxTokens);
    return;
  }
  append(ids, encode(tokenizer, head), maxTokens);
  for (const [j, option] of question.options.entries()) {
    append(ids, labels.open, maxTokens);
    append(ids, [labels.ids[j]], maxTokens);
    append(ids, encode(tokenizer, `) ${option}`), maxTokens);
  }
  append(ids, encode(tokenizer, tail), maxTokens);
}

function validateConfig(labels, limits) {
  if (
    !labels ||
    labels.names?.length !== MAPIKA_MAX_OPTIONS ||
    labels.ids?.length !== MAPIKA_MAX_OPTIONS ||
    !Array.isArray(labels.open) ||
    !labels.open.length
  )
    throw assetError('Invalid Mapika answer labels.');
  if (
    !limits ||
    !Number.isSafeInteger(limits.maxStateTokens) ||
    limits.maxStateTokens < 1 ||
    !Number.isSafeInteger(limits.maxInputTokens) ||
    limits.maxInputTokens < 1 ||
    !Array.isArray(limits.controlIds) ||
    !limits.controlIds.length ||
    limits.controlIds.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= 2 ** 31) ||
    new Set(limits.controlIds).size !== limits.controlIds.length
  )
    throw assetError('Invalid Mapika model limits or control IDs.');
}

/** Pack the upstream joint text-choice path without truncation. */
export function packMapikaRequest(tokenizer, labels, request, limits) {
  validateConfig(labels, limits);
  if (!request || typeof request !== 'object' || Array.isArray(request))
    throw inputError('A request object is required.');
  if (!Array.isArray(request.questions) || !request.questions.length)
    throw inputError('Provide at least one question.');
  const controls = new Set(limits.controlIds);
  validateText(tokenizer, request.state, controls);
  const state = encode(tokenizer, `Context:\n${request.state}`);
  if (state.length > limits.maxStateTokens) throw inputError(`The state exceeds ${limits.maxStateTokens} tokens.`);
  const ids = [],
    groups = [];
  append(ids, state, limits.maxInputTokens);
  for (const [index, question] of request.questions.entries()) {
    validateQuestion(tokenizer, question, controls, index);
    appendQuestion(ids, tokenizer, labels, question, index, request.questions.length, limits.maxInputTokens);
    groups.push({slot: ids.length - 1, options: [...question.options]});
  }
  return {ids, groups};
}

/** Convert one 255-label row per question into ordered, temperature-adjusted scores. */
export function unpackMapikaScores(rows, packed, temperature) {
  const isRow = (row) => Array.isArray(row) || (ArrayBuffer.isView(row) && !(row instanceof DataView));
  if (
    !Number.isFinite(temperature) ||
    temperature <= 0 ||
    !Array.isArray(rows) ||
    rows.length !== packed.groups.length ||
    rows.some((row) => !isRow(row) || row.length !== MAPIKA_MAX_OPTIONS || row.some((value) => !Number.isFinite(value)))
  )
    throw new WebDecisionError('INFERENCE_ERROR', 'The model returned invalid scores.');
  return {
    tokenCount: packed.ids.length,
    answers: packed.groups.map(({options}, i) => {
      const logits = Array.from(rows[i].slice(0, options.length));
      const max = Math.max(...logits);
      const exps = logits.map((value) => Math.exp((value - max) / temperature));
      const total = exps.reduce((a, b) => a + b, 0);
      const probabilities = exps.map((value) => value / total);
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
