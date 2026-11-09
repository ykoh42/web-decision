// Private Nari v3.2 choice adapter. No package route, model loader or GPU backend.
import {WebDecisionError} from '../kev.js';
import {decodeNariChoice, packNariRequest} from './protocol.mjs';

interface NariInput {
  readonly inputIds: readonly number[];
  readonly optPos: readonly number[];
  readonly optQ: readonly number[];
  readonly optType: readonly number[];
  readonly slotPos: readonly number[];
  readonly nLetters: readonly number[];
  readonly options: readonly (readonly string[])[];
}
interface NariRequest {
  state: string;
  questions: {type?: unknown; criteria?: unknown; instructions: unknown; options: string[]}[];
}
interface NariOutput {
  headScores: Float32Array;
  letterScores?: Float32Array | null;
}

const TEMPERATURES = Object.freeze({'choice:2': 1, 'choice:3-5': 1.1189, 'choice:6+': 0.9632, 'choice:27+': 1.1189});
const invalid = (message: string, cause?: unknown) =>
  new WebDecisionError('INVALID_INPUT', message, cause === undefined ? undefined : {cause});
const inference = (message: string) => new WebDecisionError('INFERENCE_ERROR', message);

/** Preserve caller-owned strings/options before any asynchronous work. */
export function pack(tokenizer: unknown, request: unknown, controlIds: unknown) {
  const requestValue = request as NariRequest;
  if (
    !request ||
    typeof request !== 'object' ||
    Array.isArray(request) ||
    typeof requestValue.state !== 'string' ||
    !Array.isArray(requestValue.questions) ||
    !requestValue.questions.length
  )
    throw invalid('Provide a state and at least one choice question.');
  const snapshot = {
    state: requestValue.state,
    questions: requestValue.questions.map((q) => {
      if (!q || typeof q !== 'object' || !Array.isArray(q.options)) throw invalid('Invalid choice question.');
      if ((q.type !== undefined && q.type !== 'choice') || q.criteria !== undefined)
        throw invalid('Only string choice questions are supported.');
      return {instructions: q.instructions, options: [...q.options]};
    })
  };
  let value;
  try {
    value = packNariRequest(tokenizer as Parameters<typeof packNariRequest>[0], snapshot, controlIds);
  } catch (cause) {
    throw invalid('Could not encode a Nari choice request.', cause);
  }
  validatePacked(value);
  return Object.freeze({
    ...value,
    inputIds: Object.freeze([...value.inputIds]),
    optPos: Object.freeze([...value.optPos]),
    optQ: Object.freeze([...value.optQ]),
    optType: Object.freeze([...value.optType]),
    slotPos: Object.freeze([...value.slotPos]),
    nLetters: Object.freeze([...value.nLetters]),
    options: Object.freeze(value.options.map((items) => Object.freeze([...items])))
  });
}

function validateGroup(
  value: NariInput,
  options: readonly string[],
  question: number,
  n: number,
  offset: number,
  previous: number
) {
  if (
    !Array.isArray(options) ||
    options.length < 2 ||
    options.length > 255 ||
    Array.from(options).some((option) => typeof option !== 'string' || !option.trim()) ||
    new Set(options).size !== options.length
  )
    throw invalid('Invalid joint options.');
  const count = options.length,
    letters = count <= 26 ? count : 0,
    slot = value.slotPos[question]!;
  if (
    value.nLetters[question] !== letters ||
    !Number.isSafeInteger(slot) ||
    (letters ? slot < 0 || slot >= n : slot !== -1)
  )
    throw invalid('Invalid joint letter position.');
  for (let i = 0; i < count; i++) {
    const index = offset + i,
      position = value.optPos[index]!;
    if (
      value.optQ[index] !== question ||
      value.optType[index] !== 1 ||
      !Number.isSafeInteger(position) ||
      position <= previous ||
      position >= n
    )
      throw invalid('Invalid joint option position or owner.');
    previous = position;
  }
  if (letters) {
    if (slot <= previous) throw invalid('Letter position precedes its options.');
    previous = slot;
  }
  return previous;
}

/** A single joint request is required: head self-attention crosses questions. */
export function validatePacked(value: NariInput) {
  const n = value?.inputIds?.length,
    groups = value?.options;
  if (
    !Array.isArray(value?.inputIds) ||
    !Number.isSafeInteger(n) ||
    n < 1 ||
    n > 2048 ||
    Array.from(value.inputIds).some((id) => !Number.isSafeInteger(id) || id < 0 || id >= 248320) ||
    !Array.isArray(groups) ||
    groups.length < 1 ||
    !Array.isArray(value.optPos) ||
    !Array.isArray(value.optQ) ||
    !Array.isArray(value.optType) ||
    !Array.isArray(value.slotPos) ||
    !Array.isArray(value.nLetters) ||
    value.slotPos.length !== groups.length ||
    value.nLetters.length !== groups.length
  )
    throw invalid('Invalid joint Nari input.');
  const total = groups.reduce((sum, items) => sum + (Array.isArray(items) ? items.length : 0), 0);
  if (value.optPos.length !== total || value.optQ.length !== total || value.optType.length !== total)
    throw invalid('Invalid joint option count.');
  let offset = 0,
    previous = -1;
  for (const [question, options] of groups.entries()) {
    previous = validateGroup(value, options, question, n, offset, previous);
    offset += options.length;
  }
  return total;
}

/** Decode flattened FP32 [1,totalOptions] and optional [1,questions,26] output. */
export function unpack(value: NariInput, output: unknown) {
  const outputValue = output as NariOutput;
  const total = validatePacked(value),
    questions = value.options.length;
  if (
    !(outputValue?.headScores instanceof Float32Array) ||
    outputValue.headScores.length !== total ||
    outputValue.headScores.some((number) => !Number.isFinite(number))
  )
    throw inference('Invalid Nari head output.');
  const needsLetters = value.nLetters.some(Boolean),
    letters = outputValue.letterScores;
  if (
    (needsLetters && !(letters instanceof Float32Array)) ||
    (letters != null && (!(letters instanceof Float32Array) || letters.length !== questions * 26))
  )
    throw inference('Invalid Nari letter output.');
  const head = Float32Array.from(outputValue.headScores),
    copiedLetters = letters == null ? null : Float32Array.from(letters);
  const answers = [];
  let offset = 0;
  for (const [question, options] of value.options.entries()) {
    const count = options.length,
      letterCount = value.nLetters[question];
    const optionHead = head.subarray(offset, offset + count);
    const optionLetters = letterCount ? copiedLetters!.subarray(question * 26, question * 26 + count) : null;
    if (optionLetters?.some((number) => !Number.isFinite(number))) throw inference('Invalid Nari letter scores.');
    answers.push(decodeNariChoice(options, optionHead, optionLetters, TEMPERATURES, 0.5));
    offset += count;
  }
  return {tokenCount: value.inputIds.length, answers};
}
