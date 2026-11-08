import {encodeRequest} from './engine.js';
import {TINYDECIDE_MAX_CHOICES, TINYDECIDE_QUESTIONS_PER_CHUNK} from './contract.mjs';
import {WebDecisionError} from '../kev.js';

const invalid = (message) => new WebDecisionError('INVALID_INPUT', message);
const questionTypes = new Set(['choice', 'noul', 'score', 'span']);

function readObject(value, allowed, message) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw invalid(message);
  const result = Object.create(null);
  for (const key of Reflect.ownKeys(value)) {
    const field = Object.getOwnPropertyDescriptor(value, key);
    if (typeof key !== 'string' || !allowed.has(key) || !field || !Object.hasOwn(field, 'value'))
      throw invalid(message);
    result[key] = field.value;
  }
  for (const key in value) if (!allowed.has(key)) throw invalid(message);
  return result;
}

function readOptions(options, type, index) {
  if (options === undefined && type !== 'choice' && type !== 'score') return [];
  if (!Array.isArray(options)) throw invalid(`Question ${index} options must be an array.`);
  const count = options.length;
  if (count > TINYDECIDE_MAX_CHOICES)
    throw invalid(`Question ${index} accepts at most ${TINYDECIDE_MAX_CHOICES} options.`);
  const values = new Array(count);
  for (let i = 0; i < count; i++) {
    const item = Object.getOwnPropertyDescriptor(options, String(i));
    if (!item || !Object.hasOwn(item, 'value') || typeof item.value !== 'string')
      throw invalid(`Question ${index} options must be strings.`);
    values[i] = item.value;
  }
  if (type === 'choice' || type === 'score') {
    if (count < 2) throw invalid(`Question ${index} needs at least 2 options.`);
  } else if (count > 0) {
    throw invalid(`Question ${index} type ${type} does not accept options.`);
  }
  return values;
}

function readQuestion(question, index) {
  const fields = readObject(question, new Set(['type', 'text', 'options']), `Question ${index} is invalid.`);
  if (!questionTypes.has(fields.type)) throw invalid(`Question ${index} has an unsupported type.`);
  if (typeof fields.text !== 'string') throw invalid(`Question ${index} text must be a string.`);
  const options = readOptions(fields.options, fields.type, index);
  return {type: fields.type, text: fields.text, options};
}

export function validateTinyDecideRequest(request) {
  const fields = readObject(
    request,
    new Set(['state', 'questions']),
    'A TinyDecide request needs state and questions.'
  );
  if (typeof fields.state !== 'string') throw invalid('TinyDecide state must be a string.');
  if (!Array.isArray(fields.questions)) throw invalid('TinyDecide questions must be an array.');
  if (fields.questions.length < 1) throw invalid('TinyDecide needs at least one question.');
  const questions = new Array(fields.questions.length);
  for (let i = 0; i < questions.length; i++) {
    if (!Object.hasOwn(fields.questions, i)) throw invalid(`Question ${i + 1} is missing.`);
    questions[i] = readQuestion(fields.questions[i], i + 1);
  }
  return {state: fields.state, questions};
}

export function validateTinyDecideSystemOne(state, questions) {
  if (typeof state !== 'string') throw invalid('TinyDecide state must be a string.');
  if (!questions || typeof questions !== 'object' || Array.isArray(questions))
    throw invalid('TinyDecide questions must be a named object.');
  const entries = Reflect.ownKeys(questions);
  if (entries.length < 1) throw invalid('TinyDecide needs at least one named question.');
  if (entries.some((key) => typeof key !== 'string')) throw invalid('TinyDecide question names must be strings.');
  const names = [];
  const native = [];
  for (const name of entries) {
    const descriptor = Object.getOwnPropertyDescriptor(questions, name);
    if (!descriptor || !Object.hasOwn(descriptor, 'value')) throw invalid(`Question ${name} must be a data property.`);
    const fields = readObject(
      descriptor.value,
      new Set(['type', 'instructions', 'options']),
      `Question ${name} has unsupported fields.`
    );
    if (!questionTypes.has(fields.type)) throw invalid(`Question ${name} has an unsupported type.`);
    if (typeof fields.instructions !== 'string') throw invalid(`Question ${name} instructions must be a string.`);
    native.push(readQuestion({type: fields.type, text: fields.instructions, options: fields.options}, name));
    names.push(name);
  }
  return {state, names, request: {state, questions: native}};
}

export function validateTinyDecideRows(engine, request) {
  for (let start = 0; start < request.questions.length; start += TINYDECIDE_QUESTIONS_PER_CHUNK) {
    const questions = request.questions.slice(start, start + TINYDECIDE_QUESTIONS_PER_CHUNK);
    try {
      encodeRequest(engine.tok, engine.meta, {state: request.state, questions});
    } catch (cause) {
      if (cause instanceof WebDecisionError) throw cause;
      throw invalid(cause instanceof Error ? cause.message : 'TinyDecide could not encode a question chunk.');
    }
  }
}

function validatePrototype(value, question, index, dimensions) {
  const fields = readObject(value, new Set(['vec', 'cnt', 'center', 'lam']), `Prototype ${index + 1} is invalid.`);
  const count = question.type === 'noul' ? 2 : question.options.length;
  if (
    !(fields.vec instanceof Float32Array) ||
    fields.vec.length !== count * dimensions ||
    !(fields.cnt instanceof Int32Array) ||
    fields.cnt.length !== count ||
    (fields.center != null && (!(fields.center instanceof Float32Array) || fields.center.length !== dimensions)) ||
    (fields.lam !== undefined && !Number.isFinite(fields.lam))
  )
    throw invalid(`Prototype ${index + 1} does not match its question.`);
  if (
    Array.from(fields.vec).some((value) => !Number.isFinite(value)) ||
    Array.from(fields.center ?? []).some((value) => !Number.isFinite(value)) ||
    Array.from(fields.cnt).some((value) => value < 0)
  )
    throw invalid(`Prototype ${index + 1} contains invalid values.`);
  return {
    vec: new Float32Array(fields.vec),
    cnt: new Int32Array(fields.cnt),
    center: fields.center == null ? null : new Float32Array(fields.center),
    ...(fields.lam === undefined ? {} : {lam: fields.lam})
  };
}

function validatePrototypes(value, request, dimensions) {
  if (value === undefined) return undefined;
  if (!Array.isArray(value) || value.length > request.questions.length)
    throw invalid('TinyDecide prototypes must match question order.');
  const prototypes = new Array(request.questions.length);
  for (let index = 0; index < value.length; index++) {
    const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
    if (!descriptor) continue;
    if (!Object.hasOwn(descriptor, 'value')) throw invalid(`Prototype ${index + 1} must be a data property.`);
    const prototype = descriptor.value;
    if (prototype == null) {
      prototypes[index] = prototype;
      continue;
    }
    const question = request.questions[index];
    if (question.type === 'span') throw invalid(`Span question ${index + 1} cannot use prototypes.`);
    prototypes[index] = validatePrototype(prototype, question, index, dimensions);
  }
  return prototypes;
}

export function validateTinyDecideDecisionOptions(options, request, dimensions) {
  if (options == null) return undefined;
  const fields = readObject(options, new Set(['signal', 'protos']), 'Unsupported TinyDecide decision option.');
  if (fields.signal != null && typeof fields.signal.throwIfAborted !== 'function')
    throw invalid('TinyDecide signal must be an AbortSignal.');
  return {
    signal: fields.signal,
    protos: request ? validatePrototypes(fields.protos, request, dimensions) : undefined
  };
}
