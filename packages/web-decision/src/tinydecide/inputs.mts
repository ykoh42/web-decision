import {encodeRequest} from './engine.js';
import {TINYDECIDE_MAX_CHOICES, TINYDECIDE_QUESTIONS_PER_CHUNK} from './contract.mjs';
import {WebDecisionError} from '../kev.js';
import type {TinyDecide, TinyDecidePrototype, TinyDecideRequest} from './engine.js';

type TinyDecideQuestionType = 'choice' | 'noul' | 'score' | 'span';
type TinyDecideAbortSignal = Pick<AbortSignal, 'throwIfAborted'>;

type TinyDecideQuestion = TinyDecideRequest['questions'][number];

interface TinyDecideSystemOneRequest {
  readonly state: string;
  readonly names: string[];
  readonly request: TinyDecideRequest;
}

export interface TinyDecideDecisionOptions {
  readonly signal: TinyDecideAbortSignal | null | undefined;
  readonly protos: readonly (TinyDecidePrototype | null | undefined)[] | undefined;
}

const invalid = (message: string): WebDecisionError => new WebDecisionError('INVALID_INPUT', message);
const questionTypes = new Set<TinyDecideQuestionType>(['choice', 'noul', 'score', 'span']);

function readObject(value: unknown, allowed: ReadonlySet<string>, message: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw invalid(message);
  const result: Record<string, unknown> = Object.create(null);
  for (const key of Reflect.ownKeys(value as object)) {
    const field = Object.getOwnPropertyDescriptor(value as object, key);
    if (typeof key !== 'string' || !allowed.has(key) || !field || !Object.hasOwn(field, 'value'))
      throw invalid(message);
    result[key as string] = (field as PropertyDescriptor).value;
  }
  for (const key in value as object) if (!allowed.has(key)) throw invalid(message);
  return result;
}

function readOptions(options: unknown, type: TinyDecideQuestionType, index: string | number): string[] {
  if (options === undefined && type !== 'choice' && type !== 'score') return [];
  if (!Array.isArray(options)) throw invalid(`Question ${index} options must be an array.`);
  const count = (options as unknown[]).length;
  if (count > TINYDECIDE_MAX_CHOICES)
    throw invalid(`Question ${index} accepts at most ${TINYDECIDE_MAX_CHOICES} options.`);
  // oxlint-disable-next-line unicorn/no-new-array -- Preserve the original Array constructor and sparse allocation.
  const values: string[] = new Array(count);
  for (let i = 0; i < count; i++) {
    const item = Object.getOwnPropertyDescriptor(options as unknown[], String(i));
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

function readQuestion(question: unknown, index: string | number): TinyDecideQuestion {
  const fields = readObject(question, new Set(['type', 'text', 'options']), `Question ${index} is invalid.`);
  if (!questionTypes.has(fields.type as TinyDecideQuestionType))
    throw invalid(`Question ${index} has an unsupported type.`);
  if (typeof fields.text !== 'string') throw invalid(`Question ${index} text must be a string.`);
  const options = readOptions(fields.options, fields.type as TinyDecideQuestionType, index);
  return {type: fields.type as TinyDecideQuestionType, text: fields.text as string, options};
}

export function validateTinyDecideRequest(request: unknown): TinyDecideRequest {
  const fields = readObject(
    request,
    new Set(['state', 'questions']),
    'A TinyDecide request needs state and questions.'
  );
  if (typeof fields.state !== 'string') throw invalid('TinyDecide state must be a string.');
  if (!Array.isArray(fields.questions)) throw invalid('TinyDecide questions must be an array.');
  if ((fields.questions as unknown[]).length < 1) throw invalid('TinyDecide needs at least one question.');
  // oxlint-disable-next-line unicorn/no-new-array -- Preserve the original Array constructor and sparse allocation.
  const questions: TinyDecideQuestion[] = new Array((fields.questions as unknown[]).length);
  for (let i = 0; i < questions.length; i++) {
    if (!Object.hasOwn(fields.questions as unknown[], i)) throw invalid(`Question ${i + 1} is missing.`);
    questions[i] = readQuestion((fields.questions as unknown[])[i], i + 1);
  }
  return {state: fields.state as string, questions};
}

export function validateTinyDecideSystemOne(state: unknown, questions: unknown): TinyDecideSystemOneRequest {
  if (typeof state !== 'string') throw invalid('TinyDecide state must be a string.');
  if (!questions || typeof questions !== 'object' || Array.isArray(questions))
    throw invalid('TinyDecide questions must be a named object.');
  const entries = Reflect.ownKeys(questions as object);
  if (entries.length < 1) throw invalid('TinyDecide needs at least one named question.');
  if (entries.some((key) => typeof key !== 'string')) throw invalid('TinyDecide question names must be strings.');
  const names: string[] = [];
  const native: TinyDecideQuestion[] = [];
  for (const name of entries) {
    const descriptor = Object.getOwnPropertyDescriptor(questions as object, name);
    if (!descriptor || !Object.hasOwn(descriptor, 'value'))
      throw invalid(`Question ${name as string} must be a data property.`);
    const fields = readObject(
      (descriptor as PropertyDescriptor).value,
      new Set(['type', 'instructions', 'options']),
      `Question ${name as string} has unsupported fields.`
    );
    if (!questionTypes.has(fields.type as TinyDecideQuestionType))
      throw invalid(`Question ${name as string} has an unsupported type.`);
    if (typeof fields.instructions !== 'string')
      throw invalid(`Question ${name as string} instructions must be a string.`);
    native.push(readQuestion({type: fields.type, text: fields.instructions, options: fields.options}, name as string));
    names.push(name as string);
  }
  return {state, names, request: {state, questions: native}};
}

export function validateTinyDecideRows(engine: Pick<TinyDecide, 'tok' | 'meta'>, request: TinyDecideRequest): void {
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

function validatePrototype(
  value: unknown,
  question: TinyDecideQuestion,
  index: number,
  dimensions: number
): TinyDecidePrototype {
  const fields = readObject(value, new Set(['vec', 'cnt', 'center', 'lam']), `Prototype ${index + 1} is invalid.`);
  const count = question.type === 'noul' ? 2 : question.options!.length;
  if (
    !(fields.vec instanceof Float32Array) ||
    fields.vec.length !== count * dimensions ||
    !(fields.cnt instanceof Int32Array) ||
    fields.cnt.length !== count ||
    (fields.center != null &&
      (!(fields.center instanceof Float32Array) || (fields.center as Float32Array).length !== dimensions)) ||
    (fields.lam !== undefined && !Number.isFinite(fields.lam))
  )
    throw invalid(`Prototype ${index + 1} does not match its question.`);
  if (
    Array.from(fields.vec as Float32Array).some((value: number) => !Number.isFinite(value)) ||
    Array.from((fields.center as Float32Array | null | undefined) ?? []).some(
      (value: number) => !Number.isFinite(value)
    ) ||
    Array.from(fields.cnt as Int32Array).some((value: number) => value < 0)
  )
    throw invalid(`Prototype ${index + 1} contains invalid values.`);
  return {
    vec: new Float32Array(fields.vec as Float32Array),
    cnt: new Int32Array(fields.cnt as Int32Array),
    center: fields.center == null ? null : new Float32Array(fields.center as Float32Array),
    ...(fields.lam === undefined ? {} : {lam: fields.lam as number})
  };
}

function validatePrototypes(
  value: unknown,
  request: TinyDecideRequest,
  dimensions: number
): readonly (TinyDecidePrototype | null | undefined)[] | undefined {
  if (value === undefined) return undefined;
  if (!Array.isArray(value) || value.length > request.questions.length)
    throw invalid('TinyDecide prototypes must match question order.');
  // oxlint-disable-next-line unicorn/no-new-array -- Preserve holes for prototype entries omitted by the caller.
  const prototypes: Array<TinyDecidePrototype | null | undefined> = new Array(request.questions.length);
  for (let index = 0; index < value.length; index++) {
    const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
    if (!descriptor) continue;
    if (!Object.hasOwn(descriptor, 'value')) throw invalid(`Prototype ${index + 1} must be a data property.`);
    const prototype: unknown = (descriptor as PropertyDescriptor).value;
    if (prototype == null) {
      prototypes[index] = prototype;
      continue;
    }
    const question = request.questions[index]!;
    if (question.type === 'span') throw invalid(`Span question ${index + 1} cannot use prototypes.`);
    prototypes[index] = validatePrototype(prototype, question, index, dimensions);
  }
  return prototypes;
}

export function validateTinyDecideDecisionOptions(
  options: unknown,
  request: TinyDecideRequest | null | undefined,
  dimensions: number
): TinyDecideDecisionOptions | undefined {
  if (options == null) return undefined;
  const fields = readObject(options, new Set(['signal', 'protos']), 'Unsupported TinyDecide decision option.');
  if (fields.signal != null && typeof (fields.signal as TinyDecideAbortSignal).throwIfAborted !== 'function')
    throw invalid('TinyDecide signal must be an AbortSignal.');
  return {
    signal: fields.signal as TinyDecideAbortSignal | null | undefined,
    protos: request ? validatePrototypes(fields.protos, request, dimensions) : undefined
  };
}
