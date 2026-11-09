import {WebDecisionError} from './kev.js';

interface Budget {
  used: number;
  max: number;
}
interface StagedQuestion {
  instructions: unknown;
  options: unknown;
  optionCount?: number | undefined;
}
interface Limits {
  maxInputTokens: number;
  maxOptions: number;
  maxTotalOptions: number;
}
interface Counts {
  optionMarkers: number;
  optionCount: number;
}
interface ValidationOptions {
  dataPropertiesOnly?: boolean;
  allowInheritedDataProperties?: boolean;
}

const fail: () => never = () => {
  throw new WebDecisionError('INVALID_INPUT', 'Unsupported decision request or option field.');
};
const requestKeys = new Set(['state', 'questions']);
const questionKeys = new Set(['instructions', 'options']);
const decisionKeys = new Set(['signal']);
const knownUnsupported = ['type', 'criteria', 'image', 'images', 'audio', 'video'];

function checkOwnField(value: object, key: PropertyKey, allowed: Set<string>): void {
  if (typeof key !== 'string' || !allowed.has(key)) fail();
}

function checkFields(value: unknown, allowed: Set<string>): asserts value is object {
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail();
  const ownKeys = Reflect.ownKeys(value);
  if (ownKeys.length > allowed.size) fail();
  for (const key of ownKeys) checkOwnField(value, key, allowed);
  // Include enumerable inherited properties and known unsupported inherited
  // properties, while allowing ordinary and cross-realm object prototypes.
  for (const key in value) if (!allowed.has(key)) fail();
  for (const key of knownUnsupported) if (!allowed.has(key) && key in value) fail();
}

function readValue(value: unknown[], key: 'length'): number;
function readValue(value: unknown, key: string | number): unknown;
function readValue(value: unknown, key: string | number): unknown {
  return (value as Record<string | number, unknown>)[key];
}

function addTextToBudget(value: unknown, budget: Budget | null): void {
  if (typeof value !== 'string' || !budget) return;
  budget.used += value.length;
  if (budget.used > budget.max) fail();
}

function snapshotOptions(
  options: unknown,
  budget: Budget | null,
  maxOptions: number,
  expectedCount: number | undefined
): unknown {
  if (!Array.isArray(options)) return options;
  const count = readValue(options, 'length');
  if (count > maxOptions || (expectedCount !== undefined && count !== expectedCount)) fail();
  // Preserve sparse array construction and its single-argument semantics.
  // oxlint-disable-next-line unicorn/no-new-array
  const snapshot = new Array(count);
  for (let i = 0; i < count; i++) {
    if (!Object.hasOwn(options, i)) fail();
    const option = readValue(options, i);
    addTextToBudget(option, budget);
    snapshot[i] = option;
  }
  return snapshot;
}

function stageQuestions(
  questions: unknown[],
  questionCount: number,
  budget: Budget | null,
  maxInputTokens: number,
  minimumTokenCount: number,
  maxOptions: number,
  maxTotalOptions: number
): StagedQuestion[] {
  const staged = [];
  let totalOptionMarkers = 0,
    totalOptionCount = 0;
  for (let i = 0; i < questionCount; i++) {
    if (!Object.hasOwn(questions, i)) fail();
    const question = readValue(questions, i);
    checkFields(question, questionKeys);
    const instructions = readValue(question, 'instructions'),
      options = readValue(question, 'options');
    addTextToBudget(instructions, budget);
    let optionCount: number | undefined;
    if (Array.isArray(options)) {
      optionCount = readValue(options, 'length');
      if (optionCount > maxOptions) fail();
      totalOptionMarkers += Math.max(0, optionCount - 1);
      if (minimumTokenCount + totalOptionMarkers * 2 > maxInputTokens) fail();
      totalOptionCount += optionCount;
      if (totalOptionCount > maxTotalOptions) fail();
    }
    staged.push({instructions, options, optionCount});
  }
  return staged;
}

function snapshotStagedQuestions(staged: StagedQuestion[], budget: Budget | null, maxOptions: number) {
  return staged.map(({instructions, options, optionCount}) => ({
    instructions,
    options: snapshotOptions(options, budget, maxOptions, optionCount)
  }));
}

function snapshotQuestions(
  questions: unknown[],
  questionCount: number,
  budget: Budget | null,
  maxInputTokens: number,
  minimumTokenCount: number,
  maxOptions: number
) {
  const snapshot = [];
  let totalOptionMarkers = 0;
  for (let i = 0; i < questionCount; i++) {
    if (!Object.hasOwn(questions, i)) fail();
    const question = readValue(questions, i);
    checkFields(question, questionKeys);
    const instructions = readValue(question, 'instructions'),
      options = readValue(question, 'options');
    addTextToBudget(instructions, budget);
    if (Array.isArray(options)) {
      totalOptionMarkers += Math.max(0, readValue(options, 'length') - 1);
      if (minimumTokenCount + totalOptionMarkers * 2 > maxInputTokens) fail();
    }
    snapshot.push({instructions, options: snapshotOptions(options, budget, maxOptions, undefined)});
  }
  return snapshot;
}

function snapshotOwnDataFields(value: object, allowed: Set<string>) {
  const fields = new Map<string, unknown>();
  const keys = Reflect.ownKeys(value);
  if (keys.length > allowed.size) fail();
  for (const key of keys) {
    if (typeof key !== 'string' || !allowed.has(key)) fail();
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (!descriptor || !Object.hasOwn(descriptor, 'value')) fail();
    fields.set(key, descriptor.value);
  }
  return fields;
}

function snapshotInheritedDataFields(
  prototype: object,
  allowed: Set<string>,
  fields: Map<string, unknown>,
  allowInheritedDataProperties: boolean
) {
  for (const key of Reflect.ownKeys(prototype)) {
    if (typeof key !== 'string' || (allowed.has(key) && fields.has(key))) continue;
    const descriptor = Object.getOwnPropertyDescriptor(prototype, key);
    if (!descriptor) continue;
    if (!allowed.has(key) && (descriptor.enumerable || knownUnsupported.includes(key))) fail();
    if (!allowInheritedDataProperties || !allowed.has(key)) continue;
    if (!Object.hasOwn(descriptor, 'value')) fail();
    fields.set(key, descriptor.value);
  }
}

function snapshotDataRecord(value: unknown, allowed: Set<string>, allowInheritedDataProperties: boolean) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail();
  const fields = snapshotOwnDataFields(value, allowed);
  const visited = new Set([value]);
  let current = Object.getPrototypeOf(value);
  while (current !== null) {
    if (visited.has(current)) fail();
    visited.add(current);
    snapshotInheritedDataFields(current, allowed, fields, allowInheritedDataProperties);
    current = Object.getPrototypeOf(current);
  }
  return fields;
}

function dataArrayLength(value: unknown[]): number;
function dataArrayLength(value: unknown): number | undefined;
function dataArrayLength(value: unknown): number | undefined {
  if (!Array.isArray(value)) return undefined;
  const descriptor = Object.getOwnPropertyDescriptor(value, 'length');
  if (!descriptor || !Object.hasOwn(descriptor, 'value')) fail();
  return descriptor.value;
}

function dataArrayItem(value: object, index: number): unknown {
  const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
  if (!descriptor || !Object.hasOwn(descriptor, 'value')) fail();
  return descriptor.value;
}

function snapshotDataOptions(options: unknown, count: number | undefined, budget: Budget | null): unknown {
  if (!Array.isArray(options)) return options;
  // Preserve sparse array construction and its single-argument semantics.
  // oxlint-disable-next-line unicorn/no-new-array
  const snapshot = new Array(count);
  for (let index = 0; index < count!; index++) {
    const option = dataArrayItem(options, index);
    addTextToBudget(option, budget);
    snapshot[index] = option;
  }
  return snapshot;
}

function stageDataQuestion(
  questions: unknown[],
  index: number,
  budget: Budget | null,
  minimumTokenCount: number,
  limits: Limits,
  counts: Counts,
  allowInheritedDataProperties: boolean
): StagedQuestion {
  const fields = snapshotDataRecord(dataArrayItem(questions, index), questionKeys, allowInheritedDataProperties);
  const instructions = fields.get('instructions');
  const options = fields.get('options');
  addTextToBudget(instructions, budget);
  if (!Array.isArray(options)) return {instructions, options};

  const optionCount = dataArrayLength(options);
  if (optionCount > limits.maxOptions) fail();
  counts.optionMarkers += Math.max(0, optionCount - 1);
  if (minimumTokenCount + counts.optionMarkers * 2 > limits.maxInputTokens) fail();
  counts.optionCount += optionCount;
  if (counts.optionCount > limits.maxTotalOptions) fail();
  return {instructions, options, optionCount};
}

function stageDataQuestions(
  questions: unknown[],
  questionCount: number,
  budget: Budget | null,
  minimumTokenCount: number,
  limits: Limits,
  allowInheritedDataProperties: boolean
): StagedQuestion[] {
  const staged = [];
  const counts = {optionMarkers: 0, optionCount: 0};
  for (let index = 0; index < questionCount; index++)
    staged.push(
      stageDataQuestion(questions, index, budget, minimumTokenCount, limits, counts, allowInheritedDataProperties)
    );
  return staged;
}

function snapshotDataRequestFields(
  request: unknown,
  maxQuestions: number,
  maxInputCharacters: number,
  maxInputTokens: number,
  maxOptions: number,
  maxTotalOptions: number,
  allowInheritedDataProperties: boolean
) {
  const requestFields = snapshotDataRecord(request, requestKeys, allowInheritedDataProperties);
  const state = requestFields.get('state');
  const questions = requestFields.get('questions');
  if (!Array.isArray(questions)) fail();
  const questionCount = dataArrayLength(questions);
  if (questionCount > maxQuestions) fail();

  const budget = Number.isSafeInteger(maxInputCharacters) ? {max: maxInputCharacters, used: 0} : null;
  const minimumTokenCount = Number.isSafeInteger(maxInputTokens) ? 1 + questionCount * 4 : 0;
  if (minimumTokenCount > maxInputTokens) fail();
  addTextToBudget(state, budget);

  const staged = stageDataQuestions(
    questions,
    questionCount,
    budget,
    minimumTokenCount,
    {maxInputTokens, maxOptions, maxTotalOptions},
    allowInheritedDataProperties
  );

  return {
    state,
    questions: staged.map(({instructions, options, optionCount}) => ({
      instructions,
      options: snapshotDataOptions(options, optionCount, budget)
    }))
  };
}

/** Call before tokenization; existing model packers retain all other validation. */
export function validateRequestFields(
  request: unknown,
  maxQuestions = Infinity,
  maxInputCharacters = Infinity,
  maxInputTokens = Infinity,
  maxOptions = Infinity,
  maxTotalOptions = Infinity,
  validation: ValidationOptions | null = {}
) {
  if (validation?.dataPropertiesOnly === true)
    return snapshotDataRequestFields(
      request,
      maxQuestions,
      maxInputCharacters,
      maxInputTokens,
      maxOptions,
      maxTotalOptions,
      validation.allowInheritedDataProperties === true
    );
  checkFields(request, requestKeys);
  const questions = readValue(request, 'questions');
  if (!Array.isArray(questions)) fail();
  const questionCount = readValue(questions, 'length');
  if (questionCount > maxQuestions) fail();
  const textBudget = Number.isSafeInteger(maxInputCharacters) ? {max: maxInputCharacters, used: 0} : null;
  const minimumTokenCount = Number.isSafeInteger(maxInputTokens) ? 1 + questionCount * 4 : 0;
  if (minimumTokenCount > maxInputTokens) fail();
  const state = readValue(request, 'state');
  addTextToBudget(state, textBudget);
  const snapshot = Number.isSafeInteger(maxTotalOptions)
    ? snapshotStagedQuestions(
        stageQuestions(
          questions,
          questionCount,
          textBudget,
          maxInputTokens,
          minimumTokenCount,
          maxOptions,
          maxTotalOptions
        ),
        textBudget,
        maxOptions
      )
    : snapshotQuestions(questions, questionCount, textBudget, maxInputTokens, minimumTokenCount, maxOptions);
  return {state, questions: snapshot};
}

/** Preserve the existing optional null/undefined options behavior. */
export function validateDecisionFields(options: unknown): void {
  if (options == null) return;
  checkFields(options, decisionKeys);
}
