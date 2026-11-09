import {validateRequestFields} from '../request-fields.mjs';
import {WebDecisionError} from '../kev.js';
import {TINYDECIDE_MAX_CHOICES, TINYDECIDE_QUESTIONS_PER_CHUNK} from './contract.mjs';
import {makeProtos as buildProtos} from './corrections.js';
import {
  validateTinyDecideDecisionOptions,
  validateTinyDecideRequest,
  validateTinyDecideSystemOne,
  validateTinyDecideRows
} from './inputs.mjs';
import type {
  TinyDecideEngineAnswer,
  TinyDecideMetadata,
  TinyDecidePrototype,
  TinyDecideRequest,
  TinyDecide
} from './engine.js';
import type {WordPiece} from './engine.js';

export interface TinyDecideSessionEngine {
  readonly meta: TinyDecideMetadata;
  readonly cfg: TinyDecide['cfg'];
  readonly tok: WordPiece;
  answer: TinyDecide['answer'];
}

export interface TinyDecideDecisionAnswer {
  readonly selected: string | undefined;
  readonly selectedIndex: number;
  readonly probabilities: readonly {
    readonly option: string | undefined;
    readonly probability: number;
    readonly logit: number | undefined;
  }[];
}

export interface TinyDecideDecisionResult {
  readonly tokenCount: number;
  readonly truncated?: true;
  readonly answers: readonly TinyDecideDecisionAnswer[];
}

export interface TinyDecideSystemOneResult {
  readonly model: string;
  readonly tokenCount: number;
  readonly truncated?: true;
  readonly answers: Readonly<Record<string, TinyDecideEngineAnswer['answers'][number]>>;
}

export interface TinyDecideSession {
  answer(request: unknown, options?: unknown): Promise<TinyDecideEngineAnswer>;
  makeProtos(type: string, lists: unknown, center?: unknown): TinyDecidePrototype | null;
  decide(request: unknown, options?: unknown): Promise<TinyDecideDecisionResult>;
  systemOne(state: unknown, questions: unknown, options?: unknown): Promise<TinyDecideSystemOneResult>;
  releaseCachedMemory(): Promise<void>;
  dispose(): Promise<void>;
}

interface TinyDecideInferenceResult {
  readonly answers: TinyDecideEngineAnswer['answers'];
  readonly tokens: {readonly state: number; readonly questions: number; readonly total: number};
  readonly truncated: boolean;
  readonly ms: number;
  readonly ids: number[];
}

type TinyDecidePrototypeList = readonly (TinyDecidePrototype | null | undefined)[] | undefined;
type TinyDecideAbortSignal = Pick<AbortSignal, 'throwIfAborted'>;
type TinyDecideChoiceAnswer = Extract<TinyDecideEngineAnswer['answers'][number], {readonly pick: number}>;

const closed = () => new WebDecisionError('SESSION_CLOSED', 'TinyDecide model session is closed.');
const inference = (cause: unknown): unknown =>
  cause instanceof WebDecisionError || (cause as {readonly name?: unknown} | null | undefined)?.name === 'AbortError'
    ? cause
    : new WebDecisionError('INFERENCE_ERROR', 'TinyDecide inference failed.', {cause});

function answerInChunks(
  engine: TinyDecideSessionEngine,
  request: TinyDecideRequest,
  protos: TinyDecidePrototypeList,
  signal: TinyDecideAbortSignal | null | undefined
): TinyDecideInferenceResult {
  const answers: TinyDecideEngineAnswer['answers'][number][] = [];
  const ids: number[] = [];
  let stateTokens = 0;
  let questionTokens = 0;
  let truncated = false;
  let ms = 0;

  for (let start = 0; start < request.questions.length; start += TINYDECIDE_QUESTIONS_PER_CHUNK) {
    signal?.throwIfAborted();
    const end = Math.min(start + TINYDECIDE_QUESTIONS_PER_CHUNK, request.questions.length);
    const chunk = {state: request.state, questions: request.questions.slice(start, end)};
    const chunkProtos = protos?.slice(start, end);
    const result = engine.answer(chunk, chunkProtos);
    answers.push(...result.answers);
    questionTokens += result.tokens.questions;
    truncated ||= result.truncated;
    ms += result.ms;
    if (start === 0) {
      stateTokens = result.tokens.state;
      ids.push(...result.ids);
    } else {
      ids.push(...result.ids.slice(result.tokens.state));
    }
  }

  return {
    answers,
    tokens: {state: stateTokens, questions: questionTokens, total: stateTokens + questionTokens},
    truncated,
    ms,
    ids
  };
}

function readCorrectionVector(value: unknown, length: number, label: string): number[] {
  if ((!Array.isArray(value) && !ArrayBuffer.isView(value)) || (value as ArrayLike<number>).length !== length)
    throw new WebDecisionError('INVALID_INPUT', `TinyDecide correction ${label} must have ${length} values.`);
  const vector = Array.from(value as ArrayLike<number>);
  if (vector.some((item) => !Number.isFinite(item)))
    throw new WebDecisionError('INVALID_INPUT', `TinyDecide correction ${label} must contain finite numbers.`);
  return vector;
}

function validateCorrectionExample(
  value: unknown,
  optionIndex: number,
  exampleIndex: number,
  count: number,
  dimensions: number
): {readonly v: number[]; readonly z: number[]} {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new WebDecisionError('INVALID_INPUT', `Correction example ${exampleIndex + 1} is invalid.`);
  const fields: Record<string, unknown> = Object.create(null);
  for (const key of Reflect.ownKeys(value as object)) {
    const descriptor = Object.getOwnPropertyDescriptor(value as object, key);
    if ((key !== 'v' && key !== 'z') || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw new WebDecisionError('INVALID_INPUT', `Correction example ${exampleIndex + 1} is invalid.`);
    fields[key as string] = (descriptor as PropertyDescriptor).value;
  }
  if (!Object.hasOwn(fields, 'v') || !Object.hasOwn(fields, 'z'))
    throw new WebDecisionError('INVALID_INPUT', `Correction example ${exampleIndex + 1} is incomplete.`);
  return {
    v: readCorrectionVector(fields.v, dimensions, `option ${optionIndex + 1} example ${exampleIndex + 1} vector`),
    z: readCorrectionVector(fields.z, count, `option ${optionIndex + 1} example ${exampleIndex + 1} logits`)
  };
}

export function createTinyDecideSession({
  engine,
  model
}: {
  readonly engine: TinyDecideSessionEngine;
  readonly model: string;
}): TinyDecideSession {
  let queue: Promise<unknown> = Promise.resolve();
  let disposed: Promise<void> | undefined;
  let isClosed = false;

  function submit(
    request: TinyDecideRequest,
    signal: TinyDecideAbortSignal | null | undefined,
    protos: TinyDecidePrototypeList
  ): Promise<TinyDecideInferenceResult> {
    signal?.throwIfAborted();
    const result = queue.then(() => {
      signal?.throwIfAborted();
      validateTinyDecideRows(engine!, request);
      signal?.throwIfAborted();
      try {
        return answerInChunks(engine!, request, protos, signal);
      } catch (cause: unknown) {
        throw inference(cause);
      }
    });
    queue = result.then(
      () => undefined,
      () => undefined
    );
    return result;
  }

  function requireOpen() {
    if (isClosed) throw closed();
  }

  function makeProtos(type: string, lists: unknown, center: unknown): TinyDecidePrototype | null {
    requireOpen();
    if (!['choice', 'noul', 'score', 'span'].includes(type) || !Array.isArray(lists))
      throw new WebDecisionError('INVALID_INPUT', 'TinyDecide correction examples are invalid.');
    if (type === 'span')
      return buildProtos(
        type,
        lists as Parameters<typeof buildProtos>[1],
        engine!.meta.beta,
        center as readonly number[] | null | undefined
      );
    const count = type === 'noul' ? 2 : (lists as unknown[]).length;
    if (count < 2 || count > 32 || (type === 'noul' && (lists as unknown[]).length !== 2))
      throw new WebDecisionError('INVALID_INPUT', 'TinyDecide correction lists do not match the question.');
    const dimensions = engine!.meta.cfg.dh_head;
    const checkedLists = (lists as unknown[]).map((list: unknown, optionIndex) => {
      if (!Array.isArray(list))
        throw new WebDecisionError('INVALID_INPUT', `Correction list ${optionIndex + 1} is invalid.`);
      return Array.from({length: (list as unknown[]).length}, (_, exampleIndex) => {
        const descriptor = Object.getOwnPropertyDescriptor(list as unknown[], String(exampleIndex));
        if (!descriptor || !Object.hasOwn(descriptor, 'value'))
          throw new WebDecisionError('INVALID_INPUT', `Correction example ${exampleIndex + 1} is missing.`);
        const example: unknown = (descriptor as PropertyDescriptor).value;
        const fields = validateCorrectionExample(example, optionIndex, exampleIndex, count, dimensions);
        return {v: fields.v, z: fields.z};
      });
    });
    const checkedCenter = center == null ? null : readCorrectionVector(center, dimensions, 'center');
    return buildProtos(type, checkedLists, engine!.meta.beta, checkedCenter);
  }

  return {
    async answer(request: unknown, options: unknown = {}): Promise<TinyDecideEngineAnswer> {
      requireOpen();
      const checkedRequest = validateTinyDecideRequest(request);
      const checkedOptions = validateTinyDecideDecisionOptions(options, checkedRequest, engine!.meta.cfg.dh_head);
      return submit(checkedRequest, checkedOptions?.signal, checkedOptions?.protos);
    },
    makeProtos,
    async decide(request: unknown, options: unknown = {}): Promise<TinyDecideDecisionResult> {
      requireOpen();
      const checked = validateRequestFields(request, Infinity, Infinity, Infinity, TINYDECIDE_MAX_CHOICES) as {
        readonly state: string;
        readonly questions: readonly {readonly instructions: string; readonly options: readonly string[]}[];
      };
      if (typeof checked.state !== 'string' || checked.questions.length === 0)
        throw new WebDecisionError('INVALID_INPUT', 'TinyDecide needs a string state and at least one question.');
      const nativeRequest = validateTinyDecideRequest({
        state: checked.state,
        questions: checked.questions.map(({instructions, options: choices}) => ({
          type: 'choice',
          text: instructions,
          options: choices
        }))
      });
      const checkedOptions = validateTinyDecideDecisionOptions(options, nativeRequest, engine!.meta.cfg.dh_head);
      const result = await submit(nativeRequest, checkedOptions?.signal, checkedOptions?.protos);
      return {
        tokenCount: result.tokens.total,
        ...(result.truncated ? {truncated: true} : {}),
        answers: result.answers.map((answer, index) => ({
          selected: nativeRequest.questions[index]!.options![(answer as TinyDecideChoiceAnswer).pick],
          selectedIndex: (answer as TinyDecideChoiceAnswer).pick,
          probabilities: (answer as TinyDecideChoiceAnswer).probs.map((probability, optionIndex) => ({
            option: nativeRequest.questions[index]!.options![optionIndex],
            probability,
            logit: (answer as TinyDecideChoiceAnswer).z0[optionIndex]
          }))
        }))
      };
    },
    async systemOne(state: unknown, questions: unknown, options: unknown = {}): Promise<TinyDecideSystemOneResult> {
      requireOpen();
      const checked = validateTinyDecideSystemOne(state, questions);
      const checkedOptions = validateTinyDecideDecisionOptions(options, checked.request, engine!.meta.cfg.dh_head);
      const result = await submit(checked.request, checkedOptions?.signal, checkedOptions?.protos);
      return {
        model,
        tokenCount: result.tokens.total,
        ...(result.truncated ? {truncated: true} : {}),
        answers: Object.fromEntries(checked.names.map((name, index) => [name, result.answers[index]!]))
      };
    },
    async releaseCachedMemory(): Promise<void> {
      requireOpen();
    },
    dispose(): Promise<void> {
      if (disposed) return disposed;
      isClosed = true;
      disposed = queue.then(() => {
        engine = null as never;
      });
      return disposed;
    }
  };
}
