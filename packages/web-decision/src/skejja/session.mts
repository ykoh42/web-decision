import {waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {fitSkejjaPlatt} from './calibration.mjs';
import {
  SKEJJA_MAX_CALIBRATION_EXAMPLES,
  SKEJJA_MIN_CALIBRATION_EXAMPLES,
  SKEJJA_INT8_PROFILE,
  SKEJJA_INT8DYN_PROFILE
} from './contract.mjs';
import type {SkejjaRunner} from './runner.mjs';
import type {SkejjaTokenizer} from './tokenizer.mjs';
import {
  finishSkejjaDecision,
  finishSkejjaSystemOne,
  normalizeSkejjaNoulQuestion,
  prepareSkejjaDecision,
  prepareSkejjaSystemOne,
  asSkejjaText
} from './protocol.mjs';

type SkejjaProfile = typeof SKEJJA_INT8_PROFILE | typeof SKEJJA_INT8DYN_PROFILE;
type SkejjaAbortSignal = Pick<AbortSignal, 'throwIfAborted' | 'addEventListener' | 'removeEventListener'>;

interface SkejjaPreparedQuestion {
  readonly inputIds: number[];
  readonly labels: readonly string[];
}

interface SkejjaCalibratedQuestion extends SkejjaPreparedQuestion {
  readonly calibrationKey: string;
}

interface SkejjaPreparedRequest {
  readonly questions: readonly SkejjaPreparedQuestion[];
  readonly tokenCount: number;
  readonly truncated: boolean;
}

interface SkejjaCalibrationSnapshot {
  readonly states: readonly {readonly state: string; readonly answer: 0 | 1}[];
  readonly positives: number;
}

export interface SkejjaSession {
  decide(request: unknown, options?: unknown): Promise<unknown>;
  systemOne(state: unknown, questions: unknown, options?: unknown): Promise<unknown>;
  calibrateQuestion(
    question: unknown,
    examples: unknown,
    options?: unknown
  ): Promise<{readonly a: number; readonly b: number; readonly examples: number; readonly positives: number}>;
  releaseCachedMemory(): Promise<void>;
  dispose(): Promise<void>;
}

const invalid = (message: string): WebDecisionError => new WebDecisionError('INVALID_INPUT', message);
const closedError = () => new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');

function readSignal(options: unknown): SkejjaAbortSignal | null | undefined {
  if (options == null) return undefined;
  validateDecisionFields(options);
  const descriptor = Object.getOwnPropertyDescriptor(options as object, 'signal');
  if (descriptor && !Object.hasOwn(descriptor, 'value')) throw invalid('Skejja does not accept signal accessors.');
  const signal: unknown = descriptor?.value;
  const signalLike = signal as SkejjaAbortSignal;
  if (
    signal != null &&
    (typeof signalLike.throwIfAborted !== 'function' ||
      typeof signalLike.addEventListener !== 'function' ||
      typeof signalLike.removeEventListener !== 'function')
  )
    throw invalid('Skejja signal must be an AbortSignal.');
  return signalLike;
}

function snapshotExamples(examples: unknown): SkejjaCalibrationSnapshot {
  const exampleList = examples as unknown[];
  if (
    !Array.isArray(examples) ||
    exampleList.length < SKEJJA_MIN_CALIBRATION_EXAMPLES ||
    exampleList.length > SKEJJA_MAX_CALIBRATION_EXAMPLES
  )
    throw invalid(
      `Skejja calibration needs ${SKEJJA_MIN_CALIBRATION_EXAMPLES} to ${SKEJJA_MAX_CALIBRATION_EXAMPLES} examples.`
    );

  let positives = 0;
  const states = exampleList.map((example: unknown, index: number) => {
    if (!example || typeof example !== 'object' || Array.isArray(example))
      throw invalid(`Skejja calibration example ${index} must be an object.`);
    const exampleRecord = example as object;
    const keys = Reflect.ownKeys(exampleRecord);
    if (keys.some((key) => key !== 'state' && key !== 'answer'))
      throw invalid(`Skejja calibration example ${index} has an unsupported field.`);
    const state = Object.getOwnPropertyDescriptor(exampleRecord, 'state');
    const answer = Object.getOwnPropertyDescriptor(exampleRecord, 'answer');
    if (
      !state ||
      !Object.hasOwn(state, 'value') ||
      !answer ||
      !Object.hasOwn(answer, 'value') ||
      !([true, false, 0, 1] as (boolean | number)[]).includes((answer as PropertyDescriptor).value)
    )
      throw invalid(`Skejja calibration example ${index} needs state and a true or false answer.`);
    if ((answer as PropertyDescriptor).value === true || (answer as PropertyDescriptor).value === 1) positives++;
    const text = asSkejjaText((state as PropertyDescriptor).value);
    return {
      state: text,
      answer:
        (answer as PropertyDescriptor).value === true || (answer as PropertyDescriptor).value === 1
          ? (1 as const)
          : (0 as const)
    };
  });
  if (positives === 0 || positives === states.length)
    throw invalid('Skejja calibration needs both true and false answers.');
  return {states, positives};
}

export function createSkejjaSession({
  runner: suppliedRunner,
  tokenizer: suppliedTokenizer,
  profile = SKEJJA_INT8_PROFILE
}: {
  readonly runner: SkejjaRunner;
  readonly tokenizer: SkejjaTokenizer;
  readonly profile?: SkejjaProfile;
}): SkejjaSession {
  let runner: SkejjaRunner | null = suppliedRunner;
  let tokenizer: SkejjaTokenizer | null = suppliedTokenizer;
  if (
    typeof runner?.run !== 'function' ||
    typeof runner?.dispose !== 'function' ||
    typeof runner?.releaseCachedMemory !== 'function' ||
    typeof tokenizer?.encode !== 'function' ||
    typeof tokenizer?.decode !== 'function'
  )
    throw new TypeError(`${profile.name} session components are required.`);

  let queue: Promise<unknown> = Promise.resolve();
  let closed = false;
  let disposing: Promise<void> | undefined;
  const questionCalibration = new Map<string, readonly [number, number]>();

  const requireOpen = () => {
    if (closed) throw closedError();
  };

  function enqueue<T>(task: () => T | PromiseLike<T>): Promise<T> {
    const current = queue.then(task);
    queue = current.catch(() => {});
    return current;
  }

  async function infer(
    prepared: SkejjaPreparedRequest,
    signal: SkejjaAbortSignal | null | undefined
  ): Promise<number[][]> {
    requireOpen();
    const schedule = () =>
      enqueue(async () => {
        requireOpen();
        const rows = [];
        for (const question of prepared.questions) {
          signal?.throwIfAborted();
          rows.push(await runner!.run(question.inputIds, question.labels.length));
        }
        return rows;
      });
    return signal ? waitForAbort(schedule, signal as AbortSignal) : schedule();
  }

  return {
    async decide(request: unknown, options: unknown = {}): Promise<unknown> {
      requireOpen();
      const signal = readSignal(options);
      signal?.throwIfAborted();
      const snapshot = validateRequestFields(request, Infinity, Infinity, Infinity, profile.maxChoices);
      const prepared = (
        prepareSkejjaDecision as (
          request: unknown,
          tokenizer: SkejjaTokenizer,
          selectedProfile: SkejjaProfile
        ) => unknown
      )(snapshot, tokenizer!, profile) as ReturnType<typeof prepareSkejjaDecision>;
      const rows = await infer(prepared, signal);
      return finishSkejjaDecision(prepared, rows, profile as typeof SKEJJA_INT8_PROFILE);
    },
    async systemOne(state: unknown, questions: unknown, options: unknown = {}): Promise<unknown> {
      requireOpen();
      const signal = readSignal(options);
      signal?.throwIfAborted();
      const prepared = (
        prepareSkejjaSystemOne as (
          inputState: unknown,
          inputQuestions: unknown,
          inputTokenizer: SkejjaTokenizer,
          selectedProfile: SkejjaProfile
        ) => unknown
      )(state, questions, tokenizer!, profile) as ReturnType<typeof prepareSkejjaSystemOne>;
      const rows = await infer(prepared, signal);
      return finishSkejjaSystemOne(prepared, rows, profile as typeof SKEJJA_INT8_PROFILE, questionCalibration);
    },
    async calibrateQuestion(
      question: unknown,
      examples: unknown,
      options: unknown = {}
    ): Promise<{readonly a: number; readonly b: number; readonly examples: number; readonly positives: number}> {
      requireOpen();
      const signal = readSignal(options);
      signal?.throwIfAborted();
      const normalizedQuestion = normalizeSkejjaNoulQuestion(question);
      const snapshots = snapshotExamples(examples);
      const keyQuestion = {calibration: normalizedQuestion};
      const preparedExamples = snapshots.states.map(
        (example) =>
          (
            (
              prepareSkejjaSystemOne as (
                inputState: unknown,
                inputQuestions: unknown,
                inputTokenizer: SkejjaTokenizer,
                selectedProfile: SkejjaProfile
              ) => unknown
            )(example.state, keyQuestion, tokenizer!, profile) as SkejjaPreparedRequest
          ).questions[0]! as SkejjaCalibratedQuestion
      );
      const calibrationKey = preparedExamples[0]!.calibrationKey;

      const schedule = () =>
        enqueue(async () => {
          requireOpen();
          const logits = [];
          for (const row of preparedExamples) {
            signal?.throwIfAborted();
            logits.push((await runner!.run(row.inputIds, row.labels.length))[0]!);
          }
          const coefficients = fitSkejjaPlatt(
            logits,
            snapshots.states.map(({answer}) => answer),
            [profile.plattA, profile.plattB]
          );
          questionCalibration.set(calibrationKey, [coefficients.a, coefficients.b]);
          return {...coefficients, examples: snapshots.states.length, positives: snapshots.positives};
        });
      return signal ? waitForAbort(schedule, signal as AbortSignal) : schedule();
    },
    async releaseCachedMemory(): Promise<void> {
      requireOpen();
      return enqueue(() => runner!.releaseCachedMemory());
    },
    dispose() {
      if (disposing) return disposing;
      closed = true;
      disposing = queue.then(async () => {
        try {
          await runner!.dispose();
        } finally {
          runner = tokenizer = null;
          questionCalibration.clear();
        }
      });
      return disposing;
    }
  };
}
