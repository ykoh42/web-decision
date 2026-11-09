import {waitForAbort} from '../assets.js';
import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {WebDecisionError} from '../kev.js';
import {GLINER_MAX_OPTIONS, GLINER_MAX_TOKENS, GLINER_MOBILE_MODEL} from './contract.mjs';
import {finishGlinerMobileSystemOne, prepareGlinerMobileSystemOne} from './protocol.mjs';
import type {GlinerMobileDecisionResult, PreparedGlinerMobileSystemOne} from './protocol.mjs';
import type {createGlinerMobileTokenizer} from './tokenizer.mjs';

type GlinerTokenizer = ReturnType<typeof createGlinerMobileTokenizer>;

interface GlinerRunner {
  run(inputIds: number[], markerPositions: number[], signal?: AbortSignal | null): Promise<number[]>;
  dispose(): Promise<void>;
}

interface GlinerSessionOptions {
  readonly runner: GlinerRunner;
  readonly tokenizer: GlinerTokenizer;
  readonly model?: string;
}

interface GlinerDecisionOptions {
  readonly signal?: unknown;
}

interface GlinerGenericRequest {
  readonly state: unknown;
  readonly questions: Array<{readonly instructions: string; readonly options: string[]}>;
}

interface GlinerChoiceAnswer {
  readonly choice: string;
  readonly probabilities: Record<string, number>;
}

interface GlinerTaskResult {
  readonly result: GlinerMobileDecisionResult;
  readonly logits: number[];
}

export interface GlinerMobileSession {
  systemOne(state: unknown, questions: unknown, options?: GlinerDecisionOptions): Promise<GlinerMobileDecisionResult>;
  decide(
    request: unknown,
    options?: GlinerDecisionOptions
  ): Promise<{
    readonly tokenCount: number;
    readonly truncated?: true;
    readonly answers: Array<{
      readonly selected: string;
      readonly selectedIndex: number;
      readonly probabilities: Array<{readonly option: string; readonly probability: number; readonly logit: number}>;
    }>;
  }>;
  releaseCachedMemory(): Promise<void>;
  dispose(): Promise<void>;
}

const invalid = (message: string): WebDecisionError => new WebDecisionError('INVALID_INPUT', message);
const closedError = () => new WebDecisionError('SESSION_CLOSED', 'The GLiNER model has been disposed.');

function validateSignal(options: GlinerDecisionOptions | null | undefined): AbortSignal | null | undefined {
  validateDecisionFields(options);
  const signal = options?.signal;
  if (
    signal != null &&
    (typeof (signal as {throwIfAborted?: unknown}).throwIfAborted !== 'function' ||
      typeof (signal as {addEventListener?: unknown}).addEventListener !== 'function' ||
      typeof (signal as {removeEventListener?: unknown}).removeEventListener !== 'function')
  )
    throw invalid('GLiNER signal must be an AbortSignal.');
  return signal as AbortSignal | null | undefined;
}

export function createGlinerMobileSession({
  runner,
  tokenizer,
  model = GLINER_MOBILE_MODEL
}: GlinerSessionOptions): GlinerMobileSession {
  if (
    typeof runner?.run !== 'function' ||
    typeof runner?.dispose !== 'function' ||
    typeof tokenizer?.encode !== 'function'
  )
    throw new TypeError('GLiNER session components are required.');

  let queue: Promise<void | GlinerTaskResult> = Promise.resolve();
  let closed = false;
  let disposing: Promise<void> | undefined;

  function runPrepared(
    prepared: PreparedGlinerMobileSystemOne,
    signal: AbortSignal | null | undefined
  ): Promise<GlinerTaskResult> {
    if (closed) throw closedError();
    signal?.throwIfAborted();
    const task = queue.then(async () => {
      if (closed) throw closedError();
      signal?.throwIfAborted();
      const logits = await runner.run(prepared.inputIds, prepared.markerPositions, signal);
      signal?.throwIfAborted();
      return {result: finishGlinerMobileSystemOne(prepared, logits, model), logits};
    });
    queue = task.catch(() => {});
    return signal ? waitForAbort(() => task, signal) : task;
  }

  async function systemOne(
    state: unknown,
    questions: unknown,
    options: GlinerDecisionOptions = {}
  ): Promise<GlinerMobileDecisionResult> {
    if (closed) throw closedError();
    const signal = validateSignal(options);
    signal?.throwIfAborted();
    const prepared = prepareGlinerMobileSystemOne(state, questions, tokenizer);
    return (await runPrepared(prepared, signal)).result;
  }

  return {
    systemOne,
    async decide(request: unknown, options: GlinerDecisionOptions = {}) {
      if (closed) throw closedError();
      const checked = validateRequestFields(
        request,
        Infinity,
        Infinity,
        GLINER_MAX_TOKENS,
        GLINER_MAX_OPTIONS
      ) as GlinerGenericRequest;
      if (typeof checked.state !== 'string') throw invalid('GLiNER state must be text.');
      const questions: Record<string, unknown> = Object.create(null);
      for (let index = 0; index < checked.questions.length; index++) {
        const question = checked.questions[index]!;
        questions[String(index)] = {type: 'choice', instructions: question.instructions, criteria: question.options};
      }
      const signal = validateSignal(options);
      signal?.throwIfAborted();
      const prepared = prepareGlinerMobileSystemOne(checked.state, questions, tokenizer);
      const {result, logits} = await runPrepared(prepared, signal);
      return {
        tokenCount: result.tokenCount,
        ...(result.truncated ? {truncated: true} : {}),
        answers: checked.questions.map((question, index) => {
          const id = String(index);
          const answer = result.answers[id] as GlinerChoiceAnswer;
          const preparedQuestion = prepared.questions[index]!;
          return {
            selected: answer.choice,
            selectedIndex: preparedQuestion.labels.indexOf(answer.choice),
            probabilities: preparedQuestion.labels.map((option, optionIndex) => ({
              option,
              probability: answer.probabilities[option]!,
              logit: logits[preparedQuestion.markerOffset + optionIndex]!
            }))
          };
        })
      };
    },
    async releaseCachedMemory() {
      if (closed) throw closedError();
      await queue;
    },
    dispose() {
      if (disposing) return disposing;
      closed = true;
      disposing = queue.then(() => runner.dispose());
      return disposing;
    }
  };
}
