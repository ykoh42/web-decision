import {waitForAbort} from '../assets.js';
import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {WebDecisionError} from '../kev.js';
import {LAYA_MAX_OPTIONS, LAYA_MODEL} from './contract.mjs';
import {finishLayaSystemOne, prepareLayaSystemOne, validateLayaDecisionRequest} from './protocol.mjs';
import type {
  LayaChoiceAnswer,
  LayaDecisionResult,
  LayaGenericRequest,
  LayaGenericResult,
  LayaQuestionValue,
  LayaTemperatures,
  PreparedLayaRow
} from './protocol.mjs';
import type {LayaTokenizer} from './tokenizer.mjs';

type LayaSignal = AbortSignal;

interface LayaDecisionOptions {
  readonly signal?: LayaSignal | null;
}

interface LayaRunner {
  run(
    rows: PreparedLayaRow[],
    signal?: LayaSignal | null
  ): Promise<{
    rows: Array<{logits: number[]; actLogits: number[]}>;
  }>;
  dispose(): Promise<void>;
}

interface LayaSessionOptions {
  readonly runner: LayaRunner;
  readonly tokenizer: LayaTokenizer;
  readonly temperatures: LayaTemperatures;
  readonly model?: string;
}

export interface LayaSession {
  systemOne(state: unknown, questions: unknown, options?: LayaDecisionOptions): Promise<LayaDecisionResult>;
  decide(request: unknown, options?: LayaDecisionOptions): Promise<LayaGenericResult>;
  releaseCachedMemory(): Promise<void>;
  dispose(): Promise<void>;
}

const invalid = (message: string): WebDecisionError => new WebDecisionError('INVALID_INPUT', message);
const closedError = () => new WebDecisionError('SESSION_CLOSED', 'The Laya model has been disposed.');

function validateSignal(options: LayaDecisionOptions | null | undefined): LayaSignal | null | undefined {
  validateDecisionFields(options);
  const signal = options?.signal;
  if (
    signal != null &&
    (typeof signal.throwIfAborted !== 'function' ||
      typeof signal.addEventListener !== 'function' ||
      typeof signal.removeEventListener !== 'function')
  )
    throw invalid('Laya signal must be an AbortSignal.');
  return signal;
}

export function createLayaSession({
  runner,
  tokenizer,
  temperatures,
  model = LAYA_MODEL
}: LayaSessionOptions): LayaSession {
  if (
    typeof runner?.run !== 'function' ||
    typeof runner?.dispose !== 'function' ||
    typeof tokenizer?.encode !== 'function' ||
    !Array.isArray(temperatures?.byType) ||
    temperatures.byType.length !== 3 ||
    !temperatures.byOptions ||
    typeof model !== 'string'
  )
    throw new TypeError('Laya session components are required.');

  let queue: Promise<void | LayaDecisionResult> = Promise.resolve();
  let closed = false;
  let disposing: Promise<void> | undefined;

  function infer(
    prepared: ReturnType<typeof prepareLayaSystemOne>,
    signal: LayaSignal | null | undefined
  ): Promise<LayaDecisionResult> {
    if (closed) throw closedError();
    signal?.throwIfAborted();
    const task = queue.then(async () => {
      if (closed) throw closedError();
      signal?.throwIfAborted();
      const raw = await runner.run(prepared.rows, signal);
      signal?.throwIfAborted();
      return finishLayaSystemOne(prepared, raw, temperatures, model);
    });
    queue = task.catch(() => {});
    return signal ? waitForAbort(() => task, signal) : task;
  }

  async function systemOne(
    state: unknown,
    questions: unknown,
    options: LayaDecisionOptions = {}
  ): Promise<LayaDecisionResult> {
    if (closed) throw closedError();
    const signal = validateSignal(options);
    signal?.throwIfAborted();
    const prepared = prepareLayaSystemOne(state, questions, tokenizer);
    return infer(prepared, signal);
  }

  return {
    systemOne,
    async decide(request: unknown, options: LayaDecisionOptions = {}): Promise<LayaGenericResult> {
      if (closed) throw closedError();
      const checked = validateRequestFields(
        request,
        Infinity,
        Infinity,
        Infinity,
        LAYA_MAX_OPTIONS
      ) as LayaGenericRequest;
      validateLayaDecisionRequest(checked);
      const questions: Record<string, LayaQuestionValue> = Object.create(null);
      for (let index = 0; index < checked.questions.length; index++) {
        const question = checked.questions[index]!;
        questions[String(index)] = {type: 'choice', instructions: question.instructions, criteria: question.options};
      }
      const typed = await systemOne(checked.state, questions, options);
      return {
        tokenCount: typed.tokenCount,
        ...(typed.truncated ? {truncated: true} : {}),
        answers: checked.questions.map((question, index) => {
          const answer = typed.answers[String(index)] as LayaChoiceAnswer;
          return {
            selected: answer.choice,
            selectedIndex: question.options.indexOf(answer.choice),
            probabilities: question.options.map((option) => ({
              option,
              probability: answer.probabilities[option]!
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
