import {waitForAbort} from '../assets.js';
import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {WebDecisionError} from '../kev.js';
import {validateH2oSignal} from '../h2o/signal.mjs';
import {D1_OMNI_MODEL, D1_OMNI_TEMPERATURES} from './contract.mjs';
import {finishD1OmniSystemOne, prepareD1OmniSystemOne} from './protocol.mjs';
import type {D1OmniTokenizer} from './tokenizer.mjs';

interface D1OmniQuestion {
  readonly id: string;
  readonly type: 'choice' | 'score' | 'noul';
  readonly qtype: number;
  readonly instructions: string;
  readonly labels: string[];
  readonly options: string[];
  readonly legend?: string[];
}

interface D1OmniEncodedRow {
  readonly ids: number[];
  readonly markerPositions: number[];
  readonly truncated: boolean;
  readonly question: D1OmniQuestion;
}

interface D1OmniPreparedSystemOne {
  readonly model: string;
  readonly rows: D1OmniEncodedRow[];
  readonly tokenCount: number;
  readonly truncated: boolean;
}

interface D1OmniRunner {
  run(rows: D1OmniEncodedRow[], signal?: AbortSignal | null): Promise<number[][]>;
  dispose(): Promise<void>;
}

interface D1OmniDecisionOptions {
  readonly signal?: unknown;
}

interface D1OmniRequestQuestion {
  readonly instructions: unknown;
  readonly options: unknown;
}

interface D1OmniCheckedRequest {
  readonly state: unknown;
  readonly questions: D1OmniRequestQuestion[];
}

interface D1OmniAnswerBase {
  readonly confidence: number;
  readonly probabilities: Record<string, number>;
}

interface D1OmniChoiceAnswer extends D1OmniAnswerBase {
  readonly type: 'choice';
  readonly choice: string;
}

interface D1OmniScoreAnswer extends D1OmniAnswerBase {
  readonly type: 'score';
  readonly score: number;
  readonly expected: number;
  readonly mostLikelyLevel: number;
  readonly legend: Record<string, string>;
}

interface D1OmniNoulAnswer extends D1OmniAnswerBase {
  readonly type: 'noul';
  readonly noul: number;
}

type D1OmniAnswer = D1OmniChoiceAnswer | D1OmniScoreAnswer | D1OmniNoulAnswer;

export interface D1OmniSystemOneResult {
  readonly model: string;
  readonly tokenCount: number;
  readonly outputTokenCount: number;
  readonly truncated?: true;
  readonly answers: Record<string, D1OmniAnswer>;
}

export interface D1OmniDecisionResult {
  readonly tokenCount: number;
  readonly truncated?: true;
  readonly answers: Array<{
    readonly selected: string;
    readonly selectedIndex: number;
    readonly probabilities: Array<{readonly option: string; readonly probability: number}>;
  }>;
}

interface D1OmniSessionOptions {
  readonly tokenizer?: D1OmniTokenizer | null;
  readonly runner?: D1OmniRunner | null;
  readonly temperatures?: Readonly<Record<string, number>> | null;
  readonly model?: unknown;
}

export interface D1OmniSession {
  systemOne(state: unknown, questions: unknown, options?: unknown): Promise<D1OmniSystemOneResult>;
  decide(request: unknown, options?: unknown): Promise<D1OmniDecisionResult>;
  releaseCachedMemory(): Promise<void>;
  dispose(): Promise<void>;
}

const invalid = (message: string): WebDecisionError => new WebDecisionError('INVALID_INPUT', message);
const closedError = () => new WebDecisionError('SESSION_CLOSED', 'The d1-omni model has been disposed.');

function signalFrom(options: unknown): AbortSignal | null | undefined {
  validateDecisionFields(options);
  const signal = (options as D1OmniDecisionOptions | null | undefined)?.signal;
  if (signal != null) validateH2oSignal(signal);
  (signal as AbortSignal | null | undefined)?.throwIfAborted();
  return signal as AbortSignal | null | undefined;
}

export function createD1OmniSession({
  tokenizer,
  runner,
  temperatures = D1_OMNI_TEMPERATURES,
  model = D1_OMNI_MODEL
}: D1OmniSessionOptions): D1OmniSession {
  if (
    typeof tokenizer?.encode !== 'function' ||
    typeof runner?.run !== 'function' ||
    typeof runner?.dispose !== 'function' ||
    !temperatures ||
    typeof temperatures !== 'object' ||
    typeof model !== 'string'
  )
    throw new TypeError('d1-omni session components are required.');

  let queue: Promise<void | D1OmniSystemOneResult> = Promise.resolve();
  let closed = false;
  let disposing: Promise<void> | undefined;

  function infer(
    prepared: D1OmniPreparedSystemOne,
    signal: AbortSignal | null | undefined
  ): Promise<D1OmniSystemOneResult> {
    if (closed) throw closedError();
    signal?.throwIfAborted();
    const task = queue.then(async () => {
      if (closed) throw closedError();
      signal?.throwIfAborted();
      const rows = await (runner as D1OmniRunner).run(prepared.rows, signal);
      signal?.throwIfAborted();
      return finishD1OmniSystemOne(prepared, rows, temperatures) as D1OmniSystemOneResult;
    });
    queue = task.catch(() => {});
    return signal ? waitForAbort(() => task, signal) : task;
  }

  async function systemOne(state: unknown, questions: unknown, options: unknown = {}): Promise<D1OmniSystemOneResult> {
    if (closed) throw closedError();
    const signal = signalFrom(options);
    const prepared = prepareD1OmniSystemOne(state, questions, tokenizer) as D1OmniPreparedSystemOne;
    return infer(prepared, signal);
  }

  return {
    systemOne,
    async decide(request: unknown, options: unknown = {}): Promise<D1OmniDecisionResult> {
      if (closed) throw closedError();
      const checked = validateRequestFields(request) as D1OmniCheckedRequest;
      const questions: Record<string, unknown> = Object.create(null);
      checked.questions.forEach((question, index) => {
        if (
          !Array.isArray(question.options as unknown) ||
          (question.options as unknown[]).length < 2 ||
          (question.options as unknown[]).some((option: unknown) => typeof option !== 'string')
        )
          throw invalid(`Question ${index} needs at least 2 string options.`);
        questions[String(index)] = {
          type: 'choice',
          instructions: question.instructions,
          criteria: question.options
        };
      });
      const typed = await systemOne(checked.state, questions, options);
      return {
        tokenCount: typed.tokenCount,
        ...(typed.truncated ? {truncated: true} : {}),
        answers: checked.questions.map((question, index) => {
          const answer = typed.answers[String(index)]! as D1OmniChoiceAnswer;
          return {
            selected: answer.choice,
            selectedIndex: (question.options as string[]).indexOf(answer.choice),
            probabilities: (question.options as string[]).map((option) => ({
              option,
              probability: answer.probabilities[option] as number
            }))
          };
        })
      };
    },
    async releaseCachedMemory(): Promise<void> {
      if (closed) throw closedError();
      await queue;
    },
    dispose(): Promise<void> {
      if (disposing) return disposing;
      closed = true;
      disposing = queue.then(() => (runner as D1OmniRunner).dispose());
      return disposing;
    }
  };
}
