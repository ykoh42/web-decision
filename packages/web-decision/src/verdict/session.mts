import {waitForAbort} from '../assets.js';
import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {WebDecisionError} from '../kev.js';
import {validateH2oSignal} from '../h2o/signal.mjs';
import {VERDICT_ABSTENTION_ID, VERDICT_MAX_CANDIDATES, VERDICT_MODEL} from './contract.mjs';
import {finishVerdictSystemOne, prepareVerdictSystemOne} from './protocol.mjs';
import type {createVerdictTokenizer} from './tokenizer.mjs';

type VerdictTokenizer = ReturnType<typeof createVerdictTokenizer>;

interface VerdictRunner {
  run(rows: unknown[], signal?: AbortSignal | null): Promise<number[][]>;
  dispose(): Promise<void>;
}

interface VerdictSessionOptions {
  readonly runner: VerdictRunner;
  readonly tokenizer: VerdictTokenizer;
  readonly calibrator?: unknown;
  readonly model?: string;
}

interface VerdictDecisionOptions {
  readonly signal?: unknown;
}

interface VerdictRequestQuestion {
  readonly instructions: string;
  readonly options: string[];
}

interface VerdictRequest {
  readonly state: unknown;
  readonly questions: VerdictRequestQuestion[];
}

interface VerdictCommonAnswer {
  readonly isAbstention: boolean;
  readonly probabilities: Record<string, number>;
  readonly logits: Record<string, number>;
  readonly calibrationStatus: 'calibrated_for_scope';
  readonly calibrationTemperature: number;
  readonly calibrationSource: 'global' | 'per-cardinality';
}

interface VerdictChoiceAnswer extends VerdictCommonAnswer {
  readonly type: 'choice';
  readonly choice: string;
  readonly selectedProbability: number;
  readonly concentration: number;
}

interface VerdictScoreAnswer extends VerdictCommonAnswer {
  readonly type: 'score';
  readonly selectedLevelId: string;
  readonly selectedValue: number | null;
  readonly expectedScore: number | null;
  readonly abstentionProbability: number;
}

interface VerdictNoulAnswer extends VerdictCommonAnswer {
  readonly type: 'noul';
  readonly selectedOutcome: string;
  readonly pTrueGivenSufficientEvidence: number | null;
  readonly pInsufficientEvidence: number;
}

type VerdictAnswer = VerdictChoiceAnswer | VerdictScoreAnswer | VerdictNoulAnswer;

interface VerdictSystemOneResult {
  readonly model: string;
  readonly tokenCount: number;
  readonly outputTokenCount: number;
  readonly forwardPassCount: number;
  readonly truncated?: true;
  readonly truncatedQuestions?: string[];
  readonly answers: Record<string, VerdictAnswer>;
}

interface PreparedVerdictSystemOne {
  readonly model: string;
  readonly rows: unknown[];
  readonly tokenCount: number;
  readonly truncatedQuestions: string[];
}

interface VerdictGenericResult {
  readonly tokenCount: number;
  readonly truncated?: true;
  readonly answers: Array<{
    readonly selected: string;
    readonly selectedIndex: number;
    readonly selectedProbability: number;
    readonly abstained: boolean;
    readonly abstentionProbability: number;
    readonly probabilities: Array<{readonly option: string; readonly probability: number; readonly logit: number}>;
  }>;
}

export interface VerdictSession {
  systemOne(state: unknown, questions: unknown, options?: VerdictDecisionOptions): Promise<VerdictSystemOneResult>;
  decide(request: unknown, options?: VerdictDecisionOptions): Promise<VerdictGenericResult>;
  releaseCachedMemory(): Promise<void>;
  dispose(): Promise<void>;
}

const invalid = (message: string): WebDecisionError => new WebDecisionError('INVALID_INPUT', message);
const closedError = () => new WebDecisionError('SESSION_CLOSED', 'The Verdict model has been disposed.');

function signalFrom(options: VerdictDecisionOptions | null | undefined): AbortSignal | null | undefined {
  validateDecisionFields(options);
  const signal = options?.signal;
  if (signal != null) validateH2oSignal(signal);
  (signal as {throwIfAborted(): void} | null | undefined)?.throwIfAborted();
  return signal as AbortSignal | null | undefined;
}

function validateGenericQuestion(question: VerdictRequestQuestion, index: number): void {
  if (
    typeof question.instructions !== 'string' ||
    question.instructions.length === 0 ||
    !Array.isArray(question.options) ||
    question.options.length < 2 ||
    question.options.length > VERDICT_MAX_CANDIDATES ||
    question.options.some((option) => typeof option !== 'string' || option.length === 0) ||
    new Set(question.options).size !== question.options.length
  )
    throw invalid(
      `Verdict question ${index} needs non-empty instructions and 2 to ${VERDICT_MAX_CANDIDATES} unique string options.`
    );
}

export function createVerdictSession({
  runner,
  tokenizer,
  calibrator,
  model = VERDICT_MODEL
}: VerdictSessionOptions): VerdictSession {
  if (
    typeof runner?.run !== 'function' ||
    typeof runner?.dispose !== 'function' ||
    typeof tokenizer?.encode !== 'function' ||
    typeof model !== 'string'
  )
    throw new TypeError('Verdict session components are required.');

  let queue: Promise<void | VerdictSystemOneResult> = Promise.resolve();
  let closed = false;
  let disposing: Promise<void> | undefined;

  function infer(
    prepared: PreparedVerdictSystemOne,
    signal: AbortSignal | null | undefined
  ): Promise<VerdictSystemOneResult> {
    if (closed) throw closedError();
    signal?.throwIfAborted();
    const task = queue.then(async () => {
      if (closed) throw closedError();
      signal?.throwIfAborted();
      const logits = await runner.run(prepared.rows, signal);
      signal?.throwIfAborted();
      return finishVerdictSystemOne(prepared, logits, calibrator) as VerdictSystemOneResult;
    });
    queue = task.catch(() => {});
    return signal ? waitForAbort(() => task, signal) : task;
  }

  function systemOne(
    state: unknown,
    questions: unknown,
    options: VerdictDecisionOptions = {}
  ): Promise<VerdictSystemOneResult> {
    if (closed) throw closedError();
    const signal = signalFrom(options);
    const prepared = prepareVerdictSystemOne(state, questions, tokenizer) as PreparedVerdictSystemOne;
    return infer(prepared, signal);
  }

  return {
    systemOne,
    async decide(request: unknown, options: VerdictDecisionOptions = {}): Promise<VerdictGenericResult> {
      if (closed) throw closedError();
      const checked = validateRequestFields(
        request,
        Infinity,
        Infinity,
        Infinity,
        VERDICT_MAX_CANDIDATES
      ) as VerdictRequest;
      const questions: Record<string, unknown> = Object.create(null);
      checked.questions.forEach((question, index) => {
        validateGenericQuestion(question, index);
        const criteria: Record<string, string> = Object.create(null);
        for (const option of question.options) criteria[option] = option;
        questions[String(index)] = {type: 'choice', instructions: question.instructions, criteria};
      });
      const typed = await systemOne(checked.state, questions, options);
      return {
        tokenCount: typed.tokenCount,
        ...(typed.truncated ? {truncated: true} : {}),
        answers: checked.questions.map((question, index) => {
          const answer = typed.answers[String(index)] as VerdictChoiceAnswer;
          const abstained = answer.choice === VERDICT_ABSTENTION_ID;
          return {
            selected: answer.choice,
            selectedIndex: abstained ? -1 : question.options.indexOf(answer.choice),
            selectedProbability: answer.selectedProbability,
            abstained,
            abstentionProbability: answer.probabilities[VERDICT_ABSTENTION_ID]!,
            probabilities: [...question.options, VERDICT_ABSTENTION_ID].map((option) => ({
              option,
              probability: answer.probabilities[option]!,
              logit: answer.logits[option]!
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
