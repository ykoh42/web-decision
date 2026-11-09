import {waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {validateH2oSignal} from '../h2o/signal.mjs';
import {assertUsable} from './scope.mjs';
import {packStrandsGemma4Request, unpackStrandsGemma4Scores} from './gemma4-protocol.mjs';
import {
  STRANDS_GEMMA4_E2B_CONTEXT,
  STRANDS_GEMMA4_E2B_DECISION_CONFIG,
  STRANDS_GEMMA4_E2B_MODEL
} from './gemma4-e2b-contract.mjs';

type Gemma4QuestionType = 'choice' | 'score' | 'noul';

interface Gemma4PackedRow {
  readonly name: string;
  readonly type: Gemma4QuestionType;
  readonly options: readonly string[];
  readonly descriptions: readonly string[];
  readonly inputIds: readonly number[];
  readonly optionIndices: readonly number[];
  readonly answerIndex: number;
}

interface Gemma4PackedRequest {
  readonly rows: readonly Gemma4PackedRow[];
  readonly tokenCount: number;
}

interface Gemma4ChoiceAnswer {
  readonly type: 'choice';
  readonly choice: string;
  readonly probabilities: Readonly<Record<string, number>>;
}

interface Gemma4ScoreAnswer {
  readonly type: 'score';
  readonly score: number;
  readonly legend: Readonly<Record<string, string>>;
  readonly probabilities: Readonly<Record<string, number>>;
  readonly confidence: number;
}

interface Gemma4NoulAnswer {
  readonly type: 'noul';
  readonly noul: number;
}

type Gemma4Answer = Gemma4ChoiceAnswer | Gemma4ScoreAnswer | Gemma4NoulAnswer;

interface Gemma4InferenceResult {
  readonly model: string;
  readonly tokenCount: number;
  readonly answers: Readonly<Record<string, Gemma4Answer>>;
}

interface Gemma4DecisionResult {
  readonly tokenCount: number;
  readonly answers: readonly {
    readonly selected: string;
    readonly selectedIndex: number;
    readonly probabilities: readonly {readonly option: string; readonly probability: number; readonly logit: number}[];
  }[];
}

interface Gemma4CheckedQuestion {
  readonly instructions: unknown;
  readonly options: unknown;
}

interface Gemma4CheckedRequest {
  readonly state: unknown;
  readonly questions: readonly Gemma4CheckedQuestion[];
}

interface Gemma4CallOptions {
  readonly signal?: unknown;
}

interface Gemma4Tokenizer {
  encode: unknown;
}

interface Gemma4Runner {
  run(row: Gemma4PackedRow): unknown;
  releaseCachedMemory(): unknown;
  dispose(): unknown;
}

interface Gemma4SessionOptions {
  readonly tokenizer: Gemma4Tokenizer | null;
  readonly runner: Gemma4Runner | null;
}

interface Gemma4Inference {
  readonly result: Gemma4InferenceResult;
  readonly scoreRows: readonly ArrayLike<number>[];
}

function closedError(): WebDecisionError {
  return new WebDecisionError('SESSION_CLOSED', 'The Strands Gemma 4 model has been disposed.');
}

function requestSignal(options: unknown): AbortSignal | null | undefined {
  validateDecisionFields(options);
  const signal = (options as Gemma4CallOptions | null | undefined)?.signal;
  if (signal != null) validateH2oSignal(signal);
  (signal as AbortSignal | null | undefined)?.throwIfAborted();
  return signal as AbortSignal | null | undefined;
}

function genericQuestions(checked: Gemma4CheckedRequest): Record<string, unknown> {
  if (typeof checked.state !== 'string')
    throw new WebDecisionError('INVALID_INPUT', 'state must be a string for decide().');
  const entries = checked.questions.map((question, index) => {
    if (
      typeof question.instructions !== 'string' ||
      !Array.isArray(question.options) ||
      (question.options as unknown[]).some((option: unknown) => typeof option !== 'string')
    )
      throw new WebDecisionError('INVALID_INPUT', `Question ${index} needs text instructions and string options.`);
    return [
      String(index),
      {
        type: 'choice',
        instructions: question.instructions as string,
        criteria: question.options as string[]
      }
    ];
  });
  return Object.fromEntries(entries);
}

export function createGemma4Session({tokenizer, runner}: Gemma4SessionOptions) {
  if (
    typeof tokenizer?.encode !== 'function' ||
    typeof runner?.run !== 'function' ||
    typeof runner?.releaseCachedMemory !== 'function' ||
    typeof runner?.dispose !== 'function'
  )
    throw new TypeError('Strands Gemma 4 needs a tokenizer and owned runner.');
  let queue: Promise<unknown> = Promise.resolve();
  let closed = false;
  let disposal: Promise<unknown> | undefined;

  function requireOpen(): void {
    if (closed) throw closedError();
    assertUsable(runner as Gemma4Runner);
  }

  function infer(packed: Gemma4PackedRequest, signal: AbortSignal | null | undefined): Promise<Gemma4Inference> {
    requireOpen();
    signal?.throwIfAborted();
    const task = queue.then(async () => {
      requireOpen();
      const scoreRows: ArrayLike<number>[] = [];
      for (const row of packed.rows) {
        signal?.throwIfAborted();
        scoreRows.push((await (runner as Gemma4Runner).run(row)) as ArrayLike<number>);
        signal?.throwIfAborted();
      }
      return {
        result: unpackStrandsGemma4Scores(
          packed,
          scoreRows,
          STRANDS_GEMMA4_E2B_DECISION_CONFIG,
          STRANDS_GEMMA4_E2B_MODEL
        ) as Gemma4InferenceResult,
        scoreRows
      };
    });
    queue = task.catch(() => {});
    return signal ? waitForAbort(() => task, signal) : task;
  }

  return {
    async systemOne(state: unknown, questions: unknown, options: unknown = {}): Promise<Gemma4InferenceResult> {
      requireOpen();
      const signal = requestSignal(options);
      const packed = packStrandsGemma4Request(
        tokenizer,
        state,
        questions,
        STRANDS_GEMMA4_E2B_CONTEXT
      ) as Gemma4PackedRequest;
      signal?.throwIfAborted();
      return (await infer(packed, signal)).result;
    },

    async decide(request: unknown, options: unknown = {}): Promise<Gemma4DecisionResult> {
      requireOpen();
      const signal = requestSignal(options);
      const checked = validateRequestFields(request) as Gemma4CheckedRequest;
      const questions = genericQuestions(checked);
      const packed = packStrandsGemma4Request(
        tokenizer,
        checked.state,
        questions,
        STRANDS_GEMMA4_E2B_CONTEXT
      ) as Gemma4PackedRequest;
      signal?.throwIfAborted();
      const {result, scoreRows} = await infer(packed, signal);
      return {
        tokenCount: result.tokenCount,
        answers: checked.questions.map((question, index) => {
          const answer = result.answers[String(index)] as Gemma4ChoiceAnswer;
          return {
            selected: answer.choice,
            selectedIndex: (question.options as string[]).indexOf(answer.choice),
            probabilities: (question.options as string[]).map((option, optionIndex) => ({
              option,
              probability: answer.probabilities[option] as number,
              logit: scoreRows[index]![optionIndex]!
            }))
          };
        })
      };
    },

    async releaseCachedMemory(): Promise<unknown> {
      requireOpen();
      const task = queue.then(() => (runner as Gemma4Runner).releaseCachedMemory());
      queue = task.catch(() => {});
      return task;
    },

    dispose(): Promise<unknown> {
      if (disposal) return disposal;
      closed = true;
      disposal = queue.then(() => (runner as Gemma4Runner).dispose());
      return disposal.finally(() => {
        queue = Promise.resolve();
        tokenizer = runner = null;
      });
    }
  };
}
