import {waitForAbort} from '../assets.js';
import {validateRequestFields} from '../request-fields.mjs';
import {WebDecisionError} from '../kev.js';
import {SAFEDECIDE_MODEL} from './contract.mjs';
import {finishSafeDecideSystemOne, prepareSafeDecideSystemOne} from './protocol.mjs';

type SafeDecideAlpha = 0.05 | 0.1 | 0.2;

interface SafeDecideRunner {
  run(rows: unknown[], signal?: AbortSignal | null): Promise<number[]>;
  dispose(): Promise<void>;
}

interface SafeDecideTokenizer {
  encodePair(first: string, second: string): unknown;
}

interface SafeDecideRule {
  readonly alpha: SafeDecideAlpha;
  readonly threshold: number | null;
  readonly delta: number;
  readonly statement: string;
}

interface SafeDecideConfig {
  readonly temperatures: Record<string, number>;
  readonly selective_rules: SafeDecideRule[];
  readonly selective_rule: SafeDecideRule;
}

interface SafeDecideQuestion {
  readonly instructions: string;
  readonly options: string[];
}

interface SafeDecideRequestSnapshot {
  readonly state: unknown;
  readonly questions: SafeDecideQuestion[];
}

interface SafeDecideCheckedRequest extends SafeDecideRequestSnapshot {
  readonly offsets: number[];
}

interface SafeDecideAnswerFields {
  readonly probabilities: Record<string, number>;
  readonly confidence: number;
  readonly action: 'automate' | 'escalate';
}

interface SafeDecideChoiceAnswer extends SafeDecideAnswerFields {
  readonly type: 'choice';
  readonly choice: string;
}

interface SafeDecideGuarantee {
  readonly alpha: SafeDecideAlpha;
  readonly delta: number;
  readonly statement: string;
}

type SafeDecideSystemOneResult = ReturnType<typeof finishSafeDecideSystemOne>;

interface SafeDecideChoiceResult extends Omit<SafeDecideSystemOneResult, 'answers'> {
  readonly answers: Record<string, SafeDecideChoiceAnswer>;
}

interface SafeDecideDecisionResult {
  readonly tokenCount: number;
  readonly outputTokenCount: number;
  readonly truncated?: true;
  readonly truncatedQuestions?: string[];
  readonly guarantee: SafeDecideGuarantee;
  readonly answers: Array<{
    readonly selected: string;
    readonly selectedIndex: number;
    readonly probabilities: Array<{readonly option: string; readonly probability: number; readonly logit: number}>;
    readonly confidence: number;
    readonly action: 'automate' | 'escalate';
  }>;
}

type SafeDecidePrepared = ReturnType<typeof prepareSafeDecideSystemOne>;

interface SafeDecideInference {
  readonly logits: number[];
  readonly result: SafeDecideSystemOneResult;
}

interface SafeDecideCallOptions {
  readonly signal: AbortSignal | null | undefined;
  readonly alpha: SafeDecideAlpha | undefined;
}

interface SafeDecideSignalCandidate {
  readonly throwIfAborted?: unknown;
  readonly addEventListener?: unknown;
  readonly removeEventListener?: unknown;
}

interface SafeDecideSessionOptions {
  readonly runner: SafeDecideRunner;
  readonly tokenizer: SafeDecideTokenizer;
  readonly config: SafeDecideConfig;
  readonly model?: string;
}

const invalid = (message: string): never => {
  throw new WebDecisionError('INVALID_INPUT', message);
};
const closedError = (): WebDecisionError =>
  new WebDecisionError('SESSION_CLOSED', 'The SafeDecide model has been disposed.');
const ALLOWED_ALPHA: readonly SafeDecideAlpha[] = Object.freeze([0.05, 0.1, 0.2]);

function validateCallOptions(options: unknown): SafeDecideCallOptions {
  if (options == null) return {signal: undefined, alpha: undefined};
  if (typeof options !== 'object' || Array.isArray(options)) invalid('SafeDecide call options must be an object.');
  const values = Object.create(null) as Record<string, unknown>;
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (
      typeof key !== 'string' ||
      !['signal', 'alpha'].includes(key) ||
      !descriptor ||
      !Object.hasOwn(descriptor, 'value')
    )
      invalid('SafeDecide call options contain an unsupported field.');
    values[key as string] = descriptor!.value as unknown;
  }
  const signal = values.signal;
  if (
    signal != null &&
    (typeof (signal as SafeDecideSignalCandidate).throwIfAborted !== 'function' ||
      typeof (signal as SafeDecideSignalCandidate).addEventListener !== 'function' ||
      typeof (signal as SafeDecideSignalCandidate).removeEventListener !== 'function')
  )
    invalid('SafeDecide signal must be an AbortSignal.');
  const alpha = values.alpha == null ? undefined : values.alpha;
  if (alpha !== undefined && !ALLOWED_ALPHA.includes(alpha as SafeDecideAlpha))
    invalid(`SafeDecide alpha must be one of ${ALLOWED_ALPHA.join(', ')}.`);
  return {signal: signal as AbortSignal | null | undefined, alpha: alpha as SafeDecideAlpha | undefined};
}

function decisionQuestions(questions: SafeDecideQuestion[]): Record<string, unknown> {
  return Object.fromEntries(
    questions.map((question, index) => [
      String(index),
      {
        type: 'choice',
        instructions: question.instructions,
        criteria: Object.fromEntries(question.options.map((name) => [name, null]))
      }
    ])
  );
}

function finishDecisionResult(
  checked: SafeDecideCheckedRequest,
  typed: SafeDecideChoiceResult,
  logits: number[]
): SafeDecideDecisionResult {
  return {
    tokenCount: typed.tokenCount,
    outputTokenCount: 0,
    ...(typed.truncated ? {truncated: true, truncatedQuestions: typed.truncatedQuestions} : {}),
    guarantee: typed.guarantee,
    answers: checked.questions.map((question, index) => {
      const answer = typed.answers[String(index)]!;
      return {
        selected: answer.choice,
        selectedIndex: question.options.indexOf(answer.choice),
        probabilities: question.options.map((option, optionIndex) => ({
          option,
          probability: answer.probabilities[option]!,
          logit: logits[checked.offsets[index]! + optionIndex]!
        })),
        confidence: answer.confidence,
        action: answer.action
      };
    })
  };
}

export function createSafeDecideSession({
  runner,
  tokenizer,
  config,
  model = SAFEDECIDE_MODEL
}: SafeDecideSessionOptions): {
  systemOne(state: unknown, questions: unknown, options?: unknown): Promise<SafeDecideSystemOneResult>;
  decide(request: unknown, options?: unknown): Promise<SafeDecideDecisionResult>;
  releaseCachedMemory(): Promise<void>;
  dispose(): Promise<void>;
} {
  if (
    typeof runner?.run !== 'function' ||
    typeof runner?.dispose !== 'function' ||
    typeof tokenizer?.encodePair !== 'function' ||
    !config?.temperatures ||
    !Array.isArray(config.selective_rules) ||
    !config.selective_rule ||
    typeof model !== 'string'
  )
    throw new TypeError('SafeDecide session components are required.');

  let queue: Promise<void | SafeDecideInference> = Promise.resolve();
  let closed = false;
  let disposing: Promise<void> | undefined;

  function prepare(state: unknown, questions: unknown): SafeDecidePrepared {
    const prepared = prepareSafeDecideSystemOne(state, questions, tokenizer) as SafeDecidePrepared;
    return prepared;
  }

  function infer(
    prepared: SafeDecidePrepared,
    signal: AbortSignal | null | undefined,
    alpha: SafeDecideAlpha | undefined
  ): Promise<SafeDecideInference> {
    if (closed) throw closedError();
    signal?.throwIfAborted();
    const task = queue.then(async () => {
      if (closed) throw closedError();
      signal?.throwIfAborted();
      const logits = await runner.run(prepared.rows, signal);
      signal?.throwIfAborted();
      return {
        logits,
        result: finishSafeDecideSystemOne(prepared, logits, config, model, alpha) as SafeDecideSystemOneResult
      };
    });
    queue = task.catch(() => {});
    return signal ? waitForAbort(() => task, signal) : task;
  }

  async function systemOne(
    state: unknown,
    questions: unknown,
    options: unknown = {}
  ): Promise<SafeDecideSystemOneResult> {
    if (closed) throw closedError();
    const {signal, alpha} = validateCallOptions(options);
    signal?.throwIfAborted();
    return (await infer(prepare(state, questions), signal, alpha)).result;
  }

  return {
    systemOne,
    async decide(request: unknown, options: unknown = {}): Promise<SafeDecideDecisionResult> {
      if (closed) throw closedError();
      const {signal, alpha} = validateCallOptions(options);
      signal?.throwIfAborted();
      const snapshot = validateRequestFields(request) as SafeDecideRequestSnapshot;
      if (typeof snapshot.state !== 'string' || snapshot.questions.length < 1)
        invalid('SafeDecide decision requests need a string state and at least one question.');
      for (const [index, question] of snapshot.questions.entries()) {
        if (
          typeof question.instructions !== 'string' ||
          !Array.isArray(question.options) ||
          question.options.length < 2 ||
          question.options.some((option) => typeof option !== 'string') ||
          new Set(question.options).size !== question.options.length
        )
          invalid(`SafeDecide decision question ${index} needs at least 2 distinct string options.`);
      }
      const typed = await infer(prepare(snapshot.state, decisionQuestions(snapshot.questions)), signal, alpha);
      const offsets: number[] = [];
      let offset = 0;
      for (const question of snapshot.questions) {
        offsets.push(offset);
        offset += question.options.length;
      }
      return finishDecisionResult({...snapshot, offsets}, typed.result as SafeDecideChoiceResult, typed.logits);
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
