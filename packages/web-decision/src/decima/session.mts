import {waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {validateRequestFields} from '../request-fields.mjs';
import {DECIMA_SMALL_MODEL} from './contract.mjs';
import {
  decodeDecimaScores,
  finishDecimaChoice,
  validateDecimaRequest,
  type DecimaChoiceResult,
  type DecimaInferenceKind,
  type DecimaOrdinalConfig,
  type DecimaRawScores,
  type DecimaRuntimeConfig
} from './protocol.mjs';
import {prepareDecimaQuestion} from './tokenizer.mjs';
import {finishDecimaSystemOne, prepareDecimaSystemOne} from './typed.mjs';

interface TensorBuffer {
  readonly byteLength: number;
}

interface EncodedInputs {
  readonly states: {readonly data: TensorBuffer};
  readonly mask: {readonly data: TensorBuffer};
}

interface DecimaRunner {
  encode(rows: readonly (readonly number[])[]): Promise<EncodedInputs>;
  scoreEncoded(state: EncodedInputs, choices: EncodedInputs, choiceCount: number): Promise<DecimaRawScores>;
  dispose(): Promise<void>;
}

interface DecimaTokenizer {
  readonly special: {
    readonly clsId: number;
    readonly sepId: number;
    readonly padId: number;
  };
  encode(text: string): number[];
}

interface DecimaConfig extends DecimaRuntimeConfig {
  readonly ordinal: DecimaOrdinalConfig;
  readonly state_prefix: string;
  readonly choice_prefix: string;
  readonly max_state_tokens: number;
  readonly max_choice_tokens: number;
  readonly choice_fit?: boolean | undefined;
  readonly state_overflow?: string | undefined;
  readonly supported_question_types: DecimaInferenceKind[];
}

interface PreparedQuestion {
  readonly stateIds: number[];
  readonly choiceIds: number[][];
  readonly choices: string[];
  readonly tokenCount: number;
  readonly truncated: boolean;
}

interface ValidatedDecisionRequest {
  readonly state: string;
  readonly questions: readonly Readonly<{
    instructions: string;
    options: readonly string[];
  }>[];
}

interface PreparedSystemOneQuestion {
  readonly id: string;
  readonly instructions: string;
  readonly labels: string[];
  readonly choices: string[];
  readonly type: DecimaInferenceKind;
  readonly outputType: DecimaInferenceKind;
}

interface PreparedSystemOne {
  readonly state: string;
  readonly questions: readonly PreparedSystemOneQuestion[];
}

interface InferenceOptions {
  readonly signal?: AbortSignal | null | undefined;
  readonly language?: string | undefined;
}

interface InferenceResult {
  readonly probabilities: number[];
  readonly scores: readonly number[];
  readonly logits: number[];
  readonly tokenCount: number;
  readonly truncated: boolean;
}

export interface DecimaDecisionResult {
  readonly tokenCount: number;
  readonly truncated?: true;
  readonly answers: DecimaChoiceResult[];
}

export interface DecimaSystemOneResult {
  readonly model: string;
  readonly tokenCount: number;
  readonly truncated?: true;
  readonly answers: Readonly<Record<string, unknown>>;
}

export interface DecimaSession {
  decide(request: unknown, options?: unknown): Promise<DecimaDecisionResult>;
  systemOne(state: unknown, questions: unknown, options?: unknown): Promise<DecimaSystemOneResult>;
  releaseCachedMemory(): Promise<void>;
  dispose(): Promise<void>;
}

interface DecimaSessionOptions {
  readonly runner: DecimaRunner | null | undefined;
  readonly tokenizer: DecimaTokenizer | null | undefined;
  readonly config: DecimaConfig | null | undefined;
  readonly model?: string | undefined;
}

const OPTION_CACHE_LIMIT = 16 * 1024 * 1024;
const closedError = (): WebDecisionError => new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
const invalid = (message: string): WebDecisionError => new WebDecisionError('INVALID_INPUT', message);

export function createDecimaSession({
  runner,
  tokenizer,
  config,
  model = DECIMA_SMALL_MODEL
}: DecimaSessionOptions): DecimaSession {
  if (
    typeof runner?.encode !== 'function' ||
    typeof runner?.scoreEncoded !== 'function' ||
    typeof runner?.dispose !== 'function' ||
    typeof tokenizer?.encode !== 'function' ||
    !config?.ordinal ||
    typeof model !== 'string'
  )
    throw new TypeError('Decima session components are required.');
  let queue: Promise<unknown> = Promise.resolve();
  let closed = false;
  let disposing: Promise<void> | undefined;
  let choiceCacheBytes = 0;
  const choiceCache = new Map<string, {readonly rows: EncodedInputs; readonly bytes: number}>();

  const requireOpen = (): void => {
    if (closed) throw closedError();
  };

  function enqueue<T>(task: () => T | PromiseLike<T>): Promise<T> {
    const current = queue.then(task);
    queue = current.catch(() => {});
    return current;
  }

  function choiceKey(prepared: PreparedQuestion): string {
    return JSON.stringify([prepared.choiceIds, prepared.choices]);
  }

  function inferenceOptions(options: unknown): InferenceOptions {
    if (options == null) return {};
    if (typeof options !== 'object' || Array.isArray(options))
      throw invalid('Decima inference options must be an object.');
    const allowed = new Set(['signal', 'language']);
    const values: Record<string, unknown> = Object.create(null);
    const ownKeys = new Set(Reflect.ownKeys(options));
    for (const key of Reflect.ownKeys(options)) {
      const descriptor = Object.getOwnPropertyDescriptor(options, key);
      if (typeof key !== 'string' || !allowed.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
        throw invalid('Unsupported Decima inference option.');
      values[key] = descriptor.value;
    }
    for (const key in options)
      if (!ownKeys.has(key)) throw invalid('Inherited Decima inference options are unsupported.');
    const signal = values.signal as AbortSignal | null | undefined;
    const language = values.language as string | undefined;
    if (
      signal != null &&
      (typeof signal.throwIfAborted !== 'function' ||
        typeof signal.addEventListener !== 'function' ||
        typeof signal.removeEventListener !== 'function')
    )
      throw invalid('Decima signal must be an AbortSignal.');
    if (language !== undefined && (typeof language !== 'string' || !language.trim()))
      throw invalid('Decima language must be a non-empty string.');
    return {signal, language: language ?? 'en'};
  }

  async function cachedChoices(prepared: PreparedQuestion): Promise<EncodedInputs> {
    const key = choiceKey(prepared);
    const cached = choiceCache.get(key);
    if (cached) {
      choiceCache.delete(key);
      choiceCache.set(key, cached);
      return cached.rows;
    }
    const encoded = await runner!.encode(prepared.choiceIds);
    const bytes = encoded.states.data.byteLength + encoded.mask.data.byteLength;
    if (bytes > OPTION_CACHE_LIMIT) return encoded;
    choiceCache.set(key, {rows: encoded, bytes});
    choiceCacheBytes += bytes;
    while (choiceCacheBytes > OPTION_CACHE_LIMIT) {
      const [oldest, value] = choiceCache.entries().next().value!;
      choiceCache.delete(oldest);
      choiceCacheBytes -= value.bytes;
    }
    return encoded;
  }

  async function infer(
    prepared: PreparedQuestion,
    kind: DecimaInferenceKind,
    signal?: AbortSignal | null
  ): Promise<InferenceResult> {
    requireOpen();
    const schedule = () =>
      enqueue(async () => {
        requireOpen();
        signal?.throwIfAborted();
        const choices = await cachedChoices(prepared);
        const state = await runner!.encode([prepared.stateIds]);
        const raw = await runner!.scoreEncoded(state, choices, prepared.choiceIds.length);
        const probabilities = decodeDecimaScores(raw, config!, kind);
        return {
          probabilities,
          scores: raw.scores,
          logits: raw.scores.map((score) => score / config!.temperature),
          tokenCount: prepared.tokenCount,
          truncated: prepared.truncated
        };
      });
    return signal ? waitForAbort(schedule, signal) : schedule();
  }

  function pack(
    state: string,
    instructions: string,
    choices: readonly string[],
    language: string | undefined = 'en',
    allowDuplicates = false
  ): PreparedQuestion {
    return prepareDecimaQuestion(
      state,
      instructions,
      choices,
      config!,
      tokenizer!,
      language,
      allowDuplicates
    ) as PreparedQuestion;
  }

  return {
    async decide(request: unknown, options: unknown = {}): Promise<DecimaDecisionResult> {
      requireOpen();
      const {signal, language} = inferenceOptions(options);
      const checked = validateRequestFields(request) as ValidatedDecisionRequest;
      validateDecimaRequest(checked);
      const prepared = checked.questions.map((question) =>
        pack(checked.state, question.instructions, question.options, language)
      );
      const answers: DecimaChoiceResult[] = [];
      for (const item of prepared) {
        const result = await infer(item, 'choice', signal);
        const answer = finishDecimaChoice(item, result.probabilities, result.logits);
        answers.push(answer);
      }
      return {
        tokenCount: prepared.reduce((sum, item) => sum + item.tokenCount, 0),
        ...(prepared.some((item) => item.truncated) ? {truncated: true} : {}),
        answers
      };
    },
    async systemOne(state: unknown, questions: unknown, options: unknown = {}): Promise<DecimaSystemOneResult> {
      requireOpen();
      const {signal, language} = inferenceOptions(options);
      const prepared = prepareDecimaSystemOne(
        state,
        questions,
        language,
        config!.supported_question_types
      ) as PreparedSystemOne;
      const answers: InferenceResult[] = [];
      for (const question of prepared.questions) {
        const input = pack(
          prepared.state,
          question.instructions,
          question.choices,
          language,
          question.outputType === 'score'
        );
        const result = await infer(input, question.type, signal);
        answers.push({...result, tokenCount: input.tokenCount});
      }
      return finishDecimaSystemOne(prepared, answers, model) as DecimaSystemOneResult;
    },
    async releaseCachedMemory(): Promise<void> {
      requireOpen();
      return enqueue(() => {
        choiceCache.clear();
        choiceCacheBytes = 0;
      });
    },
    dispose(): Promise<void> {
      if (disposing) return disposing;
      closed = true;
      disposing = queue.then(async () => {
        choiceCache.clear();
        choiceCacheBytes = 0;
        await runner!.dispose();
        runner = tokenizer = config = null;
      });
      return disposing;
    }
  };
}
