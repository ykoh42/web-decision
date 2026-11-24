import {waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {validateH2oSignal} from '../h2o/signal.mjs';
import {readKananaFloat} from '../kanana/json-float.ts';
import {
  renderStrandsV2,
  type Gemma4V2Content,
  type Gemma4V2Question,
  type Gemma4V2Dictionary
} from './gemma4-v2-protocol.mjs';
import {finishGemma4V2Answer, type Gemma4V2ScoringKernel} from './gemma4-v2-scoring.mjs';
import {qwen35OrderedRecord} from './qwen35-protocol.mjs';

export interface Gemma4V2Request {
  readonly model?: string;
  readonly state?: Gemma4V2Content;
  readonly questions: Gemma4V2Dictionary<Gemma4V2Question>;
  readonly images?: readonly string[];
  readonly audio?: unknown;
}

export interface Gemma4V2PreparedImage {
  readonly tokens: number;
  /** Host-owned image data consumed by the graph runner. */
  readonly value: unknown;
}

export interface Gemma4V2RunnerInput {
  readonly stateIds: readonly number[];
  readonly questionIds: readonly number[];
  readonly imageSpans: readonly Readonly<{start: number; end: number}>[];
  readonly images: readonly unknown[];
}

export interface Gemma4V2OwnedRunner {
  run(input: Gemma4V2RunnerInput, signal?: AbortSignal | null): Promise<ArrayLike<number>>;
  releaseCachedMemory(): Promise<unknown> | unknown;
  dispose(): Promise<unknown> | unknown;
}

export interface Gemma4V2SessionOptions {
  readonly tokenizer: {encode(text: string, options: {add_special_tokens: false}): {ids: number[]}};
  readonly runner: Gemma4V2OwnedRunner;
  readonly prepareImage: (image: string, signal?: AbortSignal | null) => Promise<Gemma4V2PreparedImage>;
  readonly modelName: string;
  readonly maxLength?: number;
  readonly temperature: number;
  readonly softcap: number;
  readonly ordinalSmoothing?: number;
  readonly scoringKernel?: Gemma4V2ScoringKernel;
}

export interface Gemma4V2CallOptions {
  readonly signal?: AbortSignal | null;
  readonly maxLength?: number;
}

const invalid = (message: string) => new WebDecisionError('INVALID_INPUT', message);
const closedError = () => new WebDecisionError('SESSION_CLOSED', 'The Strands Gemma 4 v2 model has been disposed.');

function maxLength(value: unknown): number {
  if (!Number.isSafeInteger(value) || (value as number) < 1) throw invalid('maxLength must be a positive integer.');
  return value as number;
}

function callOptions(value: Gemma4V2CallOptions | null | undefined, defaultLength: number) {
  if (value != null && (typeof value !== 'object' || Array.isArray(value)))
    throw invalid('Session options must be an object.');
  const signal = value?.signal;
  if (signal != null) validateH2oSignal(signal);
  signal?.throwIfAborted();
  return {signal, maxLength: value?.maxLength === undefined ? defaultLength : maxLength(value.maxLength)};
}

/** Clone request data while retaining the immutable public Python-float marker. */
function snapshotValue(value: unknown, seen = new Map<object, unknown>()): unknown {
  if (readKananaFloat(value) !== undefined) return value;
  if (typeof value === 'function' || typeof value === 'symbol') return structuredClone(value);
  if (value === null || typeof value !== 'object') return value;
  if (seen.has(value)) return seen.get(value);
  if (value instanceof Map) {
    const result = new Map<unknown, unknown>();
    seen.set(value, result);
    for (const [key, item] of value) result.set(snapshotValue(key, seen), snapshotValue(item, seen));
    return result;
  }
  if (Array.isArray(value) || Object.prototype.toString.call(value) === '[object Object]') {
    const result: Record<string, unknown> | unknown[] = Array.isArray(value) ? [] : {};
    if (Array.isArray(result) && Array.isArray(value)) result.length = value.length;
    seen.set(value, result);
    for (const key of Object.keys(value))
      Object.defineProperty(result, key, {
        value: snapshotValue((value as Record<string, unknown>)[key], seen),
        enumerable: true,
        configurable: true,
        writable: true
      });
    return result;
  }
  const result = structuredClone(value);
  seen.set(value, result);
  return result;
}

function snapshotRequest(request: Gemma4V2Request): Gemma4V2Request {
  if (!request || typeof request !== 'object' || Array.isArray(request))
    throw invalid('A System One request is required.');
  let snapshot: Gemma4V2Request;
  try {
    snapshot = snapshotValue(request) as Gemma4V2Request;
  } catch (cause) {
    throw new WebDecisionError('INVALID_INPUT', 'The request must contain cloneable data.', {cause});
  }
  if (snapshot.audio != null) throw invalid('Audio input is not supported by this model.');
  if (snapshot.model !== undefined && typeof snapshot.model !== 'string') throw invalid('Model must be a string.');
  if (
    snapshot.images !== undefined &&
    (!Array.isArray(snapshot.images) || snapshot.images.some((item) => typeof item !== 'string'))
  )
    throw invalid('Images must be base64 strings.');
  return snapshot;
}

/** System One orchestration around an owned, model-specific TVM graph runner. */
export function createGemma4V2Session(options: Gemma4V2SessionOptions) {
  if (
    typeof options?.tokenizer?.encode !== 'function' ||
    typeof options?.runner?.run !== 'function' ||
    typeof options.runner.releaseCachedMemory !== 'function' ||
    typeof options.runner.dispose !== 'function' ||
    typeof options.prepareImage !== 'function' ||
    typeof options.modelName !== 'string' ||
    !options.modelName
  )
    throw new TypeError('Gemma 4 v2 needs a tokenizer, image preparer, model name, and owned runner.');
  if (
    !Number.isFinite(options.temperature) ||
    options.temperature <= 0 ||
    !Number.isFinite(options.softcap) ||
    options.softcap <= 0 ||
    (options.ordinalSmoothing !== undefined &&
      (!Number.isFinite(options.ordinalSmoothing) || options.ordinalSmoothing < 0))
  )
    throw new TypeError('Invalid Gemma 4 v2 decision configuration.');
  const defaultLength = maxLength(options.maxLength ?? 32768);
  const modelName = options.modelName;
  const scoring = {
    temperature: options.temperature,
    softcap: options.softcap,
    ...(options.scoringKernel === undefined ? {} : {kernel: options.scoringKernel}),
    ...(options.ordinalSmoothing === undefined ? {} : {ordinalSmoothing: options.ordinalSmoothing})
  };
  let tokenizer: Gemma4V2SessionOptions['tokenizer'] | null = options.tokenizer;
  let runner: Gemma4V2OwnedRunner | null = options.runner;
  let prepareImage: Gemma4V2SessionOptions['prepareImage'] | null = options.prepareImage;
  let queue: Promise<unknown> = Promise.resolve();
  let closed = false;
  let disposal: Promise<void> | undefined;
  const requireOpen = () => {
    if (closed || !runner || !tokenizer || !prepareImage) throw closedError();
    return {runner, tokenizer, prepareImage};
  };

  async function systemOne(request: Gemma4V2Request, inputOptions?: Gemma4V2CallOptions | null) {
    requireOpen();
    const {signal, maxLength: limit} = callOptions(inputOptions, defaultLength);
    const snapshot = snapshotRequest(request);
    const task = queue.then(async () => {
      const active = requireOpen();
      signal?.throwIfAborted();
      const images: Gemma4V2PreparedImage[] = [];
      for (const image of snapshot.images ?? []) {
        images.push(await active.prepareImage(image, signal));
        signal?.throwIfAborted();
      }
      const rendered = renderStrandsV2(
        snapshot,
        active.tokenizer,
        images.map((image) => image.tokens)
      );
      for (const question of rendered.questions) {
        const length = rendered.stateIds.length + question.questionIds.length;
        if (length > limit)
          throw invalid(
            `Question ${JSON.stringify(question.name)} exceeds the configured context length of ${limit} tokens.`
          );
      }
      const answers: Array<[string, ReturnType<typeof finishGemma4V2Answer>]> = [];
      for (const question of rendered.questions) {
        signal?.throwIfAborted();
        const logits = await active.runner.run(
          {
            stateIds: rendered.stateIds,
            questionIds: question.questionIds,
            imageSpans: rendered.images,
            images: images.map((image) => image.value)
          },
          signal
        );
        signal?.throwIfAborted();
        answers.push([question.name, finishGemma4V2Answer(question, logits, scoring)]);
      }
      return {
        model: modelName,
        answers: qwen35OrderedRecord(answers),
        usage: {input_tokens: rendered.inputTokens, output_tokens: rendered.questions.length}
      };
    });
    queue = task.catch(() => {});
    return signal ? waitForAbort(() => task, signal) : task;
  }

  async function releaseCachedMemory(): Promise<void> {
    requireOpen();
    const task = queue.then(async () => {
      const active = requireOpen();
      await active.runner.releaseCachedMemory();
    });
    queue = task.catch(() => {});
    return task;
  }

  function dispose(): Promise<void> {
    if (!disposal) {
      closed = true;
      const owned = runner;
      disposal = queue.then(async () => {
        try {
          await owned?.dispose();
        } finally {
          tokenizer = null;
          runner = null;
          prepareImage = null;
        }
      });
    }
    return disposal;
  }

  return {systemOne, releaseCachedMemory, dispose};
}
