import {prepareKananaCandidates, truncateKananaQueryWithCandidateFallback} from './candidates.js';
import {KANANA_MODEL_ID} from './contract.mjs';
import {finishKananaSystemOne, prepareKananaSystemOne} from './protocol.js';
import {KANANA_DECISION_TEMPERATURE, dotKananaVectors, poolKananaHiddenStates, scaleKananaLogits} from './scoring.js';
import {KANANA_MAX_LENGTH} from './truncation.js';

type DecisionTokenizer = {
  encode(text: string, options?: {add_special_tokens?: boolean}): {ids: ArrayLike<number>};
  decode?(ids: number[], options?: {skip_special_tokens?: boolean}): string;
};

type ImageGrid = readonly [temporal: number, height: number, width: number];

export type KananaDecisionInferenceInput = Readonly<{
  inputIds: Int32Array;
  attentionMask: Int32Array;
  /** Original image source objects in prompt order. Candidate documents receive an empty list. */
  image: readonly unknown[];
  signal?: AbortSignal;
}>;

export type KananaDecisionInferenceOutput = Readonly<{
  shape: readonly number[];
  dtype: 'float32';
  values: ArrayLike<number>;
}>;

export type KananaDecisionSessionOptions = Readonly<{
  tokenizer: DecisionTokenizer;
  infer(
    input: KananaDecisionInferenceInput
  ): KananaDecisionInferenceOutput | PromiseLike<KananaDecisionInferenceOutput>;
  /** Resolve the pinned preprocessor's merged image grid without replacing the original source passed to infer(). */
  imageTokenThw?: (image: unknown, index: number) => ImageGrid | PromiseLike<ImageGrid>;
  embeddingDimension?: number;
  dispose?: () => void | PromiseLike<void>;
}>;

export type KananaDecisionRequestOptions = Readonly<{
  signal?: AbortSignal;
  maxLength?: number;
  truncateInputs?: boolean;
  truncateQuestionCandidates?: boolean;
  temperature?: number;
}>;

export interface KananaDecisionSession {
  systemOne(request: unknown, options?: KananaDecisionRequestOptions): Promise<Record<string, unknown>>;
  decide(request: unknown, options?: KananaDecisionRequestOptions): Promise<Record<string, unknown>>;
  dispose(): Promise<void>;
}

class KananaDecisionSessionError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = 'KananaDecisionSessionError';
    this.code = code;
  }
}

function fail(code: string, message: string): never {
  throw new KananaDecisionSessionError(code, message);
}

function validateAbortSignal(signal: unknown): asserts signal is AbortSignal | undefined {
  if (
    signal !== undefined &&
    (!signal ||
      typeof signal !== 'object' ||
      typeof (signal as AbortSignal).aborted !== 'boolean' ||
      typeof (signal as AbortSignal).addEventListener !== 'function' ||
      typeof (signal as AbortSignal).removeEventListener !== 'function')
  )
    throw new TypeError('signal must be an AbortSignal.');
}

function abortReason(signal: AbortSignal): unknown {
  if (signal.reason !== undefined) return signal.reason;
  const error = new Error('The operation was aborted.');
  error.name = 'AbortError';
  return error;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) throw abortReason(signal);
}

function snapshotGrid(value: unknown, index: number): ImageGrid {
  if (
    !Array.isArray(value) ||
    value.length !== 3 ||
    !value.every((dimension) => Number.isSafeInteger(dimension) && dimension > 0)
  )
    throw new TypeError(`imageTokenThw for image ${index} must contain three positive safe integers.`);
  const [temporal, height, width] = value as [number, number, number];
  if (!Number.isSafeInteger(temporal * height * width))
    throw new RangeError(`imageTokenThw for image ${index} exceeds the safe integer range.`);
  return Object.freeze([temporal, height, width]);
}

function validateInferenceOutput(output: KananaDecisionInferenceOutput): void {
  if (!output || output.dtype !== 'float32' || !Array.isArray(output.shape))
    fail('inference_output', 'Kanana inference must return float32 hidden states and a shape.');
}

function normalizeImageSources(prepared: ReturnType<typeof prepareKananaSystemOne>): readonly unknown[] {
  const sources = [...prepared.images, ...prepared.stateImages].map(({image}) => image);
  return Object.freeze(sources);
}

/** Create the serialized decision layer around an injected Kanana hidden-state inference function. */
export function createKananaDecisionSession({
  tokenizer,
  infer,
  imageTokenThw,
  embeddingDimension,
  dispose: disposeInference
}: KananaDecisionSessionOptions): KananaDecisionSession {
  if (!tokenizer || typeof tokenizer.encode !== 'function')
    throw new TypeError('A Kanana tokenizer with encode() is required.');
  if (typeof infer !== 'function') throw new TypeError('A Kanana hidden-state infer() function is required.');
  if (imageTokenThw !== undefined && typeof imageTokenThw !== 'function')
    throw new TypeError('imageTokenThw must be a function when provided.');
  if (embeddingDimension !== undefined && (!Number.isSafeInteger(embeddingDimension) || embeddingDimension < 1))
    throw new RangeError('embeddingDimension must be a positive safe integer.');
  if (disposeInference !== undefined && typeof disposeInference !== 'function')
    throw new TypeError('dispose must be a function when provided.');

  let closed = false;
  let queue = Promise.resolve();
  let disposal: Promise<void> | undefined;

  async function run(
    prepared: ReturnType<typeof prepareKananaSystemOne>,
    controls: {
      signal?: AbortSignal;
      maxLength: number;
      truncateInputs: boolean;
      truncateQuestionCandidates: boolean;
      temperature: number;
    }
  ): Promise<Record<string, unknown>> {
    const signal = controls.signal;
    throwIfAborted(signal);

    const imageSources = normalizeImageSources(prepared);
    const imageGrids: ImageGrid[] = [];
    if (imageSources.length && !imageTokenThw)
      fail('image_preprocessor', 'Kanana image requests require an imageTokenThw resolver.');
    for (let index = 0; index < imageSources.length; index++) {
      throwIfAborted(signal);
      imageGrids.push(snapshotGrid(await imageTokenThw!(imageSources[index], index), index));
      throwIfAborted(signal);
    }

    const candidateEntries = prepareKananaCandidates(tokenizer, prepared, {
      maxLength: controls.maxLength,
      truncateInputs: controls.truncateInputs
    });
    const logitRows: number[][] = [];
    let inputTokens = 0;
    let resolvedHiddenDimension: number | undefined;

    async function embed(inputIds: readonly number[], image: readonly unknown[]): Promise<Float32Array> {
      throwIfAborted(signal);
      if (!inputIds.length) fail('invalid_input', 'Kanana embedding input must contain at least one token.');
      const input = Int32Array.from(inputIds);
      const attentionMask = new Int32Array(input.length).fill(1);
      const inferenceInput: {
        inputIds: Int32Array;
        attentionMask: Int32Array;
        image: readonly unknown[];
        signal?: AbortSignal;
      } = {inputIds: input, attentionMask, image};
      if (signal) inferenceInput.signal = signal;
      const output = await infer(Object.freeze(inferenceInput));
      throwIfAborted(signal);
      validateInferenceOutput(output);
      const hiddenDimension = output.shape[2];
      if (typeof hiddenDimension !== 'number' || !Number.isSafeInteger(hiddenDimension) || hiddenDimension < 1)
        fail('inference_output', 'Kanana inference returned an invalid hidden dimension.');
      if (resolvedHiddenDimension === undefined) resolvedHiddenDimension = hiddenDimension;
      if (hiddenDimension !== resolvedHiddenDimension)
        fail('inference_output', 'Kanana inference hidden dimension changed between embeddings.');
      const pooled = poolKananaHiddenStates(
        output.values,
        attentionMask,
        output.shape as [number, number, number],
        embeddingDimension ?? hiddenDimension
      );
      return pooled.values;
    }

    for (let questionIndex = 0; questionIndex < prepared.questionEntries.length; questionIndex++) {
      throwIfAborted(signal);
      const entry = prepared.questionEntries[questionIndex]!;
      const candidates = candidateEntries[questionIndex];
      if (!candidates || candidates.id !== entry.id)
        fail('candidate_order', 'Kanana candidates must preserve question order.');

      const query = truncateKananaQueryWithCandidateFallback(
        tokenizer,
        prepared,
        entry.id,
        imageGrids.map((grid) => [...grid]),
        {
          maxLength: controls.maxLength,
          truncateInputs: controls.truncateInputs,
          truncateQuestionCandidates: controls.truncateQuestionCandidates
        }
      );
      const queryVector = await embed(query.inputIds, imageSources);
      inputTokens += query.tokenCount;

      const similarities: number[] = [];
      for (let candidateIndex = 0; candidateIndex < candidates.candidateDocuments.length; candidateIndex++) {
        throwIfAborted(signal);
        const document = candidates.candidateDocuments[candidateIndex]!;
        if (document.key !== entry.candidateKeys[candidateIndex])
          fail('candidate_order', `Kanana candidate order changed for question ${entry.id}.`);
        const candidateVector = await embed(document.inputIds, []);
        similarities.push(dotKananaVectors(queryVector, candidateVector));
        inputTokens += document.tokenCount;
      }
      logitRows.push(scaleKananaLogits(similarities, KANANA_DECISION_TEMPERATURE, controls.temperature));
      if (!Number.isSafeInteger(inputTokens)) fail('usage', 'Kanana input token usage exceeds the safe integer range.');
    }

    throwIfAborted(signal);
    return finishKananaSystemOne(prepared, logitRows, KANANA_MODEL_ID, {
      billing_units: prepared.questionEntries.length,
      input_tokens: inputTokens,
      output_tokens: 0
    });
  }

  async function systemOne(
    request: unknown,
    options: KananaDecisionRequestOptions = {}
  ): Promise<Record<string, unknown>> {
    if (closed) fail('SESSION_CLOSED', 'Kanana decision session has been disposed.');
    if (!options || typeof options !== 'object' || Array.isArray(options))
      throw new TypeError('Kanana request options must be an object.');
    validateAbortSignal(options.signal);
    if (options.truncateInputs !== undefined && typeof options.truncateInputs !== 'boolean')
      throw new TypeError('truncateInputs must be a boolean.');
    if (options.truncateQuestionCandidates !== undefined && typeof options.truncateQuestionCandidates !== 'boolean')
      throw new TypeError('truncateQuestionCandidates must be a boolean.');

    const prepared = prepareKananaSystemOne(request);
    const temperature = options.temperature === undefined ? prepared.temperature : options.temperature;
    scaleKananaLogits([], KANANA_DECISION_TEMPERATURE, temperature);
    const controls = {
      ...(options.signal ? {signal: options.signal} : {}),
      maxLength: options.maxLength ?? KANANA_MAX_LENGTH,
      truncateInputs: options.truncateInputs ?? true,
      truncateQuestionCandidates: options.truncateQuestionCandidates ?? true,
      temperature
    };

    const task = queue.then(() => run(prepared, controls));
    queue = task.then(
      () => undefined,
      () => undefined
    );
    return task;
  }

  return {
    systemOne,
    decide: systemOne,
    dispose() {
      if (!disposal) {
        closed = true;
        disposal = queue.then(async () => {
          await disposeInference?.();
        });
      }
      return disposal;
    }
  };
}
