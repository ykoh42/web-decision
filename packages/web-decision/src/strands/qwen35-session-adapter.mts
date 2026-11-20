import {
  packQwen35RequestSnapshot,
  finishQwen35Answers,
  resolveQwen35BatchOptions,
  qwen35OrderedRecord,
  snapshotQwen35Request,
  type Qwen35PackedRequest,
  type Qwen35Answer
} from './qwen35-protocol.mts';
import {
  prepareQwen35GraphInputFromPacked,
  type Qwen35GraphInputChunk,
  type Qwen35GraphInputOptions,
  type Qwen35GraphInputRecord
} from './qwen35-graph-input.mts';

export interface Qwen35GraphRunner {
  /** Return the graph's post-temperature, n_slots-masked probability rows, in question order. */
  run(
    chunk: Qwen35GraphInputChunk,
    input: Qwen35GraphInputRecord,
    signal?: AbortSignal | null
  ): unknown | PromiseLike<unknown>;
}

export interface Qwen35SystemOneResponse {
  readonly model: string;
  readonly answers: Readonly<Record<string, Qwen35Answer>>;
  readonly usage: Readonly<{input_tokens: number; output_tokens: number}>;
}

export type Qwen35SessionAdapterOptions = Qwen35GraphInputOptions &
  Readonly<{
    /** Matches EngineConfig.model_name; the request's model field does not select the loaded engine. */
    modelName: string;
  }>;

/**
 * Experimental source-faithful adapter seam for a future Qwen TVM graph runner.
 * It is deliberately not wired to the package loader or public exports until a
 * complete graph runtime exists.
 */
export async function evaluateQwen35Request(
  tokenizer: unknown,
  runner: Qwen35GraphRunner,
  request: unknown,
  options: Qwen35SessionAdapterOptions
): Promise<Qwen35SystemOneResponse> {
  if (typeof options?.modelName !== 'string' || options.modelName.length === 0)
    throw new TypeError('Qwen engine modelName must be a non-empty string.');
  if (typeof runner?.run !== 'function') throw new TypeError('Qwen graph runner is required.');

  const requestSnapshot = snapshotQwen35Request(request, options);
  // Pinned VisionEngine groups only by max_batch. Its text Engine also applies
  // max_batch_tokens, so keep that source distinction in the adapter route.
  const graphOptions: Qwen35SessionAdapterOptions = requestSnapshot.images.length
    ? {...options, maxBatchTokens: null}
    : options;
  const packed = packQwen35RequestSnapshot(tokenizer, requestSnapshot, graphOptions);
  return evaluateQwen35PackedRequest(tokenizer, runner, packed, graphOptions);
}

export async function evaluateQwen35PackedRequest(
  tokenizer: unknown,
  runner: Qwen35GraphRunner,
  packed: Qwen35PackedRequest,
  options: Qwen35SessionAdapterOptions,
  signal?: AbortSignal | null
): Promise<Qwen35SystemOneResponse> {
  if (typeof options?.modelName !== 'string' || options.modelName.length === 0)
    throw new TypeError('Qwen engine modelName must be a non-empty string.');
  if (typeof runner?.run !== 'function') throw new TypeError('Qwen graph runner is required.');

  const batch = resolveQwen35BatchOptions(options);
  const input = await prepareQwen35GraphInputFromPacked(tokenizer, packed, options);
  const chunkProbabilities: unknown[] = [];
  for (const chunk of input.chunks) {
    signal?.throwIfAborted();
    const probabilities = await runner.run(chunk, input, signal);
    signal?.throwIfAborted();
    if (!Array.isArray(probabilities) || probabilities.length !== chunk.questions.length)
      throw new TypeError('Qwen graph runner must return one masked-probability row per input question.');
    chunkProbabilities.push(probabilities);
  }

  const namedAnswers = finishQwen35Answers(input.request, chunkProbabilities, batch);
  const answers = qwen35OrderedRecord(namedAnswers.map(({name, answer}) => [name, answer] as const));
  return Object.freeze({
    model: options.modelName,
    answers,
    usage: Object.freeze({input_tokens: input.inputTokenCount, output_tokens: input.request.questions.length})
  });
}
