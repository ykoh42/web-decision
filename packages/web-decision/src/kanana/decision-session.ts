import {prepareKananaCandidates, truncateKananaQueryWithCandidateFallback} from './candidates.js';
import {KANANA_EMBEDDING_SIZE, KANANA_IMAGE_TOKEN_ID, KANANA_MODEL_ID, KANANA_MODEL_MAX_LENGTH} from './contract.mjs';
import {prepareKananaPositionInputs} from './mrope.js';
import {finishKananaSystemOne, prepareKananaSystemOne} from './protocol.js';
import {KANANA_DECISION_TEMPERATURE, dotKananaVectors, poolKananaHiddenStates, scaleKananaLogits} from './scoring.js';
import {KANANA_MAX_LENGTH} from './truncation.js';

type DecisionTokenizer = {
  encode(text: string, options?: {add_special_tokens?: boolean}): {ids: ArrayLike<number>};
  decode?(ids: number[], options?: {skip_special_tokens?: boolean}): string;
};

type ImageGrid = readonly [temporal: number, height: number, width: number];
type ImageEmbeddings = Readonly<{
  shape: readonly [tokens: number, dimension: number];
  values: ArrayLike<number>;
}>;

export type KananaDecisionInferenceInput = Readonly<{
  inputIds: Int32Array;
  attentionMask: Int32Array;
  /** Original image source objects in prompt order. Candidate documents receive an empty list. */
  image: readonly unknown[];
  /** Preprocessor merged grids corresponding one-for-one with image sources. */
  imageTokenThw: readonly ImageGrid[];
  /** Four axis-major rows with shape [4, 1, inputIds.length]. */
  positionIds: Int32Array;
  /** Prompt token offsets occupied by image tokens, in source and embedding order. */
  imagePositions: Int32Array;
  /** Vision embeddings for the prompt images, in prompt order. */
  imageEmbeddings: ImageEmbeddings;
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
  imageTokenThw?: (image: unknown, index: number, signal?: AbortSignal) => ImageGrid | PromiseLike<ImageGrid>;
  /** Encode each request's images once, then reuse their embeddings across its decision questions. */
  prepareImageEmbeddings?: (
    images: readonly unknown[],
    imageTokenThw: readonly ImageGrid[],
    signal?: AbortSignal
  ) => ImageEmbeddings | PromiseLike<ImageEmbeddings>;
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

type RunControls = Readonly<{
  signal?: AbortSignal;
  maxLength: number;
  truncateInputs: boolean;
  truncateQuestionCandidates: boolean;
  temperature: number;
}>;

type SessionInference = Pick<
  KananaDecisionSessionOptions,
  'tokenizer' | 'infer' | 'imageTokenThw' | 'prepareImageEmbeddings' | 'embeddingDimension'
>;

type CandidateGroups = ReturnType<typeof prepareKananaCandidates>;
type CandidateDocument = CandidateGroups[number]['candidateDocuments'][number];
type CandidateTruncation = Readonly<{questionId: string; keys: readonly string[]}>;
type InputTruncation = Readonly<{
  truncated: true;
  maxLength: number;
  queryQuestionIds: readonly string[];
  candidateDocuments: readonly CandidateTruncation[];
}>;
type CandidateDocumentReference = Readonly<{
  questionIndex: number;
  key: string;
  document: CandidateDocument;
}>;

type PreparedImages = Readonly<{
  sources: readonly unknown[];
  grids: readonly ImageGrid[];
  embeddings: ImageEmbeddings;
}>;

type HiddenDimension = {value?: number};

const CANDIDATE_CACHE_SIZE = 4096;

function candidateCacheKey(inputIds: readonly number[]): string {
  return JSON.stringify(inputIds);
}

function rememberCandidateEmbedding(cache: Map<string, Float32Array>, key: string, embedding: Float32Array): void {
  cache.delete(key);
  cache.set(key, embedding);
  if (cache.size > CANDIDATE_CACHE_SIZE) cache.delete(cache.keys().next().value!);
}

function validateImageEmbeddings(value: unknown, grids: readonly ImageGrid[]): ImageEmbeddings {
  if (!value || typeof value !== 'object')
    fail('image_embeddings', `Kanana image embeddings must have shape [tokens, ${KANANA_EMBEDDING_SIZE}].`);
  const encoded = value as ImageEmbeddings;
  if (
    !Array.isArray(encoded.shape) ||
    encoded.shape.length !== 2 ||
    !Number.isSafeInteger(encoded.shape[0]) ||
    encoded.shape[0] < 0 ||
    encoded.shape[1] !== KANANA_EMBEDDING_SIZE ||
    !encoded.values ||
    encoded.values.length !== encoded.shape[0] * KANANA_EMBEDDING_SIZE
  )
    fail('image_embeddings', `Kanana image embeddings must have shape [tokens, ${KANANA_EMBEDDING_SIZE}].`);

  const expectedTokens = grids.reduce((total, [temporal, height, width]) => total + temporal * height * width, 0);
  if (encoded.shape[0] !== expectedTokens)
    fail(
      'image_embeddings',
      `Kanana image preprocessing returned ${encoded.shape[0]} tokens; expected ${expectedTokens}.`
    );
  return encoded;
}

async function prepareRequestImages(
  prepared: ReturnType<typeof prepareKananaSystemOne>,
  {imageTokenThw, prepareImageEmbeddings}: SessionInference,
  signal?: AbortSignal
): Promise<PreparedImages> {
  const sources = normalizeImageSources(prepared);
  const grids: ImageGrid[] = [];
  if (sources.length && !imageTokenThw)
    fail('image_preprocessor', 'Kanana image requests require an imageTokenThw resolver.');
  for (let index = 0; index < sources.length; index++) {
    throwIfAborted(signal);
    grids.push(snapshotGrid(await imageTokenThw!(sources[index], index, signal), index));
    throwIfAborted(signal);
  }

  let embeddings: ImageEmbeddings = {shape: [0, KANANA_EMBEDDING_SIZE], values: new Float32Array(0)};
  if (sources.length && prepareImageEmbeddings) {
    const encoded = await prepareImageEmbeddings(sources, grids, signal);
    throwIfAborted(signal);
    embeddings = validateImageEmbeddings(encoded, grids);
  }
  return {sources, grids, embeddings};
}

async function embedKananaInput(
  inputIds: readonly number[],
  image: readonly unknown[],
  imageTokenThw: readonly ImageGrid[],
  imageEmbeddings: ImageEmbeddings,
  dependencies: SessionInference,
  hiddenDimension: HiddenDimension,
  signal?: AbortSignal
): Promise<Float32Array> {
  throwIfAborted(signal);
  if (!inputIds.length) fail('invalid_input', 'Kanana embedding input must contain at least one token.');
  const input = Int32Array.from(inputIds);
  const attentionMask = new Int32Array(input.length).fill(1);
  const positions = prepareKananaPositionInputs(input, imageTokenThw, KANANA_IMAGE_TOKEN_ID);
  const inferenceInput: {
    inputIds: Int32Array;
    attentionMask: Int32Array;
    image: readonly unknown[];
    imageTokenThw: readonly ImageGrid[];
    positionIds: Int32Array;
    imagePositions: Int32Array;
    imageEmbeddings: ImageEmbeddings;
    signal?: AbortSignal;
  } = {
    inputIds: input,
    attentionMask,
    image,
    imageTokenThw,
    positionIds: positions.positionIds,
    imagePositions: positions.imagePositions,
    imageEmbeddings: image.length ? imageEmbeddings : {shape: [0, KANANA_EMBEDDING_SIZE], values: new Float32Array(0)}
  };
  if (signal) inferenceInput.signal = signal;
  const output = await dependencies.infer(Object.freeze(inferenceInput));
  throwIfAborted(signal);
  validateInferenceOutput(output);
  const outputDimension = output.shape[2];
  if (typeof outputDimension !== 'number' || !Number.isSafeInteger(outputDimension) || outputDimension < 1)
    fail('inference_output', 'Kanana inference returned an invalid hidden dimension.');
  if (hiddenDimension.value === undefined) hiddenDimension.value = outputDimension;
  if (outputDimension !== hiddenDimension.value)
    fail('inference_output', 'Kanana inference hidden dimension changed between embeddings.');
  const pooled = poolKananaHiddenStates(
    output.values,
    attentionMask,
    output.shape as [number, number, number],
    dependencies.embeddingDimension ?? outputDimension
  );
  return pooled.values;
}

function collectCandidateDocuments(candidateGroups: CandidateGroups): CandidateDocumentReference[] {
  const documents: CandidateDocumentReference[] = [];
  for (const [questionIndex, group] of candidateGroups.entries()) {
    for (const [index, document] of group.candidateDocuments.entries()) {
      if (document.key !== group.candidateKeys[index])
        fail('candidate_order', `Kanana candidate order changed for question group ${questionIndex}.`);
      documents.push({questionIndex, key: candidateCacheKey(document.inputIds), document});
    }
  }
  return documents;
}

function collectCachedCandidateEmbeddings(
  documents: readonly CandidateDocumentReference[],
  candidateEmbeddings: Map<string, Float32Array>
): Map<string, Float32Array> {
  const cacheHits = new Map<string, Float32Array>();
  for (const {key} of documents) {
    if (!candidateEmbeddings.has(key) || cacheHits.has(key)) continue;
    cacheHits.set(key, candidateEmbeddings.get(key)!);
  }
  for (const [key, embedding] of cacheHits) rememberCandidateEmbedding(candidateEmbeddings, key, embedding);
  return cacheHits;
}

function collectCandidateCacheMisses(
  documents: readonly CandidateDocumentReference[],
  cacheHits: ReadonlyMap<string, Float32Array>
): Map<string, CandidateDocument> {
  const cacheMisses = new Map<string, CandidateDocument>();
  for (const {key, document} of documents) {
    if (!cacheHits.has(key) && !cacheMisses.has(key)) cacheMisses.set(key, document);
  }
  return cacheMisses;
}

async function embedCandidateCacheMisses(
  cacheMisses: ReadonlyMap<string, CandidateDocument>,
  dependencies: SessionInference,
  imageEmbeddings: ImageEmbeddings,
  hiddenDimension: HiddenDimension,
  signal?: AbortSignal
): Promise<{embeddings: Map<string, Float32Array>; tokenCount: number}> {
  const embeddings = new Map<string, Float32Array>();
  let tokenCount = 0;
  for (const [key, document] of cacheMisses) {
    throwIfAborted(signal);
    const embedding = await embedKananaInput(
      document.inputIds,
      [],
      [],
      imageEmbeddings,
      dependencies,
      hiddenDimension,
      signal
    );
    embeddings.set(key, embedding);
    tokenCount += document.tokenCount;
    if (!Number.isSafeInteger(tokenCount))
      fail('usage', 'Kanana candidate token usage exceeds the safe integer range.');
  }
  return {embeddings, tokenCount};
}

function calculateCandidateSimilarities(
  documents: readonly CandidateDocumentReference[],
  candidateGroups: CandidateGroups,
  queryVectors: readonly Float32Array[],
  freshEmbeddings: ReadonlyMap<string, Float32Array>,
  cacheHits: ReadonlyMap<string, Float32Array>
): number[][] {
  const similarities = candidateGroups.map(() => [] as number[]);
  for (const {questionIndex, key} of documents) {
    const candidateVector = freshEmbeddings.get(key) ?? cacheHits.get(key);
    if (!candidateVector) fail('candidate_cache', 'Kanana candidate embedding was not found after cache resolution.');
    similarities[questionIndex]!.push(dotKananaVectors(queryVectors[questionIndex]!, candidateVector));
  }
  return similarities;
}

function inputTruncation(
  maxLength: number,
  queryQuestionIds: readonly string[],
  candidateGroups: CandidateGroups
): InputTruncation | undefined {
  const candidateDocuments: CandidateTruncation[] = [];
  for (const {id, candidateDocuments: documents} of candidateGroups) {
    const keys = documents.filter(({truncated}) => truncated).map(({key}) => key);
    if (!keys.length) continue;
    if (typeof id !== 'string') fail('candidate_order', 'Kanana candidate question IDs must be strings.');
    candidateDocuments.push({questionId: id, keys});
  }
  if (!queryQuestionIds.length && !candidateDocuments.length) return undefined;
  return {truncated: true, maxLength, queryQuestionIds, candidateDocuments};
}

async function embedCandidateDocuments(
  candidateGroups: ReturnType<typeof prepareKananaCandidates>,
  queryVectors: readonly Float32Array[],
  candidateEmbeddings: Map<string, Float32Array>,
  dependencies: SessionInference,
  imageEmbeddings: ImageEmbeddings,
  hiddenDimension: HiddenDimension,
  signal?: AbortSignal
): Promise<{similarities: number[][]; tokenCount: number}> {
  if (candidateGroups.length !== queryVectors.length)
    fail('candidate_order', 'Kanana query and candidate groups must preserve question order.');
  const documents = collectCandidateDocuments(candidateGroups);
  const cacheHits = collectCachedCandidateEmbeddings(documents, candidateEmbeddings);
  const cacheMisses = collectCandidateCacheMisses(documents, cacheHits);
  const {embeddings: freshEmbeddings, tokenCount} = await embedCandidateCacheMisses(
    cacheMisses,
    dependencies,
    imageEmbeddings,
    hiddenDimension,
    signal
  );
  for (const [key, embedding] of freshEmbeddings) rememberCandidateEmbedding(candidateEmbeddings, key, embedding);
  const similarities = calculateCandidateSimilarities(
    documents,
    candidateGroups,
    queryVectors,
    freshEmbeddings,
    cacheHits
  );
  return {similarities, tokenCount};
}

async function runKananaRequest(
  prepared: ReturnType<typeof prepareKananaSystemOne>,
  controls: RunControls,
  dependencies: SessionInference,
  candidateEmbeddings: Map<string, Float32Array>
): Promise<Record<string, unknown>> {
  const signal = controls.signal;
  throwIfAborted(signal);
  const imageState = await prepareRequestImages(prepared, dependencies, signal);
  const candidates = prepareKananaCandidates(dependencies.tokenizer, prepared, {
    maxLength: controls.maxLength,
    truncateInputs: controls.truncateInputs
  });
  const hiddenDimension: HiddenDimension = {};
  const queryVectors: Float32Array[] = [];
  const truncatedQueryQuestionIds: string[] = [];
  let inputTokens = 0;
  for (let index = 0; index < prepared.questionEntries.length; index++) {
    throwIfAborted(signal);
    const entry = prepared.questionEntries[index]!;
    if (!candidates[index] || candidates[index]!.id !== entry.id)
      fail('candidate_order', 'Kanana candidates must preserve question order.');
    const query = truncateKananaQueryWithCandidateFallback(
      dependencies.tokenizer,
      prepared,
      entry.id,
      imageState.grids.map((grid) => [...grid]),
      {
        maxLength: controls.maxLength,
        truncateInputs: controls.truncateInputs,
        truncateQuestionCandidates: controls.truncateQuestionCandidates
      }
    );
    if (query.truncated) truncatedQueryQuestionIds.push(entry.id);
    queryVectors.push(
      await embedKananaInput(
        query.inputIds,
        imageState.sources,
        imageState.grids,
        imageState.embeddings,
        dependencies,
        hiddenDimension,
        signal
      )
    );
    inputTokens += query.tokenCount;
    if (!Number.isSafeInteger(inputTokens)) fail('usage', 'Kanana input token usage exceeds the safe integer range.');
  }
  const {similarities, tokenCount} = await embedCandidateDocuments(
    candidates,
    queryVectors,
    candidateEmbeddings,
    dependencies,
    imageState.embeddings,
    hiddenDimension,
    signal
  );
  inputTokens += tokenCount;
  if (!Number.isSafeInteger(inputTokens)) fail('usage', 'Kanana input token usage exceeds the safe integer range.');
  const logitRows = similarities.map((row) =>
    scaleKananaLogits(row, KANANA_DECISION_TEMPERATURE, controls.temperature)
  );
  throwIfAborted(signal);
  const result = finishKananaSystemOne(prepared, logitRows, KANANA_MODEL_ID, {
    billing_units: prepared.questionEntries.length,
    input_tokens: inputTokens,
    output_tokens: 0
  });
  const truncation = inputTruncation(controls.maxLength, truncatedQueryQuestionIds, candidates);
  return truncation ? {...result, inputTruncation: truncation} : result;
}

/** Create the serialized decision layer around an injected Kanana hidden-state inference function. */
export function createKananaDecisionSession({
  tokenizer,
  infer,
  imageTokenThw,
  prepareImageEmbeddings,
  embeddingDimension,
  dispose: disposeInference
}: KananaDecisionSessionOptions): KananaDecisionSession {
  if (!tokenizer || typeof tokenizer.encode !== 'function')
    throw new TypeError('A Kanana tokenizer with encode() is required.');
  if (typeof infer !== 'function') throw new TypeError('A Kanana hidden-state infer() function is required.');
  if (imageTokenThw !== undefined && typeof imageTokenThw !== 'function')
    throw new TypeError('imageTokenThw must be a function when provided.');
  if (prepareImageEmbeddings !== undefined && typeof prepareImageEmbeddings !== 'function')
    throw new TypeError('prepareImageEmbeddings must be a function when provided.');
  if (embeddingDimension !== undefined && (!Number.isSafeInteger(embeddingDimension) || embeddingDimension < 1))
    throw new RangeError('embeddingDimension must be a positive safe integer.');
  if (disposeInference !== undefined && typeof disposeInference !== 'function')
    throw new TypeError('dispose must be a function when provided.');

  let closed = false;
  let queue = Promise.resolve();
  let disposal: Promise<void> | undefined;
  const candidateEmbeddings = new Map<string, Float32Array>();

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
    const maxLength = options.maxLength ?? KANANA_MAX_LENGTH;
    if (!Number.isSafeInteger(maxLength) || maxLength < 1 || maxLength > KANANA_MODEL_MAX_LENGTH)
      throw new RangeError(
        `Kanana maxLength must be a positive safe integer no greater than ${KANANA_MODEL_MAX_LENGTH}.`
      );

    const prepared = prepareKananaSystemOne(request);
    const temperature = options.temperature === undefined ? prepared.temperature : options.temperature;
    scaleKananaLogits([], KANANA_DECISION_TEMPERATURE, temperature);
    const controls = {
      ...(options.signal ? {signal: options.signal} : {}),
      maxLength,
      truncateInputs: options.truncateInputs ?? true,
      truncateQuestionCandidates: options.truncateQuestionCandidates ?? true,
      temperature
    };

    const dependencies: SessionInference = {
      tokenizer,
      infer,
      ...(imageTokenThw === undefined ? {} : {imageTokenThw}),
      ...(prepareImageEmbeddings === undefined ? {} : {prepareImageEmbeddings}),
      ...(embeddingDimension === undefined ? {} : {embeddingDimension})
    };
    const task = queue.then(() => runKananaRequest(prepared, controls, dependencies, candidateEmbeddings));
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
          candidateEmbeddings.clear();
          await disposeInference?.();
        });
      }
      return disposal;
    }
  };
}
