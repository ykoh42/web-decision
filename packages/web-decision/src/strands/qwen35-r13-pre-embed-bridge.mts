import {
  QWEN35_GRAPH_MAX_BATCH,
  QWEN35_MAX_CHOICE_OPTIONS,
  QWEN35_MAX_LENGTH,
  QWEN35_TEMPERATURE_BY_KIND
} from './qwen35-protocol.mts';
import type {Qwen35GraphInputChunk, Qwen35GraphInputRecord} from './qwen35-graph-input.mts';
import {validateH2oSignal} from '../h2o/signal.mts';
import {isQwen35FiniteActivationArray, type Qwen35VisionActivationPrecision} from './qwen35-vision-runner.mts';

export const QWEN35_R13_RUNTIME_INPUT_NAMES = Object.freeze([
  'inputs_embeds',
  'attention_mask',
  'position_ids',
  'option_indices',
  'temperatures',
  'n_slots'
] as const);

export const QWEN35_R13_HIDDEN_SIZE = 2048;
const QWEN35_IMAGE_PAD_TOKEN_ID = 248056;

type GraphInputTensor<
  Name extends string,
  Dtype extends 'float32' | 'int32',
  Values extends Float32Array | Int32Array
> = Readonly<{
  name: Name;
  dtype: Dtype;
  shape: readonly number[];
  values: Values;
}>;

export type Qwen35R13GraphInputTuple = readonly [
  GraphInputTensor<'inputs_embeds', 'float32', Float32Array>,
  GraphInputTensor<'attention_mask', 'int32', Int32Array>,
  GraphInputTensor<'position_ids', 'int32', Int32Array>,
  GraphInputTensor<'option_indices', 'int32', Int32Array>,
  GraphInputTensor<'temperatures', 'float32', Float32Array>,
  GraphInputTensor<'n_slots', 'int32', Int32Array>
];

export interface Qwen35PreEmbedSources {
  /** Resolve the supplied non-image token IDs to source BF16 embedding rows widened exactly to float32. */
  readonly lookupTokenEmbeddings: (
    tokenIds: Int32Array,
    options?: Readonly<{signal?: AbortSignal | null | undefined}>
  ) => Float32Array | PromiseLike<Float32Array>;
  /** One source-order visual output per input image, in float32 storage. */
  readonly imageFeatures: readonly Float32Array[];
  /** Activation precision of the pinned vision graph that produced imageFeatures. */
  readonly visionActivationPrecision?: Qwen35VisionActivationPrecision;
}

function invalid(message: string): never {
  throw new TypeError('Invalid Qwen3.5 R13 input bridge: ' + message);
}

function safeProduct(...values: readonly number[]): number {
  const product = values.reduce((result, value) => result * value, 1);
  if (!Number.isSafeInteger(product) || product < 0) invalid('tensor element count exceeds the safe integer range.');
  return product;
}

function assertInt32Vector(value: unknown, length: number, label: string): asserts value is Int32Array {
  if (!(value instanceof Int32Array) || value.length !== length) invalid(`${label} does not match its declared shape.`);
}

function assertEmbeddingValues(values: Float32Array, label: string, precision: Qwen35VisionActivationPrecision): void {
  if (isQwen35FiniteActivationArray(values, precision)) return;
  invalid(
    precision === 'float32'
      ? `${label} must contain finite float32 values.`
      : `${label} must contain finite source BF16 values widened exactly to float32.`
  );
}

type ChunkShape = Readonly<{batch: number; prefix: number; suffix: number; sequence: number}>;

function assertRowTokens(chunk: Qwen35GraphInputChunk, shape: ChunkShape, row: number): void {
  const {prefix, suffix, sequence} = shape;
  const question = chunk.questions[row]!;
  const active = question.inputIds.length;
  if (active < 1 || active > suffix) invalid(`question row ${row} has an invalid suffix length.`);
  const rowOffset = row * suffix;
  if (
    chunk.suffixAttentionMask.subarray(rowOffset, rowOffset + active).some((value) => value !== 1) ||
    chunk.suffixAttentionMask.subarray(rowOffset + active, rowOffset + suffix).some((value) => value !== 0)
  )
    invalid(`suffix attention mask row ${row} does not mark exactly its active tokens.`);
  for (let column = 0; column < active; column += 1)
    if (chunk.suffixInputIds[rowOffset + column] !== question.inputIds[column])
      invalid(`suffix token IDs differ from question row ${row}.`);
  const fullMaskOffset = row * sequence;
  for (let column = 0; column < prefix; column += 1)
    if (chunk.fullAttentionMask[fullMaskOffset + column] !== 1)
      invalid(`full attention mask row ${row} must keep the shared prefix active.`);
  for (let column = 0; column < suffix; column += 1)
    if (chunk.fullAttentionMask[fullMaskOffset + prefix + column] !== chunk.suffixAttentionMask[rowOffset + column])
      invalid(`full attention mask row ${row} does not preserve suffix padding.`);
}

function assertRowOptions(question: Qwen35GraphInputChunk['questions'][number], row: number): void {
  const active = question.inputIds.length;
  if (
    question.optionIndices.length < 2 ||
    question.optionIndices.length > QWEN35_MAX_CHOICE_OPTIONS ||
    question.optionIndices.length !== question.options.length
  )
    invalid(`question row ${row} must contain 2 to ${QWEN35_MAX_CHOICE_OPTIONS} matching option indices.`);
  let previous = -1;
  for (const optionIndex of question.optionIndices) {
    if (!Number.isSafeInteger(optionIndex) || optionIndex < 0 || optionIndex >= active || optionIndex <= previous)
      invalid(`question row ${row} has an out-of-range or unordered option index.`);
    previous = optionIndex;
  }
  if (!Object.hasOwn(QWEN35_TEMPERATURE_BY_KIND, question.type)) invalid(`question row ${row} has an unknown type.`);
}

function readOptionWidth(chunk: Qwen35GraphInputChunk, shape: ChunkShape): number {
  let optionWidth = 0;
  for (let row = 0; row < shape.batch; row += 1) {
    const question = chunk.questions[row]!;
    assertRowTokens(chunk, shape, row);
    assertRowOptions(question, row);
    optionWidth = Math.max(optionWidth, question.optionIndices.length);
  }
  if (optionWidth < 2 || optionWidth > QWEN35_MAX_CHOICE_OPTIONS)
    invalid('option width is outside the graph ABI range.');
  return optionWidth;
}

function assertTokenFlags(chunk: Qwen35GraphInputChunk): void {
  for (const value of chunk.fullAttentionMask)
    if (value !== 0 && value !== 1) invalid('full attention mask must contain only zero and one.');
  for (const value of chunk.mmTokenTypeIds)
    if (value !== 0 && value !== 1) invalid('multimodal token flags must contain only zero and one.');
}

function assertChunk(input: Qwen35GraphInputRecord, chunk: Qwen35GraphInputChunk): ChunkShape & {optionWidth: number} {
  if (
    input?.format !== 'qwen35-private-graph-input-v1' ||
    !Array.isArray(input.chunks) ||
    !input.chunks.includes(chunk)
  )
    invalid('chunk must belong to its graph input record.');

  const batch = chunk.questions.length;
  const prefix = chunk.prefixInputIds.length;
  const [suffixBatch, suffix] = chunk.suffixInputShape;
  const sequence = prefix + suffix;
  if (
    batch < 1 ||
    batch > QWEN35_GRAPH_MAX_BATCH ||
    input.maxLength < 2 ||
    input.maxLength > QWEN35_MAX_LENGTH ||
    sequence < 2 ||
    sequence > input.maxLength ||
    suffixBatch !== batch ||
    chunk.prefixInputShape[0] !== 1 ||
    chunk.prefixInputShape[1] !== prefix ||
    chunk.suffixInputIds.length !== safeProduct(batch, suffix) ||
    chunk.suffixAttentionMask.length !== safeProduct(batch, suffix) ||
    chunk.suffixPositionIdsShape[0] !== 4 ||
    chunk.suffixPositionIdsShape[1] !== batch ||
    chunk.suffixPositionIdsShape[2] !== suffix ||
    chunk.suffixPositionIds.length !== safeProduct(4, batch, suffix) ||
    chunk.prefixPositionIdsShape[0] !== 4 ||
    chunk.prefixPositionIdsShape[1] !== 1 ||
    chunk.prefixPositionIdsShape[2] !== prefix ||
    chunk.prefixPositionIds.length !== safeProduct(4, prefix) ||
    chunk.mmTokenTypeIds.length !== prefix ||
    chunk.fullAttentionMaskShape[0] !== batch ||
    chunk.fullAttentionMaskShape[1] !== sequence ||
    chunk.fullAttentionMask.length !== safeProduct(batch, sequence)
  )
    invalid('chunk token, mask, or position tensors do not agree with their declared shapes.');

  assertInt32Vector(chunk.prefixInputIds, prefix, 'prefix_input_ids');
  assertInt32Vector(chunk.suffixInputIds, safeProduct(batch, suffix), 'suffix_input_ids');
  assertInt32Vector(chunk.suffixAttentionMask, safeProduct(batch, suffix), 'suffix_attention_mask');
  assertInt32Vector(chunk.prefixPositionIds, safeProduct(4, prefix), 'prefix_position_ids');
  assertInt32Vector(chunk.suffixPositionIds, safeProduct(4, batch, suffix), 'suffix_position_ids');
  assertInt32Vector(chunk.mmTokenTypeIds, prefix, 'mm_token_type_ids');
  assertInt32Vector(chunk.fullAttentionMask, safeProduct(batch, sequence), 'full_attention_mask');

  const shape = {batch, prefix, suffix, sequence};
  const optionWidth = readOptionWidth(chunk, shape);
  assertTokenFlags(chunk);
  return {...shape, optionWidth};
}

function assertImageFeature(
  input: Qwen35GraphInputRecord,
  imageFeatures: readonly Float32Array[],
  imageIndex: number,
  precision: Qwen35VisionActivationPrecision
): void {
  const geometry = input.visionGeometry[imageIndex]!;
  const feature = imageFeatures[imageIndex];
  const [frames, height, width] = geometry.gridThw;
  if (
    frames !== 1 ||
    !Number.isSafeInteger(height) ||
    !Number.isSafeInteger(width) ||
    !Number.isSafeInteger(geometry.featureRows) ||
    geometry.featureRows < 1 ||
    geometry.featureRows !== (height * width) / 4 ||
    input.imageGridThw[imageIndex * 3] !== frames ||
    input.imageGridThw[imageIndex * 3 + 1] !== height ||
    input.imageGridThw[imageIndex * 3 + 2] !== width ||
    input.imageInput?.imageGridThw[imageIndex]?.some((value, field) => value !== geometry.gridThw[field]) ||
    !feature ||
    feature.length !== safeProduct(geometry.featureRows, QWEN35_R13_HIDDEN_SIZE)
  )
    invalid(`visual feature tensor ${imageIndex} does not match its pinned [S,2048] geometry.`);
  assertEmbeddingValues(feature, `visual feature tensor ${imageIndex}`, precision);
}

function readImageRun(
  input: Qwen35GraphInputRecord,
  chunk: Qwen35GraphInputChunk,
  start: number,
  imageIndex: number
): number {
  const geometry = input.visionGeometry[imageIndex];
  let token = start;
  while (chunk.mmTokenTypeIds[token] === 1) {
    if (chunk.prefixInputIds[token] !== QWEN35_IMAGE_PAD_TOKEN_ID)
      invalid('an image feature slot is not aligned with the pinned image placeholder token.');
    if (!geometry) invalid('prompt has more image runs than image inputs.');
    token += 1;
    if (token - start > geometry.featureRows)
      invalid(`prompt image run ${imageIndex} is longer than its source feature tensor.`);
  }
  if (token - start !== geometry!.featureRows)
    invalid(`prompt image run ${imageIndex} differs from its source feature row count.`);
  return token;
}

function assertImageRuns(input: Qwen35GraphInputRecord, chunk: Qwen35GraphInputChunk): void {
  let imageIndex = 0;
  let token = 0;
  while (token < chunk.mmTokenTypeIds.length) {
    if (chunk.mmTokenTypeIds[token] === 1) {
      token = readImageRun(input, chunk, token, imageIndex++);
    } else {
      token += 1;
    }
  }
  if (imageIndex !== input.imageCount) invalid('prompt image runs do not cover every source image.');
}

function assertVisionFeatures(
  input: Qwen35GraphInputRecord,
  chunk: Qwen35GraphInputChunk,
  imageFeatures: readonly Float32Array[],
  precision: Qwen35VisionActivationPrecision
): void {
  if (!Array.isArray(imageFeatures) || imageFeatures.length !== input.imageCount)
    invalid('imageFeatures must contain exactly one source-order feature tensor per image.');
  if (
    input.imageCount === 0
      ? input.imageInput !== undefined || input.visionGeometry.length !== 0
      : !input.imageInput ||
        input.imageInput.imageGridThw.length !== input.imageCount ||
        input.visionGeometry.length !== input.imageCount
  )
    invalid('image records and visual geometry do not agree.');
  if (
    !Number.isSafeInteger(input.imageCount) ||
    input.imageCount < 0 ||
    input.imageGridThwShape[0] !== input.imageCount ||
    input.imageGridThwShape[1] !== 3 ||
    input.imageGridThw.length !== safeProduct(input.imageCount, 3)
  )
    invalid('image grid tensor does not match the image count.');

  for (let imageIndex = 0; imageIndex < input.imageCount; imageIndex += 1)
    assertImageFeature(input, imageFeatures, imageIndex, precision);
  assertImageRuns(input, chunk);
}

function scatterPrefixEmbeddings(
  chunk: Qwen35GraphInputChunk,
  input: Qwen35GraphInputRecord,
  imageFeatures: readonly Float32Array[],
  rowOffset: number,
  inputsEmbeds: Float32Array,
  textTokenIds: number[],
  textDestinations: number[]
): void {
  const hiddenSize = QWEN35_R13_HIDDEN_SIZE;
  let imageIndex = 0;
  let imageRow = 0;
  for (let token = 0; token < chunk.prefixInputIds.length; token += 1) {
    const destination = (rowOffset + token) * hiddenSize;
    if (chunk.mmTokenTypeIds[token] === 1) {
      const image = imageFeatures[imageIndex]!;
      inputsEmbeds.set(image.subarray(imageRow * hiddenSize, (imageRow + 1) * hiddenSize), destination);
      imageRow += 1;
      if (imageRow === input.visionGeometry[imageIndex]!.featureRows) {
        imageIndex += 1;
        imageRow = 0;
      }
    } else {
      textTokenIds.push(chunk.prefixInputIds[token]!);
      textDestinations.push(destination);
    }
  }
}

async function buildInputEmbeddings(
  chunk: Qwen35GraphInputChunk,
  input: Qwen35GraphInputRecord,
  sources: Qwen35PreEmbedSources,
  shape: ChunkShape,
  signal: AbortSignal | null | undefined
): Promise<Float32Array> {
  const {batch, prefix, suffix, sequence} = shape;
  const hiddenSize = QWEN35_R13_HIDDEN_SIZE;
  const embeddingCount = safeProduct(batch, sequence, hiddenSize);
  const inputsEmbeds = new Float32Array(embeddingCount);
  const textTokenIds: number[] = [];
  const textDestinations: number[] = [];
  for (let row = 0; row < batch; row += 1) {
    const rowOffset = row * sequence;
    scatterPrefixEmbeddings(
      chunk,
      input,
      sources.imageFeatures,
      rowOffset,
      inputsEmbeds,
      textTokenIds,
      textDestinations
    );
    const suffixOffset = row * suffix;
    for (let token = 0; token < suffix; token += 1) {
      if (chunk.suffixAttentionMask[suffixOffset + token] !== 1) continue;
      textTokenIds.push(chunk.suffixInputIds[suffixOffset + token]!);
      textDestinations.push((rowOffset + prefix + token) * hiddenSize);
    }
  }

  const lookupIds = Int32Array.from(textTokenIds);
  const tokenEmbeddings = await sources.lookupTokenEmbeddings(lookupIds, {signal: signal});
  signal?.throwIfAborted();
  if (
    !(tokenEmbeddings instanceof Float32Array) ||
    tokenEmbeddings.length !== safeProduct(lookupIds.length, hiddenSize)
  )
    invalid('token lookup must return one float32 [2048] row for each requested token ID.');
  assertEmbeddingValues(tokenEmbeddings, 'token lookup', 'bfloat16');
  for (let row = 0; row < textDestinations.length; row += 1) {
    const sourceOffset = row * hiddenSize;
    inputsEmbeds.set(tokenEmbeddings.subarray(sourceOffset, sourceOffset + hiddenSize), textDestinations[row]!);
  }

  return inputsEmbeds;
}

function combinePositionIds(chunk: Qwen35GraphInputChunk, shape: ChunkShape): Int32Array {
  const {batch, prefix, suffix, sequence} = shape;
  const positionIds = new Int32Array(safeProduct(4, batch, sequence));
  for (let axis = 0; axis < 4; axis += 1) {
    for (let row = 0; row < batch; row += 1) {
      const target = (axis * batch + row) * sequence;
      const prefixSource = axis * prefix;
      positionIds.set(chunk.prefixPositionIds.subarray(prefixSource, prefixSource + prefix), target);
      const suffixSource = (axis * batch + row) * suffix;
      positionIds.set(chunk.suffixPositionIds.subarray(suffixSource, suffixSource + suffix), target + prefix);
    }
  }

  return positionIds;
}

function buildQuestionArrays(chunk: Qwen35GraphInputChunk, shape: ChunkShape & {optionWidth: number}) {
  const {batch, prefix, optionWidth} = shape;
  const optionIndices = new Int32Array(safeProduct(batch, optionWidth)).fill(-1);
  const temperatures = new Float32Array(batch);
  const nSlots = new Int32Array(batch);
  for (let row = 0; row < batch; row += 1) {
    const question = chunk.questions[row]!;
    const optionOffset = row * optionWidth;
    for (let option = 0; option < question.optionIndices.length; option += 1)
      optionIndices[optionOffset + option] = prefix + question.optionIndices[option]!;
    temperatures[row] = QWEN35_TEMPERATURE_BY_KIND[question.type];
    nSlots[row] = question.optionIndices.length;
  }

  return {optionIndices, temperatures, nSlots};
}

/**
 * Compose package text/vision chunks into the pinned R13 six-input ABI.
 * Embedding lookup and the visual tower remain caller-owned; this bridge only
 * assembles their source values in row-major token and source image order.
 */
export async function prepareQwen35R13GraphInputTuple(
  chunk: Qwen35GraphInputChunk,
  input: Qwen35GraphInputRecord,
  sources: Qwen35PreEmbedSources,
  options: Readonly<{signal?: AbortSignal | null | undefined}> = {}
): Promise<Qwen35R13GraphInputTuple> {
  if (typeof sources?.lookupTokenEmbeddings !== 'function') invalid('a source token embedding lookup is required.');
  if (options.signal != null) validateH2oSignal(options.signal);
  options.signal?.throwIfAborted();
  const shape = assertChunk(input, chunk);
  const {batch, sequence, optionWidth} = shape;
  assertVisionFeatures(input, chunk, sources.imageFeatures, sources.visionActivationPrecision ?? 'bfloat16');

  const hiddenSize = QWEN35_R13_HIDDEN_SIZE;
  const inputsEmbeds = await buildInputEmbeddings(chunk, input, sources, shape, options.signal);
  const positionIds = combinePositionIds(chunk, shape);
  const {optionIndices, temperatures, nSlots} = buildQuestionArrays(chunk, shape);

  return Object.freeze([
    Object.freeze({
      name: 'inputs_embeds',
      dtype: 'float32',
      shape: Object.freeze([batch, sequence, hiddenSize]),
      values: inputsEmbeds
    }),
    Object.freeze({
      name: 'attention_mask',
      dtype: 'int32',
      shape: Object.freeze([batch, sequence]),
      values: new Int32Array(chunk.fullAttentionMask)
    }),
    Object.freeze({
      name: 'position_ids',
      dtype: 'int32',
      shape: Object.freeze([4, batch, sequence]),
      values: positionIds
    }),
    Object.freeze({
      name: 'option_indices',
      dtype: 'int32',
      shape: Object.freeze([batch, optionWidth]),
      values: optionIndices
    }),
    Object.freeze({name: 'temperatures', dtype: 'float32', shape: Object.freeze([batch]), values: temperatures}),
    Object.freeze({name: 'n_slots', dtype: 'int32', shape: Object.freeze([batch]), values: nSlots})
  ]) as Qwen35R13GraphInputTuple;
}
