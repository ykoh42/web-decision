import {
  packQwen35Request,
  QWEN35_GRAPH_MAX_BATCH,
  QWEN35_MAX_CHOICE_OPTIONS,
  renderQwen35Question,
  resolveQwen35BatchOptions,
  resolveQwen35MaxLength,
  splitQwen35BatchRanges
} from './qwen35-protocol.mts';
import type {
  Qwen35PackedRow,
  Qwen35RequestOptions,
  Qwen35RequestSnapshot,
  Qwen35RenderedQuestion
} from './qwen35-protocol.mts';
import type {Qwen35ImagesInput} from './qwen35-image-input.mts';
import {buildQwen35PerImageVisionGeometry, type Qwen35PerImageVisionGeometry} from './qwen35-vision-geometry.mts';

const IMAGE_PAD = '<|image_pad|>';
const VISION_START = '<|vision_start|>';
const VISION_END = '<|vision_end|>';
const PAD_TOKEN = '<|endoftext|>';
const STATE_OPEN = '<state>\n';
const STATE_CLOSE = '\n</state>\n';
const IMAGE_PATCH_WIDTH = 1536;
const INT32_MAX = 2_147_483_647;
const utf8 = new TextEncoder();

interface Qwen35Encoded {
  readonly ids?: unknown;
  readonly tokens?: unknown;
}

interface Qwen35Tokenizer {
  readonly decoder?:
    | {
        readonly byte_decoder?: Readonly<Record<string, string | number>> | null | undefined;
      }
    | null
    | undefined;
  get_added_tokens_decoder?: (() => {has(value: number): boolean} | null | undefined) | undefined;
  encode?: ((text: string, options: {readonly add_special_tokens: boolean}) => unknown) | undefined;
}

type TokenSpan = readonly [start: number, end: number];

export type Qwen35TextVisionConfig = Qwen35RequestOptions &
  Readonly<{
    imageTokenId: number;
    visionEndTokenId: number;
    padTokenId: number;
    numPositionEmbeddings: number;
    spatialMergeSize: number;
    strictWindow?: boolean;
  }>;

export type Qwen35TextVisionQuestion = Readonly<{
  name: string;
  type: Qwen35PackedRow['type'];
  options: Qwen35PackedRow['options'];
  questionText: string;
  inputIds: Int32Array;
  optionIndices: Int32Array;
  answerIndex: number;
}>;

export type Qwen35TextVisionChunk = Readonly<{
  expandedStateText: string;
  prefixInputIds: Int32Array;
  prefixInputShape: readonly [batch: 1, sequence: number];
  mmTokenTypeIds: Int32Array;
  prefixPositionIds: Int32Array;
  prefixPositionIdsShape: readonly [axes: 4, batch: 1, sequence: number];
  ropeDelta: number;
  suffixInputIds: Int32Array;
  suffixInputShape: readonly [batch: number, sequence: number];
  suffixAttentionMask: Int32Array;
  suffixPositionIds: Int32Array;
  suffixPositionIdsShape: readonly [axes: 4, batch: number, sequence: number];
  questions: readonly Qwen35TextVisionQuestion[];
  /** Token volume for this graph dispatch, including its repeated prefix. */
  inputTokenCount: number;
}>;

export type Qwen35TextVisionInput = Readonly<{
  maxLength: number;
  request: Qwen35RequestSnapshot;
  imageInput: Qwen35ImagesInput | undefined;
  imageGridThw: Int32Array;
  imageGridThwShape: readonly [images: number, fields: 3];
  visionGeometry: readonly Qwen35PerImageVisionGeometry[];
  chunks: readonly Qwen35TextVisionChunk[];
  /** Strands usage accounting: count the prefix once per max_batch/max_batch_tokens group. */
  inputTokenCount: number;
}>;

function invalid(message: string): never {
  throw new TypeError(message);
}

function int32(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0 || value > INT32_MAX) invalid(`${label} must be an int32 token ID.`);
  return value;
}

function encoded(tokenizer: unknown, text: string, addSpecialTokens: boolean): {ids: number[]; tokens: string[]} {
  const runtime = tokenizer as Qwen35Tokenizer | null | undefined;
  if (typeof runtime?.encode !== 'function') invalid('Qwen tokenizer is unavailable.');
  const output = runtime.encode(text, {add_special_tokens: addSpecialTokens}) as Qwen35Encoded | null | undefined;
  if (!Array.isArray(output?.ids) || !Array.isArray(output.tokens) || output.ids.length !== output.tokens.length)
    invalid('Qwen tokenizer must return matching indexed IDs and token strings.');
  const ids: number[] = [];
  const tokens: string[] = [];
  for (let index = 0; index < output.ids.length; index += 1) {
    const id = output.ids[index];
    const token = output.tokens[index];
    if (!Number.isSafeInteger(id) || (id as number) < 0 || typeof token !== 'string')
      invalid('Qwen tokenizer returned invalid token data.');
    ids.push(int32(id as number, 'Qwen token'));
    tokens.push(token);
  }
  return {ids, tokens};
}

function tokenByteSpans(
  tokenizer: unknown,
  text: string,
  tokens: readonly string[],
  ids: readonly number[]
): TokenSpan[] {
  const runtime = tokenizer as Qwen35Tokenizer | null | undefined;
  const decoder = runtime?.decoder?.byte_decoder;
  const added = runtime?.get_added_tokens_decoder?.();
  if (!decoder || !added) invalid('Expected a ByteLevel tokenizer with an added-token table.');
  const expected = utf8.encode(text.normalize('NFC'));
  const spans: TokenSpan[] = [];
  let cursor = 0;
  for (let index = 0; index < ids.length; index += 1) {
    const token = tokens[index]!;
    const bytes = added.has(ids[index]!)
      ? utf8.encode(token)
      : Uint8Array.from(
          Array.from(token, (character) => {
            const byte = decoder[character];
            if (byte === undefined) invalid('Tokenizer returned a non-ByteLevel token.');
            return Number(byte);
          })
        );
    const start = cursor;
    for (const byte of bytes) {
      if (cursor >= expected.length || expected[cursor] !== byte)
        invalid('Tokenizer bytes do not match NFC prompt text.');
      cursor += 1;
    }
    spans.push([start, cursor]);
  }
  if (cursor !== expected.length) invalid('Tokenizer did not cover the full prompt text.');
  return spans;
}

function questionOptionIndices(rendered: Qwen35RenderedQuestion, spans: readonly TokenSpan[], cut: number): Int32Array {
  return new Int32Array(
    rendered.optionSpans.map(([start, end]) => {
      const prefix = rendered.questionText.slice(0, start).normalize('NFC');
      const option = rendered.questionText.slice(start, end).normalize('NFC');
      const suffix = rendered.questionText.slice(end).normalize('NFC');
      if (prefix + option + suffix !== rendered.questionText.normalize('NFC'))
        invalid('Unicode normalization crossed an option boundary.');
      const byteStart = utf8.encode(prefix).length;
      const byteEnd = byteStart + utf8.encode(option).length;
      let last = -1;
      for (let index = cut; index < spans.length; index += 1) {
        const [left, right] = spans[index]!;
        if (right > left && left >= byteStart && right <= byteEnd) last = index - cut;
      }
      if (last < 0) invalid('An option has no surviving token for the pointer head.');
      return last;
    })
  );
}

function renderImageState(stateText: string, imageCount: number): string {
  if (imageCount === 0) return stateText;
  if (!stateText.startsWith(STATE_OPEN) || !stateText.endsWith(STATE_CLOSE))
    invalid('Qwen image prompts require a rendered <state> block.');
  const body = stateText.slice(STATE_OPEN.length, -STATE_CLOSE.length);
  const placeholders = Array.from({length: imageCount}, () => `${VISION_START}${IMAGE_PAD}${VISION_END}`).join('\n');
  return `${STATE_OPEN}${body ? `${placeholders}\n${body}` : placeholders}${STATE_CLOSE}`;
}

function expandImageTokens(stateText: string, counts: readonly number[]): string {
  const parts = stateText.split(IMAGE_PAD);
  if (parts.length - 1 !== counts.length)
    invalid(`Prompt has ${parts.length - 1} image placeholders for ${counts.length} images.`);
  let result = parts[0]!;
  for (let index = 0; index < counts.length; index += 1) result += IMAGE_PAD.repeat(counts[index]!) + parts[index + 1]!;
  return result;
}

function imageTokenCounts(imageInput: Qwen35ImagesInput, mergeSize: number): number[] {
  return imageInput.imageGridThw.map(([frames, height, width]) => {
    if (![frames, height, width].every(Number.isSafeInteger) || frames !== 1 || height < 1 || width < 1)
      throw new RangeError('Qwen3.5 image grids must contain positive integer [1,H,W] rows.');
    const patches = frames * height * width;
    const divisor = mergeSize * mergeSize;
    if (height % mergeSize !== 0 || width % mergeSize !== 0 || patches % divisor !== 0)
      throw new RangeError(`Qwen3.5 image grid rows must align to spatialMergeSize=${mergeSize}.`);
    return patches / divisor;
  });
}

function writeImageRopePositions(
  positions: Int32Array,
  tokenStart: number,
  tokenCount: number,
  position: number,
  grid: readonly [number, number, number] | undefined,
  mergeSize: number
): number {
  if (!grid) invalid('Qwen image token run has no matching image grid.');
  const [frames, height, width] = grid;
  const expectedTokens = (frames * height * width) / (mergeSize * mergeSize);
  if (tokenCount !== expectedTokens)
    invalid(`Qwen image token run has ${tokenCount} tokens; its grid requires ${expectedTokens}.`);
  const stride = positions.length / 3;
  const mergedHeight = height / mergeSize;
  const mergedWidth = width / mergeSize;
  let index = tokenStart;
  for (let frame = 0; frame < frames; frame += 1) {
    for (let row = 0; row < mergedHeight; row += 1) {
      for (let column = 0; column < mergedWidth; column += 1) {
        positions[index] = position + frame;
        positions[stride + index] = position + row;
        positions[2 * stride + index] = position + column;
        index += 1;
      }
    }
  }
  return Math.floor(Math.max(height, width) / mergeSize);
}

function writeTextRopePositions(positions: Int32Array, start: number, end: number, position: number): void {
  const stride = positions.length / 3;
  for (let index = start; index < end; index += 1) {
    const value = position + (index - start);
    positions[index] = value;
    positions[stride + index] = value;
    positions[2 * stride + index] = value;
  }
}

function ropePositions(
  inputIds: ArrayLike<number>,
  mmTokenTypeIds: ArrayLike<number>,
  grids: readonly (readonly [number, number, number])[],
  mergeSize: number
): {positions: Int32Array; delta: number} {
  const sequenceLength = inputIds.length;
  const positions = new Int32Array(3 * sequenceLength);
  let currentPosition = 0;
  let imageIndex = 0;
  let tokenIndex = 0;

  while (tokenIndex < sequenceLength) {
    const modality = mmTokenTypeIds[tokenIndex]!;
    let end = tokenIndex + 1;
    while (end < sequenceLength && mmTokenTypeIds[end] === modality) end += 1;
    const runLength = end - tokenIndex;
    if (modality === 0) {
      writeTextRopePositions(positions, tokenIndex, end, currentPosition);
      currentPosition += runLength;
    } else if (modality === 1) {
      currentPosition += writeImageRopePositions(
        positions,
        tokenIndex,
        runLength,
        currentPosition,
        grids[imageIndex++],
        mergeSize
      );
    } else {
      invalid(`Unsupported Qwen multimodal token type ${modality}.`);
    }
    tokenIndex = end;
  }

  if (imageIndex !== grids.length) invalid(`Qwen prompt has ${imageIndex} image runs for ${grids.length} grids.`);
  let maximum = -1;
  for (const position of positions) maximum = Math.max(maximum, position);
  return {positions, delta: maximum + 1 - sequenceLength};
}

function prefixPositions(mrope: Int32Array, sequenceLength: number): Int32Array {
  const result = new Int32Array(4 * sequenceLength);
  for (let index = 0; index < sequenceLength; index += 1) result[index] = index;
  result.set(mrope.subarray(0, sequenceLength), sequenceLength);
  result.set(mrope.subarray(sequenceLength, 2 * sequenceLength), 2 * sequenceLength);
  result.set(mrope.subarray(2 * sequenceLength), 3 * sequenceLength);
  return result;
}

function suffixInputs(
  questions: readonly Qwen35TextVisionQuestion[],
  prefixLength: number,
  ropeDelta: number,
  padTokenId: number
): {
  inputIds: Int32Array;
  attentionMask: Int32Array;
  positionIds: Int32Array;
  shape: readonly [number, number];
  positionShape: readonly [4, number, number];
} {
  const width = questions.reduce((maximum, question) => Math.max(maximum, question.inputIds.length), 0);
  const batch = questions.length;
  const inputIds = new Int32Array(batch * width).fill(padTokenId);
  const attentionMask = new Int32Array(batch * width);
  const positionIds = new Int32Array(4 * batch * width);
  for (let row = 0; row < batch; row += 1) {
    const question = questions[row]!;
    inputIds.set(question.inputIds, row * width);
    attentionMask.fill(1, row * width, row * width + question.inputIds.length);
    for (let column = 0; column < width; column += 1) {
      const index = row * width + column;
      positionIds[index] = prefixLength + column;
      const multimodalPosition = prefixLength + column + ropeDelta;
      positionIds[batch * width + index] = multimodalPosition;
      positionIds[2 * batch * width + index] = multimodalPosition;
      positionIds[3 * batch * width + index] = multimodalPosition;
    }
  }
  return {
    inputIds,
    attentionMask,
    positionIds,
    shape: [batch, width],
    positionShape: [4, batch, width]
  };
}

function validateImages(imageCount: number, imageInput: Qwen35ImagesInput | undefined): void {
  if (imageCount === 0) {
    if (imageInput) invalid('A text-only Qwen request cannot include preprocessed images.');
    return;
  }
  if (!imageInput) invalid('Preprocess every Qwen image before building the text and vision inputs.');
  if (imageInput.imageGridThw.length !== imageCount)
    invalid(`Qwen request has ${imageCount} images but preprocessing returned ${imageInput.imageGridThw.length}.`);
  const patchRows = imageInput.imageGridThw.reduce((sum, [frames, height, width]) => sum + frames * height * width, 0);
  if (
    imageInput.pixelShape[0] !== patchRows ||
    imageInput.pixelShape[1] !== IMAGE_PATCH_WIDTH ||
    imageInput.pixelValues.length !== patchRows * IMAGE_PATCH_WIDTH
  )
    invalid('Qwen preprocessed pixel rows do not match image_grid_thw.');
}

function configuredSpecialToken(tokenizer: unknown, text: string, expected: number, label: string): void {
  const ids = encoded(tokenizer, text, false).ids;
  if (ids.length !== 1 || ids[0] !== expected) invalid(`Qwen ${label} ID does not match the pinned tokenizer.`);
}

export function prepareQwen35TextVisionInput(
  tokenizer: unknown,
  request: unknown,
  imageInput: Qwen35ImagesInput | undefined,
  config: Qwen35TextVisionConfig
): Qwen35TextVisionInput {
  const requestOptions = {
    ...(config.maxImages === undefined ? {} : {maxImages: config.maxImages}),
    ...(config.maxBatch === undefined ? {} : {maxBatch: config.maxBatch}),
    ...(config.maxLength === undefined ? {} : {maxLength: config.maxLength}),
    ...(config.maxQuestionFraction === undefined ? {} : {maxQuestionFraction: config.maxQuestionFraction}),
    ...(config.maxBatchTokens === undefined ? {} : {maxBatchTokens: config.maxBatchTokens})
  };
  const packed = packQwen35Request(tokenizer, request, requestOptions);
  return prepareQwen35TextVisionInputFromPacked(tokenizer, packed, imageInput, config);
}

type PackedRequest = ReturnType<typeof packQwen35Request>;
type PrefixInputs = Pick<
  Qwen35TextVisionChunk,
  | 'expandedStateText'
  | 'prefixInputIds'
  | 'prefixInputShape'
  | 'mmTokenTypeIds'
  | 'prefixPositionIds'
  | 'prefixPositionIdsShape'
  | 'ropeDelta'
>;

function validateTextVisionConfig(tokenizer: unknown, config: Qwen35TextVisionConfig): void {
  if (!Number.isSafeInteger(config.spatialMergeSize) || config.spatialMergeSize !== 2)
    throw new RangeError('Qwen3.5 image input requires spatialMergeSize=2.');
  int32(config.imageTokenId, 'Qwen image token');
  int32(config.visionEndTokenId, 'Qwen vision-end token');
  int32(config.padTokenId, 'Qwen pad token');
  configuredSpecialToken(tokenizer, IMAGE_PAD, config.imageTokenId, 'image token');
  configuredSpecialToken(tokenizer, VISION_END, config.visionEndTokenId, 'vision-end token');
  configuredSpecialToken(tokenizer, PAD_TOKEN, config.padTokenId, 'pad token');
}

function fitStateTokens(
  stateTokens: number[],
  longestQuestion: number,
  maxLength: number,
  maxQuestionFraction: number,
  strictWindow: boolean
) {
  if (strictWindow) {
    if (stateTokens.length + longestQuestion > maxLength)
      throw new RangeError(
        `Prompt of ${stateTokens.length + longestQuestion} tokens exceeds the context window of ${maxLength} tokens.`
      );
    return {reserve: longestQuestion, stateBudget: maxLength, prefixIds: stateTokens};
  }
  const reserve = Math.min(longestQuestion, Math.max(1, Math.trunc(maxLength * maxQuestionFraction)));
  const stateBudget = maxLength - reserve;
  return {reserve, stateBudget, prefixIds: stateTokens.slice(0, stateBudget)};
}

function checkImageState(
  stateTokens: number[],
  visionEndTokenId: number,
  stateBudget: number,
  strictWindow: boolean
): void {
  const lastImageToken = stateTokens.lastIndexOf(visionEndTokenId);
  if (lastImageToken < 0) invalid('Expanded Qwen image prompt is missing <|vision_end|>.');
  const keep = lastImageToken + 1;
  if (!strictWindow && keep > stateBudget)
    throw new RangeError(`Qwen images take ${keep} tokens of a ${stateBudget}-token state budget.`);
}

function buildPrefixInputs(
  promptText: string,
  prefixIds: number[],
  grids: readonly (readonly [number, number, number])[],
  config: Qwen35TextVisionConfig
): PrefixInputs {
  const prefixInputIds = new Int32Array(prefixIds);
  const prefixLength = prefixInputIds.length;
  const mmTokenTypeIds = grids.length
    ? prefixInputIds.map((id) => Number(id === config.imageTokenId))
    : new Int32Array(prefixLength);
  const rope = ropePositions(prefixInputIds, mmTokenTypeIds, grids, config.spatialMergeSize);
  return {
    expandedStateText: promptText,
    prefixInputIds,
    prefixInputShape: [1, prefixLength],
    mmTokenTypeIds,
    prefixPositionIds: prefixPositions(rope.positions, prefixLength),
    prefixPositionIdsShape: [4, 1, prefixLength],
    ropeDelta: rope.delta
  };
}

function prepareQuestions(
  tokenizer: unknown,
  packed: PackedRequest,
  encodedQuestions: readonly ReturnType<typeof encoded>[],
  start: number,
  reserve: number,
  strictWindow: boolean
): Qwen35TextVisionQuestion[] {
  return encodedQuestions.map((question, offset) => {
    const index = start + offset;
    const row = packed.rows[index]!;
    const cut = strictWindow ? 0 : Math.max(0, question.ids.length - reserve);
    const inputIds = new Int32Array(question.ids.slice(cut));
    const rendered = renderQwen35Question(packed.questions[index]!);
    const spans = tokenByteSpans(tokenizer, row.questionText, question.tokens, question.ids);
    return Object.freeze({
      name: row.name,
      type: row.type,
      options: row.options,
      questionText: row.questionText,
      inputIds,
      optionIndices: questionOptionIndices(rendered, spans, cut),
      answerIndex: inputIds.length - 1
    });
  });
}

function countSuffixTokens(questions: readonly Qwen35TextVisionQuestion[]): number {
  return questions.reduce((sum, question) => sum + question.inputIds.length, 0);
}

function createGraphChunk(
  prefix: PrefixInputs,
  questionBatch: Qwen35TextVisionQuestion[],
  maxLength: number,
  padTokenId: number
): Qwen35TextVisionChunk {
  const prefixLength = prefix.prefixInputIds.length;
  const suffix = suffixInputs(questionBatch, prefixLength, prefix.ropeDelta, padTokenId);
  const combinedSequenceWidth = prefixLength + suffix.shape[1];
  if (questionBatch.length < 1 || questionBatch.length > QWEN35_GRAPH_MAX_BATCH)
    throw new RangeError(`Qwen graph batches must contain 1 to ${QWEN35_GRAPH_MAX_BATCH} rows.`);
  if (combinedSequenceWidth < 2 || combinedSequenceWidth > maxLength)
    throw new RangeError(`Qwen graph context must contain 2 to ${maxLength} tokens.`);
  for (const question of questionBatch)
    if (question.optionIndices.length < 2 || question.optionIndices.length > QWEN35_MAX_CHOICE_OPTIONS)
      throw new RangeError(`Qwen graph questions must contain 2 to ${QWEN35_MAX_CHOICE_OPTIONS} option indices.`);
  return Object.freeze({
    ...prefix,
    suffixInputIds: suffix.inputIds,
    suffixInputShape: suffix.shape,
    suffixAttentionMask: suffix.attentionMask,
    suffixPositionIds: suffix.positionIds,
    suffixPositionIdsShape: suffix.positionShape,
    questions: Object.freeze(questionBatch),
    inputTokenCount: prefixLength + countSuffixTokens(questionBatch)
  });
}

function splitQuestionDispatches(
  prefix: PrefixInputs,
  questions: readonly Qwen35TextVisionQuestion[],
  maxBatchTokens: number | null,
  maxLength: number,
  padTokenId: number
) {
  const prefixLength = prefix.prefixInputIds.length;
  const ranges = splitQwen35BatchRanges(
    questions.map((question) => question.inputIds.length),
    prefixLength,
    maxBatchTokens
  );
  const chunks: Qwen35TextVisionChunk[] = [];
  let sourceTokenCount = 0;
  for (const [start, end] of ranges) {
    sourceTokenCount += prefixLength + countSuffixTokens(questions.slice(start, end));
    for (let graphStart = start; graphStart < end; graphStart += QWEN35_GRAPH_MAX_BATCH) {
      const graphEnd = Math.min(graphStart + QWEN35_GRAPH_MAX_BATCH, end);
      chunks.push(createGraphChunk(prefix, questions.slice(graphStart, graphEnd), maxLength, padTokenId));
    }
  }
  return {chunks, sourceTokenCount};
}

export function prepareQwen35TextVisionInputFromPacked(
  tokenizer: unknown,
  packed: PackedRequest,
  imageInput: Qwen35ImagesInput | undefined,
  config: Qwen35TextVisionConfig
): Qwen35TextVisionInput {
  const imageCount = packed.images.length;
  const maxLength = resolveQwen35MaxLength(config.maxLength ?? packed.maxLength);
  validateImages(imageCount, imageInput);
  validateTextVisionConfig(tokenizer, config);

  const grids = imageInput?.imageGridThw ?? [];
  const geometry = grids.length
    ? buildQwen35PerImageVisionGeometry(grids, {
        numPositionEmbeddings: config.numPositionEmbeddings,
        spatialMergeSize: config.spatialMergeSize,
        ...(config.maxImages === undefined ? {} : {maxImages: config.maxImages})
      })
    : [];
  const counts = imageInput ? imageTokenCounts(imageInput, config.spatialMergeSize) : [];
  const gridInput = new Int32Array(grids.length * 3);
  grids.forEach((grid, index) => gridInput.set(grid, index * 3));

  const rawQuestions = packed.rows.map((row) => encoded(tokenizer, row.questionText, false));
  const batchOptions = resolveQwen35BatchOptions(config);
  const strictWindow = config.strictWindow ?? false;
  const maxBatchTokens = imageCount ? null : batchOptions.maxBatchTokens;
  const chunks: Qwen35TextVisionChunk[] = [];
  let totalTokenCount = 0;
  for (let start = 0; start < packed.rows.length; start += batchOptions.maxBatch) {
    const group = rawQuestions.slice(start, start + batchOptions.maxBatch);
    const longestQuestion = group.reduce((longest, question) => Math.max(longest, question.ids.length), 0);
    const promptText = imageCount
      ? expandImageTokens(renderImageState(packed.stateText, imageCount), counts)
      : packed.stateText;
    const stateTokens = encoded(tokenizer, promptText, true).ids;
    const fit = fitStateTokens(stateTokens, longestQuestion, maxLength, batchOptions.maxQuestionFraction, strictWindow);
    if (imageCount) checkImageState(stateTokens, config.visionEndTokenId, fit.stateBudget, strictWindow);
    const prefix = buildPrefixInputs(promptText, fit.prefixIds, grids, config);
    const questions = prepareQuestions(tokenizer, packed, group, start, fit.reserve, strictWindow);
    const dispatches = splitQuestionDispatches(prefix, questions, maxBatchTokens, maxLength, config.padTokenId);
    for (const chunk of dispatches.chunks) chunks.push(chunk);
    totalTokenCount += dispatches.sourceTokenCount;
  }

  const requestSnapshot: Qwen35RequestSnapshot = {
    model: packed.model,
    state: packed.state,
    stateText: packed.stateText,
    questions: packed.questions,
    images: packed.images
  };
  return Object.freeze({
    maxLength,
    request: Object.freeze(requestSnapshot),
    imageInput,
    imageGridThw: gridInput,
    imageGridThwShape: [grids.length, 3] as const,
    visionGeometry: Object.freeze(geometry),
    chunks: Object.freeze(chunks),
    inputTokenCount: totalTokenCount
  });
}
