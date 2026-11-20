import {
  packQwen35Request,
  QWEN35_MAX_IMAGES,
  type Qwen35PackedRequest,
  type Qwen35RequestOptions
} from './qwen35-protocol.mts';
import type {Qwen35ImagesInput, Qwen35ImageInputOptions} from './qwen35-image-input.mts';
import {preprocessQwen35Images} from './qwen35-image-input.mts';
import {
  prepareQwen35TextVisionInputFromPacked,
  type Qwen35TextVisionChunk,
  type Qwen35TextVisionConfig,
  type Qwen35TextVisionInput
} from './qwen35-text-vision-input.mts';

const PINNED_CONFIG: Qwen35TextVisionConfig = Object.freeze({
  imageTokenId: 248056,
  visionEndTokenId: 248054,
  padTokenId: 248044,
  numPositionEmbeddings: 2304,
  spatialMergeSize: 2
});

const PINNED_SOURCE = Object.freeze({
  strandsRevision: '13c05149265dc404c09d2469636f71b1320e1713',
  transformersVersion: '5.22.2',
  baseRepository: 'Qwen/Qwen3.5-2B-Base',
  baseRevision: 'b1485b2fa6dfa1287294f269f5fb618e03d52d7c',
  baseConfigSha256: 'ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4',
  checkpointRepository: 'StrandsAgents/strands-decider-2B-qwen3.5-v1-2610',
  checkpointRevision: '216d1ccb46a9e9b036eb9187b248db98480f77d4'
});

export type Qwen35GraphInputChunk = Qwen35TextVisionChunk &
  Readonly<{
    prefixAttentionMask: Int32Array;
    prefixAttentionMaskShape: readonly [batch: 1, sequence: number];
    fullAttentionMask: Int32Array;
    fullAttentionMaskShape: readonly [batch: number, sequence: number];
  }>;

export type Qwen35GraphInputRecord = Readonly<{
  format: 'qwen35-private-graph-input-v1';
  model: string;
  maxLength: number;
  request: Qwen35TextVisionInput['request'];
  pinnedSource: typeof PINNED_SOURCE;
  imageCount: number;
  imageInput: Qwen35ImagesInput | undefined;
  imageGridThw: Int32Array;
  imageGridThwShape: readonly [images: number, fields: 3];
  visionGeometry: Qwen35TextVisionInput['visionGeometry'];
  chunks: readonly Qwen35GraphInputChunk[];
  inputTokenCount: number;
}>;

export type Qwen35GraphInputOptions = Qwen35ImageInputOptions &
  Qwen35RequestOptions &
  Pick<Qwen35TextVisionConfig, 'strictWindow'>;

function withAttentionMasks(chunk: Qwen35TextVisionChunk): Qwen35GraphInputChunk {
  const prefixLength = chunk.prefixInputIds.length;
  const [batch, suffixWidth] = chunk.suffixInputShape;
  const fullWidth = prefixLength + suffixWidth;
  const prefixAttentionMask = new Int32Array(prefixLength).fill(1);
  const fullAttentionMask = new Int32Array(batch * fullWidth);
  for (let row = 0; row < batch; row += 1) {
    const fullOffset = row * fullWidth;
    const suffixOffset = row * suffixWidth;
    fullAttentionMask.fill(1, fullOffset, fullOffset + prefixLength);
    fullAttentionMask.set(
      chunk.suffixAttentionMask.subarray(suffixOffset, suffixOffset + suffixWidth),
      fullOffset + prefixLength
    );
  }
  return Object.freeze({
    ...chunk,
    prefixAttentionMask,
    prefixAttentionMaskShape: [1, prefixLength] as const,
    fullAttentionMask,
    fullAttentionMaskShape: [batch, fullWidth] as const
  });
}

function assertContextFit(input: Qwen35TextVisionInput): void {
  for (const chunk of input.chunks)
    for (const question of chunk.questions)
      if (chunk.prefixInputIds.length + question.inputIds.length > input.maxLength)
        throw new RangeError(`Qwen input exceeds the configured ${input.maxLength}-token context.`);
}

export async function prepareQwen35GraphInput(
  tokenizer: unknown,
  request: unknown,
  options: Qwen35GraphInputOptions = {}
): Promise<Qwen35GraphInputRecord> {
  const packed = packQwen35Request(tokenizer, request, {
    maxImages: options.maxImages ?? QWEN35_MAX_IMAGES,
    ...(options.maxBatch === undefined ? {} : {maxBatch: options.maxBatch}),
    ...(options.maxLength === undefined ? {} : {maxLength: options.maxLength}),
    ...(options.maxQuestionFraction === undefined ? {} : {maxQuestionFraction: options.maxQuestionFraction}),
    ...(options.maxBatchTokens === undefined ? {} : {maxBatchTokens: options.maxBatchTokens})
  });
  return prepareQwen35GraphInputFromPacked(tokenizer, packed, options);
}

export async function prepareQwen35GraphInputFromPacked(
  tokenizer: unknown,
  packed: Qwen35PackedRequest,
  options: Qwen35GraphInputOptions = {}
): Promise<Qwen35GraphInputRecord> {
  const maxImages = options.maxImages ?? QWEN35_MAX_IMAGES;
  const imageCount = packed.images.length;
  const imageInput = imageCount ? await preprocessQwen35Images(packed.images, options) : undefined;
  const textVision = prepareQwen35TextVisionInputFromPacked(tokenizer, packed, imageInput, {
    ...PINNED_CONFIG,
    maxImages,
    ...(options.maxBatch === undefined ? {} : {maxBatch: options.maxBatch}),
    ...(options.maxLength === undefined ? {} : {maxLength: options.maxLength}),
    ...(options.maxQuestionFraction === undefined ? {} : {maxQuestionFraction: options.maxQuestionFraction}),
    ...(options.maxBatchTokens === undefined ? {} : {maxBatchTokens: options.maxBatchTokens}),
    ...(options.strictWindow === undefined ? {} : {strictWindow: options.strictWindow})
  });
  assertContextFit(textVision);
  const chunks = textVision.chunks.map(withAttentionMasks);

  return Object.freeze({
    format: 'qwen35-private-graph-input-v1',
    model: textVision.request.model,
    maxLength: textVision.maxLength,
    request: textVision.request,
    pinnedSource: PINNED_SOURCE,
    imageCount,
    imageInput,
    imageGridThw: textVision.imageGridThw,
    imageGridThwShape: textVision.imageGridThwShape,
    visionGeometry: textVision.visionGeometry,
    chunks: Object.freeze(chunks),
    inputTokenCount: textVision.inputTokenCount
  });
}
