export const KANANA_DECISION_TEMPERATURE = 0.056139365295102696;
export const KANANA_DEFAULT_TEMPERATURE = 1.0;

type HiddenStateShape = readonly [batchSize: number, sequenceLength: number, hiddenDimension: number];
type PooledVectorShape = readonly [batchSize: number, embeddingDimension: number];
type PooledVectors = {values: Float32Array; shape: PooledVectorShape};

function isPositiveDimension(value: number): boolean {
  return Number.isSafeInteger(value) && value > 0;
}

function attended(value: unknown): boolean {
  if (value === 1 || value === 1n || value === true) return true;
  if (value === 0 || value === 0n || value === false) return false;
  throw new TypeError('attentionMask values must be boolean or 0/1.');
}

/** Pool each row at the pinned last-attended token and return L2-normalized FP32 vectors. */
export function poolKananaHiddenStates(
  hiddenStates: ArrayLike<number>,
  attentionMask: ArrayLike<unknown>,
  shape: HiddenStateShape,
  embeddingDimension: number
): PooledVectors {
  if (!Array.isArray(shape) || shape.length !== 3 || !shape.every(isPositiveDimension))
    throw new RangeError('hidden-state shape must contain positive batch, sequence, and hidden dimensions.');
  const [batchSize, sequenceLength, hiddenDimension] = shape;
  if (!isPositiveDimension(embeddingDimension) || embeddingDimension > hiddenDimension)
    throw new RangeError('embeddingDimension must be positive and no larger than hiddenDimension.');
  const hiddenLength = batchSize * sequenceLength * hiddenDimension;
  const maskLength = batchSize * sequenceLength;
  if (!Number.isSafeInteger(hiddenLength) || hiddenStates.length !== hiddenLength)
    throw new RangeError('hidden-state data length does not match its shape.');
  if (!Number.isSafeInteger(maskLength) || attentionMask.length !== maskLength)
    throw new RangeError('attentionMask length does not match the batch and sequence dimensions.');

  const finalPositions = new Int32Array(batchSize);
  for (let batch = 0; batch < batchSize; batch++) {
    const maskOffset = batch * sequenceLength;
    let finalPosition = -1;
    for (let sequence = 0; sequence < sequenceLength; sequence++) {
      if (attended(attentionMask[maskOffset + sequence])) finalPosition = sequence;
    }
    if (finalPosition === -1) throw new RangeError('every prompt must contain an attended token.');
    finalPositions[batch] = finalPosition;
  }

  const vectors = new Float32Array(batchSize * embeddingDimension);
  const epsilon = Math.fround(1e-12);
  for (let batch = 0; batch < batchSize; batch++) {
    const position = finalPositions[batch]!;
    const hiddenOffset = (batch * sequenceLength + position) * hiddenDimension;
    let normSquared = 0;
    for (let dimension = 0; dimension < embeddingDimension; dimension++) {
      const value = Math.fround(hiddenStates[hiddenOffset + dimension]!);
      normSquared = Math.fround(normSquared + Math.fround(value * value));
    }
    const norm = Math.fround(Math.sqrt(normSquared));
    const denominator = Math.max(norm, epsilon);
    const outputOffset = batch * embeddingDimension;
    for (let dimension = 0; dimension < embeddingDimension; dimension++) {
      const value = Math.fround(hiddenStates[hiddenOffset + dimension]!);
      vectors[outputOffset + dimension] = Math.fround(value / denominator);
    }
  }

  return {values: vectors, shape: [batchSize, embeddingDimension]};
}

function isNumberArray(values: unknown): values is ArrayLike<unknown> & Iterable<unknown> {
  return Array.isArray(values) || (ArrayBuffer.isView(values) && !(values instanceof DataView));
}

/** Dot two equally sized FP32 vectors with the pinned model's output precision. */
export function dotKananaVectors(left: unknown, right: unknown): number {
  if (!isNumberArray(left) || !isNumberArray(right)) throw new TypeError('Kanana vectors must be arrays of numbers.');
  if (left.length === 0 || left.length !== right.length)
    throw new RangeError('Kanana vectors must have the same nonzero length.');

  let dot = 0;
  for (let index = 0; index < left.length; index++) {
    const leftValue = left[index];
    const rightValue = right[index];
    if (typeof leftValue !== 'number' || !Number.isFinite(leftValue))
      throw new TypeError('Kanana vectors must contain finite numbers.');
    if (typeof rightValue !== 'number' || !Number.isFinite(rightValue))
      throw new TypeError('Kanana vectors must contain finite numbers.');
    const product = Math.fround(Math.fround(leftValue) * Math.fround(rightValue));
    dot = Math.fround(dot + product);
  }
  return dot;
}

export function scaleKananaLogits(
  cosineSimilarities: unknown,
  decisionTemperature: number = KANANA_DECISION_TEMPERATURE,
  temperature: number = KANANA_DEFAULT_TEMPERATURE
): number[] {
  if (!isNumberArray(cosineSimilarities)) throw new TypeError('cosineSimilarities must be an array of numbers.');
  if (typeof decisionTemperature !== 'number' || !Number.isFinite(decisionTemperature) || decisionTemperature <= 0)
    throw new RangeError('decisionTemperature must be a finite number greater than 0.');
  if (typeof temperature !== 'number' || !Number.isFinite(temperature) || temperature <= 0 || temperature > 100)
    throw new RangeError('temperature must be in (0, 100].');

  // The upstream model scales FP32 cosine tensors before converting logits to Python floats.
  const scale = Math.fround(1 / decisionTemperature);
  const requestTemperature = Math.fround(temperature);
  return Array.from(cosineSimilarities, (cosine) => {
    if (typeof cosine !== 'number' || !Number.isFinite(cosine))
      throw new TypeError('cosineSimilarities must contain finite numbers.');
    const similarity = Math.fround(cosine);
    const scaled = Math.fround(scale * similarity);
    return Math.fround(scaled / requestTemperature);
  });
}
