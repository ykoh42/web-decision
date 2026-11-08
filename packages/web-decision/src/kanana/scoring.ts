export const KANANA_DECISION_TEMPERATURE = 0.056139365295102696;
export const KANANA_DEFAULT_TEMPERATURE = 1.0;

function isNumberArray(values: unknown): values is ArrayLike<unknown> & Iterable<unknown> {
  return Array.isArray(values) || (ArrayBuffer.isView(values) && !(values instanceof DataView));
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
