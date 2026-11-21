import {WebDecisionError} from '../kev.js';
import {MAPIKA2B_MAX_OPTIONS, type Mapika2bPlan, type Mapika2bSlot} from './contract.mts';

interface ConfidenceFields {
  readonly confidence: number;
  readonly x_p_max: number;
  readonly certainty: number;
  readonly probabilities: Readonly<Record<string, number>>;
}

export type Mapika2bAnswer =
  | ({readonly type: 'choice'; readonly choice: string} & ConfidenceFields)
  | {readonly type: 'noul'; readonly noul: number}
  | ({
      readonly type: 'score';
      readonly score: number;
      readonly legend: Readonly<Record<string, string>>;
      readonly level_fit?: Readonly<Record<string, number>>;
      readonly fit_mass?: number;
    } & ConfidenceFields);

export interface Mapika2bResult {
  readonly model: string;
  readonly answers: Readonly<Record<string, Mapika2bAnswer>>;
  readonly usage: Readonly<{input_tokens: number; output_tokens: 0}>;
}

function invalid(message: string): never {
  throw new WebDecisionError('INFERENCE_ERROR', message);
}

// Python round converts the exact floating-point value with ties to even.
function rounded(value: number, places = 4): number {
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, Math.abs(value));
  const bits = view.getBigUint64(0);
  const exponentBits = Number((bits >> 52n) & 0x7ffn);
  const fraction = bits & ((1n << 52n) - 1n);
  const mantissa = exponentBits ? fraction | (1n << 52n) : fraction;
  const exponent = exponentBits ? exponentBits - 1075 : -1074;
  const scaled = mantissa * 10n ** BigInt(places);
  if (exponent >= 0) return ((value < 0 ? -1 : 1) * Number(scaled << BigInt(exponent))) / 10 ** places;
  const divisor = 1n << BigInt(-exponent);
  const quotient = scaled / divisor;
  const remainder = (scaled % divisor) * 2n;
  const result = remainder > divisor || (remainder === divisor && quotient % 2n === 1n) ? quotient + 1n : quotient;
  return ((value < 0 ? -1 : 1) * Number(result)) / 10 ** places;
}

function sum(values: readonly number[]): number {
  return values.reduce((accumulator, value) => accumulator + value, 0);
}

function normalised(probabilities: readonly number[], zeroIsUniform: boolean): number[] {
  const total = sum(probabilities);
  return total
    ? probabilities.map((value) => value / total)
    : probabilities.map(() => (zeroIsUniform ? 1 / probabilities.length : 0));
}

function clip(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function certainty(probabilities: readonly number[]): number {
  const entropy = -probabilities.reduce((total, value) => total + (value > 0 ? value * Math.log(value) : 0), 0);
  return Math.max(0, 1 - entropy / Math.log(probabilities.length));
}

function choiceConfidence(probabilities: readonly number[]): number {
  const values = normalised(probabilities, true);
  const n = values.length;
  return clip((n * Math.max(...values) - 1) / (n - 1));
}

function scoreConfidence(probabilities: readonly number[]): number {
  const values = normalised(probabilities, true);
  const mode = values.indexOf(Math.max(...values));
  const spread = sum(values.map((value, index) => value * Math.abs(index - mode)));
  const centre = (values.length - 1) / 2;
  const uniform = sum(values.map((_, index) => Math.abs(index - centre))) / values.length;
  return clip(1 - spread / uniform);
}

function orderedRecord<T>(entries: readonly (readonly [string, T])[]): Readonly<Record<string, T>> {
  const record = Object.fromEntries(entries);
  return Object.freeze(new Proxy(record, {ownKeys: () => entries.map(([key]) => key)}));
}

function namedValues(keys: readonly string[], values: readonly number[]): Readonly<Record<string, number>> {
  return orderedRecord(keys.map((key, index) => [key, rounded(values[index]!)] as const));
}

function softmax(logits: Float32Array, slot: Mapika2bSlot, index: number): number[] {
  const start = index * MAPIKA2B_MAX_OPTIONS;
  const temperature = Math.fround(slot.temperature);
  const scaled = Array.from(logits.subarray(start, start + slot.optionCount), (value) =>
    Math.fround(value / temperature)
  );
  if (scaled.length < 2 || scaled.some((value) => !Number.isFinite(value)))
    invalid('Mapika graph returned invalid option logits.');
  const maximum = Math.max(...scaled);
  const weights = scaled.map((value) => Math.fround(Math.exp(Math.fround(value - maximum))));
  const total = weights.reduce((accumulator, value) => Math.fround(accumulator + value), 0);
  return weights.map((weight) => Math.fround(weight / total));
}

function readSlotProbabilities(plan: Mapika2bPlan, logitsByRow: readonly Float32Array[]): number[][] {
  if (logitsByRow.length !== plan.rows.length) invalid('Mapika graph returned the wrong number of scoring rows.');
  return plan.rows.flatMap(({slots}, row) => {
    const logits = logitsByRow[row];
    if (!(logits instanceof Float32Array) || logits.length !== slots.length * MAPIKA2B_MAX_OPTIONS)
      invalid('Mapika graph must return 255 logits per answer slot.');
    return slots.map((slot, index) => softmax(logits, slot, index));
  });
}

function formatChoice(names: readonly string[], row: readonly number[]): Mapika2bAnswer {
  const values = normalised(row, false);
  const max = Math.max(...values);
  return {
    type: 'choice',
    choice: names[values.indexOf(max)]!,
    confidence: rounded(choiceConfidence(values)),
    x_p_max: rounded(max),
    certainty: rounded(certainty(values)),
    probabilities: namedValues(names, values)
  };
}

function formatScore(legend: readonly string[], row: readonly number[], fits?: readonly number[]): Mapika2bAnswer {
  const values = normalised(row, false);
  const keys = legend.map((_, index) => String(index));
  return {
    type: 'score',
    score: rounded(sum(values.map((value, index) => index * value)), 2),
    confidence: rounded(scoreConfidence(values)),
    x_p_max: rounded(Math.max(...values)),
    certainty: rounded(certainty(values)),
    legend: Object.fromEntries(keys.map((key, index) => [key, legend[index]!])),
    probabilities: namedValues(keys, values),
    ...(fits === undefined ? {} : {level_fit: namedValues(keys, fits), fit_mass: rounded(sum(fits) || 1e-9)})
  };
}

/** Decode a row-major [slotsInRow, 255] tensor for each prepared forward row. */
export function finishMapika2bResult(
  plan: Mapika2bPlan,
  logitsByRow: readonly Float32Array[],
  model = 'decider-2b-v11'
): Mapika2bResult {
  if (plan.images.length > 0 && plan.imagePrefixLength <= 0)
    invalid('Mapika image rows must have vision tokens before inference.');
  const probabilities = readSlotProbabilities(plan, logitsByRow);
  const answers = plan.questions.map(({firstSlot, ...question}): readonly [string, Mapika2bAnswer] => {
    const row = probabilities[firstSlot];
    if (!row) return invalid('Mapika graph did not score a question.');
    if (question.kind === 'noul') return [question.name, {type: 'noul', noul: rounded(normalised(row, false)[1]!)}];
    if (question.kind === 'choice') return [question.name, formatChoice(question.names, row)];
    if (!question.isolated) return [question.name, formatScore(question.legend, row)];
    const fits = probabilities.slice(firstSlot, firstSlot + question.legend.length).map((values) => values[1]!);
    return [question.name, formatScore(question.legend, normalised(fits, false), fits)];
  });
  return {model, answers: orderedRecord(answers), usage: {input_tokens: plan.inputTokenCount, output_tokens: 0}};
}
