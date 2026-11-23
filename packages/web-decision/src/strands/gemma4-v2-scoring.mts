import {WebDecisionError} from '../kev.js';
import {qwen35OrderedRecord} from './qwen35-protocol.mjs';
import {gemma4V2PortableExp} from './gemma4-v2-portable-exp.mjs';

type Question = Readonly<{
  kind: 'choice' | 'score' | 'noul';
  options: readonly string[];
  descriptions: readonly string[];
}>;
type Config = Readonly<{
  temperature: number;
  softcap: number;
  ordinalSmoothing?: number;
  kernel?: Gemma4V2ScoringKernel;
}>;
export type Gemma4V2ScoringKernel = (logits: ArrayLike<number>, count: number) => number[];
export const GEMMA4_V2_SCORING_BYTES = 746272;
export const GEMMA4_V2_SCORING_SHA256 = '8976812e52bc591dee1a1b891d427b16eb0bf641ea61299a518276d35e6da3ad';
const invalid = (message: string) => new WebDecisionError('INFERENCE_ERROR', message);

function roundBfloat16(value: number): number {
  const number = new Float32Array(1);
  const bits = new Uint32Array(number.buffer);
  number[0] = value;
  if (!Number.isFinite(number[0])) throw invalid('The model returned a non-finite logit.');
  bits[0] = (bits[0]! + 0x7fff + ((bits[0]! >>> 16) & 1)) & 0xffff0000;
  return number[0]!;
}

// Match Python's round of the binary float, including ties to even.
function roundPython(value: number): number {
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, Math.abs(value));
  const bits = view.getBigUint64(0);
  const exponentBits = Number((bits >> 52n) & 0x7ffn);
  const fraction = bits & ((1n << 52n) - 1n);
  const mantissa = exponentBits === 0 ? fraction : fraction | (1n << 52n);
  const exponent = exponentBits === 0 ? -1074 : exponentBits - 1023 - 52;
  const scaled = mantissa * 10000n;
  let rounded: bigint;
  if (exponent >= 0) rounded = scaled << BigInt(exponent);
  else {
    const divisor = 1n << BigInt(-exponent);
    const quotient = scaled / divisor;
    const twice = (scaled % divisor) * 2n;
    rounded = twice > divisor || (twice === divisor && quotient % 2n === 1n) ? quotient + 1n : quotient;
  }
  const result = Number(rounded) / 10000;
  return value < 0 ? -result : result;
}

// This table records the pinned checkpoint and macOS NumPy reference, not every NumPy build.
const SCORE_BYTES = 65536 * 4;
// NumPy's float32 sum: eight lanes through 128 values, then recursive aligned blocks.
function pairwiseSum(values: readonly number[], start: number, length: number): number {
  const add = (left: number, right: number) => Math.fround(left + right);
  if (length < 8) {
    let sum = -0;
    for (let index = 0; index < length; index++) sum = add(sum, values[start + index]!);
    return sum;
  }
  if (length <= 128) {
    const lanes = values.slice(start, start + 8);
    let index = 8;
    for (; index < length - (length % 8); index += 8)
      for (let lane = 0; lane < 8; lane++) lanes[lane] = add(lanes[lane]!, values[start + index + lane]!);
    let sum = add(
      add(add(lanes[0]!, lanes[1]!), add(lanes[2]!, lanes[3]!)),
      add(add(lanes[4]!, lanes[5]!), add(lanes[6]!, lanes[7]!))
    );
    for (; index < length; index++) sum = add(sum, values[start + index]!);
    return sum;
  }
  const half = Math.floor(length / 16) * 8;
  return add(pairwiseSum(values, start, half), pairwiseSum(values, start + half, length - half));
}

/** Build the fixed-checkpoint scorer from the verified little-endian model asset. */
export function createGemma4V2ScoringKernel(asset: ArrayBuffer): Gemma4V2ScoringKernel {
  if (asset.byteLength !== GEMMA4_V2_SCORING_BYTES) throw invalid('Invalid Gemma 4 v2 scoring asset size.');
  const table = new DataView(asset.slice(0));
  const correctionCount = (asset.byteLength - SCORE_BYTES) / 8;
  const scratch = new DataView(new ArrayBuffer(4));
  const score = (value: number) => {
    scratch.setFloat32(0, value, true);
    const bits = scratch.getUint32(0, true);
    if (!Number.isFinite(scratch.getFloat32(0, true))) throw invalid('The model returned a non-finite logit.');
    const rounded = (bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16;
    return table.getFloat32(rounded * 4, true);
  };
  const exp = (value: number) => {
    scratch.setFloat32(0, value, true);
    const bits = scratch.getUint32(0, true);
    let low = 0,
      high = correctionCount;
    while (low < high) {
      const middle = Math.floor((low + high) / 2);
      if (table.getUint32(SCORE_BYTES + middle * 8, true) < bits) low = middle + 1;
      else high = middle;
    }
    if (low < correctionCount && table.getUint32(SCORE_BYTES + low * 8, true) === bits)
      return table.getFloat32(SCORE_BYTES + low * 8 + 4, true);
    return gemma4V2PortableExp(value);
  };
  return (logits, count) => {
    const scores = Array.from({length: count}, (_, index) => score(logits[index]!));
    const max = Math.max(...scores);
    const exps = scores.map((value) => exp(Math.fround(value - max)));
    const total = pairwiseSum(exps, 0, exps.length);
    return exps.map((value) => Math.fround(value / total));
  };
}

export function gemma4V2Probabilities(logits: ArrayLike<number>, count: number, config: Config): number[] {
  if (
    !Number.isInteger(count) ||
    count < 2 ||
    count > 255 ||
    logits.length < count ||
    logits.length > 255 ||
    !Number.isFinite(config.temperature) ||
    config.temperature <= 0 ||
    !Number.isFinite(config.softcap) ||
    config.softcap <= 0
  )
    throw invalid('Invalid Gemma 4 v2 readout.');
  if (config.kernel) {
    if (config.temperature !== 1.5456 || config.softcap !== 30)
      throw invalid('Scoring asset does not match the decision configuration.');
    return config.kernel(logits, count);
  }
  const f32 = Math.fround;
  const temperature = f32(config.temperature);
  const cap = f32(config.softcap);
  const scores: number[] = [];
  for (let index = 0; index < count; index++) {
    const logit = logits[index]!;
    if (!Number.isFinite(logit)) throw invalid('The model returned a non-finite logit.');
    const bounded = f32(f32(Math.tanh(f32(roundBfloat16(logit) / cap))) * cap);
    scores.push(f32(bounded / temperature));
  }
  const maximum = Math.max(...scores);
  const exps = scores.map((score) => f32(Math.exp(f32(score - maximum))));
  const total = exps.reduce((sum, value) => f32(sum + value), 0);
  return exps.map((value) => f32(value / total));
}

function scoreConfidence(probabilities: readonly number[], smoothing: number): number {
  const count = probabilities.length;
  const total = probabilities.reduce((sum, value) => sum + value, 0);
  const normalized = probabilities.map((value) => value / total);
  const mean = normalized.reduce((sum, value, index) => sum + index * value, 0);
  const variance = normalized.reduce((sum, value, index) => sum + value * (index - mean) ** 2, 0);
  const sigma = Math.sqrt(variance);
  const max = (count - 1) / 2;
  const floor = Math.sqrt(smoothing);
  if (max <= floor) return sigma <= floor ? 1 : 0;
  return Math.max(0, Math.min(1, (max - sigma) / (max - floor)));
}

/** Apply the pinned v2 readout and System One response rules. */
export function finishGemma4V2Answer(question: Question, logits: ArrayLike<number>, config: Config) {
  const count = question.options.length;
  if (
    !['choice', 'score', 'noul'].includes(question.kind) ||
    question.descriptions.length !== count ||
    (question.kind === 'score' && count > 10) ||
    (question.kind === 'noul' && (count !== 2 || question.options[0] !== 'false' || question.options[1] !== 'true'))
  )
    throw invalid('Invalid Gemma 4 v2 question.');
  const smoothing = config.ordinalSmoothing ?? 0;
  if (!Number.isFinite(smoothing) || smoothing < 0) throw invalid('Invalid ordinal smoothing.');
  const probabilities = gemma4V2Probabilities(logits, count, config);
  if (question.kind === 'noul') return {type: 'noul' as const, noul: roundPython(probabilities[1]!)};
  const named = qwen35OrderedRecord(
    question.options.map((option, index) => [option, roundPython(probabilities[index]!)] as const)
  );
  if (question.kind === 'choice') {
    const maximum = Math.max(...probabilities);
    const choice = question.options[probabilities.indexOf(maximum)]!;
    const confidence = Math.max(0, Math.min(1, (count * maximum - 1) / (count - 1)));
    return {type: 'choice' as const, choice, probabilities: named, confidence: roundPython(confidence)};
  }
  const score = probabilities.reduce((sum, value, index) => sum + index * value, 0);
  const legend = qwen35OrderedRecord(
    question.options.map((option, index) => [option, question.descriptions[index]!] as const)
  );
  return {
    type: 'score' as const,
    score: roundPython(score),
    legend,
    probabilities: named,
    confidence: roundPython(scoreConfidence(probabilities, smoothing))
  };
}
