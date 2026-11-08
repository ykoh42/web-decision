import {WebDecisionError} from '../kev.js';

const invalid = (message) => new WebDecisionError('INVALID_INPUT', message);
const inference = (message) => new WebDecisionError('INFERENCE_ERROR', message);

function softmax(values) {
  if (!values.length || values.some((value) => !Number.isFinite(value)))
    throw inference('Decima returned invalid scores.');
  const max = values.reduce((maximum, value) => Math.max(maximum, value), -Infinity);
  const exponents = values.map((value) => Math.exp(value - max));
  const total = exponents.reduce((sum, value) => sum + value, 0);
  return exponents.map((value) => value / total);
}

function sigmoid(value) {
  return value >= 0 ? 1 / (1 + Math.exp(-value)) : Math.exp(value) / (1 + Math.exp(value));
}

function softplus(value) {
  return Math.max(value, 0) + Math.log1p(Math.exp(-Math.abs(value)));
}

function ordinalProbabilities(scores, features, config) {
  const {ord_g_w: locationWeights, ord_g_b: locationBias, ord_gap_w: gapWeights, ord_gap_b: gapBias} = config.ordinal;
  const choiceProbabilities = softmax(scores);
  const expected = choiceProbabilities.reduce((sum, probability, index) => sum + probability * index, 0);
  const centered = expected - (scores.length - 1) / 2;
  const meanFeatures = locationWeights.map(
    (_, feature) =>
      Array.from({length: scores.length}, (_, index) => features[index * locationWeights.length + feature]).reduce(
        (sum, value) => sum + value,
        0
      ) / scores.length
  );
  const location =
    meanFeatures.reduce((sum, value, index) => sum + value * locationWeights[index], locationBias) + centered;
  const gaps = Array.from(
    {length: scores.length},
    (_, index) =>
      softplus(
        features
          .slice(index * gapWeights.length, (index + 1) * gapWeights.length)
          .reduce((sum, value, feature) => sum + value * gapWeights[feature], gapBias)
      ) + 1e-3
  );
  const gapTotal = gaps.reduce((sum, value) => sum + value, 0);
  const probabilities = [];
  let cumulative = 0;
  for (let index = 0; index < scores.length - 1; index++) {
    cumulative += gaps[index];
    probabilities.push(sigmoid(location - (cumulative - gapTotal / 2)));
  }
  const ordinal = [1, ...probabilities].map((value, index) => value - [...probabilities, 0][index]);
  const clipped = ordinal.map((value) => Math.max(value, 1e-7));
  return softmax(clipped.map((value) => Math.log(value)));
}

export function decodeDecimaScores(raw, config, kind) {
  const {scores, features} = raw;
  if (!Array.isArray(scores) || scores.length < 2 || scores.some((value) => !Number.isFinite(value)))
    throw inference('Decima returned invalid scores.');
  const scaled = scores.map((value) => value / config.temperature);
  if (kind === 'rank') return scaled.map(sigmoid);
  if (kind === 'score') {
    if (!Array.isArray(features) || features.length !== scores.length * config.ordinal.ord_gap_w.length)
      throw inference('Decima returned invalid ordinal features.');
    return ordinalProbabilities(scaled, features, config);
  }
  return softmax(scaled);
}

export function finishDecimaChoice(prepared, probabilities, logits) {
  if (
    !Array.isArray(probabilities) ||
    probabilities.length !== prepared.choices.length ||
    !Array.isArray(logits) ||
    logits.length !== prepared.choices.length
  )
    throw inference('Decima returned an unexpected number of choices.');
  const maximum = probabilities.reduce((current, value) => Math.max(current, value), -Infinity);
  const selectedIndex = probabilities.indexOf(maximum);
  return {
    selected: prepared.choices[selectedIndex],
    selectedIndex,
    probabilities: prepared.choices.map((option, index) => ({
      option,
      probability: probabilities[index],
      logit: logits[index]
    }))
  };
}

export function validateDecimaRequest(request) {
  if (!request || typeof request !== 'object' || Array.isArray(request))
    throw invalid('A decision request is required.');
  if (typeof request.state !== 'string') throw invalid('state must be a string.');
  if (!Array.isArray(request.questions) || request.questions.length < 1)
    throw invalid('Decima requires at least one question.');
  return request;
}
