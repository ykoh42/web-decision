import {WebDecisionError} from '../kev.js';
import {SKEJJA_FEWSHOT_SIGMA} from './contract.mjs';

const invalid = (message: string): WebDecisionError => new WebDecisionError('INVALID_INPUT', message);

function logistic(value: number): number {
  if (value >= 0) return 1 / (1 + Math.exp(-value));
  const exponent = Math.exp(value);
  return exponent / (1 + exponent);
}

function objective(
  logits: number[],
  labels: number[],
  coefficients: [number, number],
  prior: number[],
  penalty: number
): number {
  let loss = 0;
  for (let index = 0; index < logits.length; index++) {
    const value = coefficients[0] * logits[index]! + coefficients[1];
    loss += Math.max(0, value) + Math.log1p(Math.exp(-Math.abs(value))) - labels[index]! * value;
  }
  return loss + (penalty / 2) * (coefficients[0] - prior[0]!) ** 2 + (penalty / 2) * (coefficients[1] - prior[1]!) ** 2;
}

function gradientAndHessian(
  logits: number[],
  labels: number[],
  coefficients: [number, number],
  prior: number[],
  penalty: number
): {gradientA: number; gradientB: number; hessianAA: number; hessianAB: number; hessianBB: number} {
  let gradientA = penalty * (coefficients[0] - prior[0]!);
  let gradientB = penalty * (coefficients[1] - prior[1]!);
  let hessianAA = penalty;
  let hessianAB = 0;
  let hessianBB = penalty;

  for (let index = 0; index < logits.length; index++) {
    const logit = logits[index]!;
    const value = Math.max(-40, Math.min(40, coefficients[0] * logit + coefficients[1]));
    const probability = logistic(value);
    const weight = probability * (1 - probability);
    const residual = probability - labels[index]!;
    gradientA += logit * residual;
    gradientB += residual;
    hessianAA += weight * logit * logit;
    hessianAB += weight * logit;
    hessianBB += weight;
  }

  return {gradientA, gradientB, hessianAA, hessianAB, hessianBB};
}

export function fitSkejjaPlatt(
  logits: number[],
  labels: number[],
  prior: number[],
  sigma: number = SKEJJA_FEWSHOT_SIGMA
): {a: number; b: number} {
  if (
    !Array.isArray(logits) ||
    !Array.isArray(labels) ||
    logits.length < 1 ||
    logits.length !== labels.length ||
    logits.some((value) => !Number.isFinite(value)) ||
    labels.some((value) => value !== 0 && value !== 1) ||
    !Array.isArray(prior) ||
    prior.length !== 2 ||
    prior.some((value) => !Number.isFinite(value)) ||
    !Number.isFinite(sigma) ||
    sigma <= 0
  )
    throw invalid('Skejja Platt calibration inputs are invalid.');

  const lambda = 1 / (sigma * sigma);
  const coefficients = [...prior] as [number, number];
  let loss = objective(logits, labels, coefficients, prior, lambda);

  for (let iteration = 0; iteration < 50; iteration++) {
    const {gradientA, gradientB, hessianAA, hessianAB, hessianBB} = gradientAndHessian(
      logits,
      labels,
      coefficients,
      prior,
      lambda
    );
    const determinant = hessianAA * hessianBB - hessianAB * hessianAB;
    if (!Number.isFinite(determinant) || determinant <= 0) break;
    const stepA = (hessianBB * gradientA - hessianAB * gradientB) / determinant;
    const stepB = (hessianAA * gradientB - hessianAB * gradientA) / determinant;
    let scale = 1;
    let next!: [number, number];
    let nextLoss!: number;

    while (scale > 1e-6) {
      next = [coefficients[0] - scale * stepA, coefficients[1] - scale * stepB];
      nextLoss = objective(logits, labels, next, prior, lambda);
      if (nextLoss <= loss) break;
      scale /= 2;
    }
    if (scale <= 1e-6) break;

    const done = Math.max(Math.abs(next[0] - coefficients[0]), Math.abs(next[1] - coefficients[1])) < 1e-8;
    coefficients[0] = next[0];
    coefficients[1] = next[1];
    loss = nextLoss;
    if (done) break;
  }

  return {a: coefficients[0], b: coefficients[1]};
}
