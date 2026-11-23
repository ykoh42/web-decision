const EXP_COEFFICIENTS = [
  1 / 87178291200,
  1 / 6227020800,
  1 / 479001600,
  1 / 39916800,
  1 / 3628800,
  1 / 362880,
  1 / 40320,
  1 / 5040,
  1 / 720,
  1 / 120,
  1 / 24,
  1 / 6,
  1 / 2,
  1,
  1
];
const EXP_POWERS = [1];
for (let index = 1; index < 64; index++) EXP_POWERS.push(EXP_POWERS[index - 1]! * 0.5);

export function gemma4V2PortableExp(value: number): number {
  const exponent = Math.floor(value * 1.4426950408889634 + 0.5);
  const reduced = value - exponent * 0.6931471803691238 - exponent * 1.9082149292705877e-10;
  let result = EXP_COEFFICIENTS[0]!;
  for (let index = 1; index < EXP_COEFFICIENTS.length; index++) result = result * reduced + EXP_COEFFICIENTS[index]!;
  return Math.fround(result * EXP_POWERS[-exponent]!);
}
