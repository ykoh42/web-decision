export type Trial = {interval: 2 | 8; warmup: boolean};

/** Alternate measured order to limit systematic first/second position bias. */
export function visionPerformancePlan(rounds: number): Trial[] {
  if (!Number.isSafeInteger(rounds) || rounds < 1) throw new RangeError('Invalid measured round count.');
  const plan: Trial[] = [
    {interval: 2, warmup: true},
    {interval: 8, warmup: true}
  ];
  for (let round = 0; round < rounds; round++) {
    for (const interval of round % 2 ? ([8, 2] as const) : ([2, 8] as const)) plan.push({interval, warmup: false});
  }
  return plan;
}

export function compareVisionOutputs(left: Float32Array, right: Float32Array) {
  if (left.length !== right.length || left.length === 0) throw new Error('Vision output shapes differ.');
  let maxAbsoluteError = 0;
  let sumAbsoluteError = 0;
  for (let index = 0; index < left.length; index++) {
    if (!Number.isFinite(left[index]) || !Number.isFinite(right[index]))
      throw new Error('Vision output is non-finite.');
    const difference = Math.abs(left[index]! - right[index]!);
    maxAbsoluteError = Math.max(maxAbsoluteError, difference);
    sumAbsoluteError += difference;
  }
  return {count: left.length, maxAbsoluteError, meanAbsoluteError: sumAbsoluteError / left.length};
}
