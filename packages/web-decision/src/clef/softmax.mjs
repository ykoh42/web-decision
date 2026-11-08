/** FP32-rounded approximation; it is not guaranteed bitwise equal to Torch. */
export function approximateSoftmax(rows) {
  if (!Array.isArray(rows) || !rows.length) throw new Error('Invalid score rows');
  return rows.map((row) => {
    if (!(row instanceof Float32Array) || !row.length || row.some((value) => !Number.isFinite(value)))
      throw new Error('Invalid FP32 scores');
    const max = Math.max(...row);
    const weights = Float32Array.from(row, (value) => Math.fround(Math.exp(Math.fround(value - max))));
    let sum = 0;
    for (const value of weights) sum = Math.fround(sum + value);
    if (!(sum > 0) || !Number.isFinite(sum)) throw new Error('Invalid softmax sum');
    return Float32Array.from(weights, (value) => Math.fround(value / sum));
  });
}
