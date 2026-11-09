/** Encode one uncached prefix and question branch for the experimental graph. */
export interface ElephantGraphInput {
  readonly dtype: 'int32' | 'float16';
  readonly shape: number[];
  readonly data: Int32Array | Uint16Array;
}

export function validateGraphInputs(ids: unknown, prefixLength: number, maxBranchTokens = 128): number[] {
  if (
    ![128, 512].includes(maxBranchTokens) ||
    !Array.isArray(ids) ||
    ids.length < 4 ||
    ids.length > maxBranchTokens ||
    !Number.isSafeInteger(prefixLength) ||
    prefixLength < 1 ||
    prefixLength >= ids.length
  )
    throw Error('Invalid Elephant graph input.');
  const snapshot = [];
  for (let index = 0; index < (ids as unknown[]).length; index++) {
    if (!Object.hasOwn(ids as unknown[], index)) throw Error('Invalid Elephant graph input.');
    const value = (ids as unknown[])[index];
    if (!Number.isSafeInteger(value) || (value as number) < 0 || (value as number) >= 248320)
      throw Error('Invalid Elephant graph input.');
    snapshot.push(value as number);
  }
  return snapshot;
}

/** Allocate the graph inputs only for the branch being executed. */
export function graphInputs(ids: unknown, prefixLength: number, maxBranchTokens = 128): ElephantGraphInput[] {
  const snapshot = validateGraphInputs(ids, prefixLength, maxBranchTokens);
  const length = snapshot.length;
  const tokens = Int32Array.from(snapshot);
  const mask = new Uint16Array(length * length);
  for (let row = 0; row < prefixLength; row++) {
    for (let column = prefixLength; column < length; column++) mask[row * length + column] = 0xfbff;
  }
  return [
    {dtype: 'int32', shape: [1, length], data: tokens},
    {dtype: 'int32', shape: [1, length], data: Int32Array.from({length}, (_, index) => index)},
    {dtype: 'float16', shape: [1, 1, length, length], data: mask},
    {dtype: 'float16', shape: [1, length], data: new Uint16Array(length).fill(0x3c00)},
    {dtype: 'int32', shape: [1], data: Int32Array.of(length - 1)}
  ];
}
