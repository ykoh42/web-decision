export type KananaDeviceLimits = Readonly<{
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
}>;

/** Return WebGPU limits that cannot hold the largest pinned TVM weight tensor. */
export function missingKananaWeightLimits(limits: KananaDeviceLimits, requiredBufferBytes: number): string[] {
  if (!Number.isSafeInteger(requiredBufferBytes) || requiredBufferBytes < 1)
    throw new RangeError('Kanana weight buffer size must be a positive safe integer.');

  return (Object.keys(limits) as (keyof KananaDeviceLimits)[]).filter(
    (name) => !Number.isSafeInteger(limits[name]) || limits[name] < requiredBufferBytes
  );
}
