import {WebDecisionError} from '../curated/errors.ts';

type TvmOutOfMemory = Readonly<{size?: number}>;
type ErrorRecord = Readonly<Record<string, unknown>>;

function isErrorRecord(value: unknown): value is ErrorRecord {
  return value !== null && typeof value === 'object';
}

function findOutOfMemory(value: unknown, seen: Set<object>): TvmOutOfMemory | undefined {
  if (!isErrorRecord(value) || seen.has(value)) return undefined;
  seen.add(value);

  if (value.code === 'GPU_OUT_OF_MEMORY') {
    const size = value.size;
    return typeof size === 'number' && Number.isSafeInteger(size) && size >= 0 ? {size} : {};
  }

  if (Array.isArray(value.errors)) {
    for (const error of value.errors) {
      const found = findOutOfMemory(error, seen);
      if (found) return found;
    }
  }

  return findOutOfMemory(value.cause, seen);
}

/** Convert a structured TVM WebGPU allocation failure to the public device-limit error. */
export function mapKananaTvmResourceError(cause: unknown): unknown {
  if (cause instanceof WebDecisionError && cause.code === 'DEVICE_RESOURCE_LIMIT') return cause;

  const failure = findOutOfMemory(cause, new Set());
  if (!failure) return cause;

  const detail =
    failure.size === undefined
      ? 'Kanana could not allocate a required WebGPU buffer on this device.'
      : `Kanana could not allocate a ${failure.size}-byte WebGPU buffer on this device.`;
  return new WebDecisionError('DEVICE_RESOURCE_LIMIT', detail, {cause});
}
