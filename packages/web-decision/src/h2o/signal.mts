import {WebDecisionError} from '../kev.js';

/** Validate the platform signal before scheduling any work, including across realms. */
export function validateH2oSignal(signal: unknown): void {
  if (signal === undefined) return;
  try {
    const getAborted = Object.getOwnPropertyDescriptor(AbortSignal.prototype, 'aborted')?.get;
    if (!getAborted) throw new Error('invalid signal');
    const candidate = signal as AbortSignal;
    if (
      typeof getAborted.call(signal) !== 'boolean' ||
      typeof candidate.addEventListener !== 'function' ||
      typeof candidate.removeEventListener !== 'function' ||
      typeof candidate.throwIfAborted !== 'function'
    )
      throw new Error('invalid signal');
  } catch {
    throw new WebDecisionError('INVALID_INPUT', 'Invalid abort signal.');
  }
}
