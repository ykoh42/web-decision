import {WebDecisionError} from '../kev.js';

/** Validate the platform signal before scheduling any work, including across realms. */
export function validateH2oSignal(signal) {
  if (signal === undefined) return;
  try {
    const getAborted = Object.getOwnPropertyDescriptor(AbortSignal.prototype, 'aborted').get;
    if (
      typeof getAborted.call(signal) !== 'boolean' ||
      typeof signal.addEventListener !== 'function' ||
      typeof signal.removeEventListener !== 'function' ||
      typeof signal.throwIfAborted !== 'function'
    )
      throw new Error('invalid signal');
  } catch {
    throw new WebDecisionError('INVALID_INPUT', 'Invalid abort signal.');
  }
}
