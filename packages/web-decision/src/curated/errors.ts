import type {ErrorCode} from './types.js';

/** Error raised by the public decision model interface. */
export class WebDecisionError extends Error {
  readonly code: ErrorCode;

  constructor(code: ErrorCode, message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = 'WebDecisionError';
    this.code = code;
  }
}
