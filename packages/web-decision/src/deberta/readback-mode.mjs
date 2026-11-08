import {WebDecisionError} from '../kev.js';

/** Select the experimental DeBERTa readback mode; serial remains the default. */
export function validateDebertaReadback(value) {
  if (value === undefined || value === 'serial') return 'serial';
  if (value === 'packed') return 'packed';
  throw new WebDecisionError('INVALID_INPUT', 'debertaReadback must be serial or packed.');
}
