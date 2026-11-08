import {WebDecisionError} from '../kev.js';

export const IMAJEV_MODEL = 'experimental-imajev-4b-candidate';

/** No manifest, range list or claimed graph ABI is trusted for execution yet. */
export function rejectUnverifiedImajevAssets(): never {
  throw new WebDecisionError(
    'UNSUPPORTED_ENVIRONMENT',
    'Imajev execution is disabled until its range hashes and compiled graph ABI are verified.'
  );
}
