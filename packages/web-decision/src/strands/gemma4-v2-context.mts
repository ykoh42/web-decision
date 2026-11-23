import {WebDecisionError} from '../kev.js';

/** Sequence bound of the pinned Gemma 4 v2 TVM graph. */
export const GEMMA4_V2_COMPILED_CONTEXT_LENGTH = 131072;

export function assertGemma4V2CompiledContext(length: number): void {
  if (length > GEMMA4_V2_COMPILED_CONTEXT_LENGTH)
    throw new WebDecisionError(
      'INVALID_INPUT',
      `The compiled Gemma 4 v2 TVM graph supports at most ${GEMMA4_V2_COMPILED_CONTEXT_LENGTH} tokens.`
    );
}
