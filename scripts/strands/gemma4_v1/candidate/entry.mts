import {WebDecisionError} from '../../../../packages/web-decision/src/kev.js';
import {loadGemma4V1Private, type Options} from '../../../../packages/web-decision/src/strands/gemma4-v1-runtime.mjs';
import {STRANDS_GEMMA4_V1_E2B_MODEL as modelId} from '../../../../packages/web-decision/src/strands/gemma4-v1-session.mjs';
import type {AssetFileManifest} from '../../../../packages/web-decision/src/assets.js';

declare const __GEMMA_V1_PRIVATE_PINS__: AssetFileManifest;
const entries = Object.freeze([Object.freeze({modelId, status: 'private-test-candidate'})]);

export function listModels() {
  return entries;
}

export function loadDecisionModel(options: Options & {readonly modelId: string}) {
  if (options?.modelId !== modelId) throw new WebDecisionError('INVALID_INPUT', 'Unsupported Gemma 4 v1 model ID.');
  return loadGemma4V1Private(new URL('./gemma4-v1/', import.meta.url), __GEMMA_V1_PRIVATE_PINS__, options);
}
