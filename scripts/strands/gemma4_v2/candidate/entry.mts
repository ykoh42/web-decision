import {createDecisionModelLoader} from '../../../../packages/web-decision/src/curated/load-decision-model.ts';
import {createGemma4V2CuratedAdapter} from '../../../../packages/web-decision/src/curated/adapters/strands-gemma4-v2.mts';
import {GEMMA4_V2_MODEL} from '../../../../packages/web-decision/src/strands/gemma4-v2-loader.mts';
import {loadGemma4V2Runtime} from '../../../../packages/web-decision/src/strands/gemma4-v2-runtime-loader.mts';

declare const __GEMMA_V2_CANDIDATE_PINS__: Record<string, {bytes: number; sha256: string}>;
const base = new URL('./gemma4-v2/', import.meta.url);
const asset = (name: string) => ({base, name, metadata: __GEMMA_V2_CANDIDATE_PINS__[name]!});
const assets = {
  wasm: asset('graph.wasm'),
  localFrequency: asset('local.bin'),
  globalFrequency: asset('global.bin'),
  scoring: asset('scoring.bin')
};
const description = Object.freeze({modelId: GEMMA4_V2_MODEL, status: 'private-test-candidate' as const});
const adapter = createGemma4V2CuratedAdapter((options) => loadGemma4V2Runtime(options, assets));

/** Private one-off test artifact only; this does not register a public model. */
export const loadDecisionModel = createDecisionModelLoader({
  isModelId: (value): value is typeof GEMMA4_V2_MODEL => value === GEMMA4_V2_MODEL,
  describe: () => description,
  loaderDescription: () => description,
  loadAdapter: () => async () => adapter,
  approvedLoadOptions: {[GEMMA4_V2_MODEL]: ['maxLength']}
});
export function listModels() {
  return [description];
}
