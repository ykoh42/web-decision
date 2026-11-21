export {loadMapika2bCandidate} from '../../packages/web-decision/src/mapika2b/loader.mts';
import {loadModel} from '../../packages/web-decision/src/curated/adapters/mapika2b.ts';
import type {ModelDescription} from '../../packages/web-decision/src/curated/types.js';

/** Exercise the curated adapter from an installed candidate package. */
export function loadMapika2bCuratedForTest(options: object) {
  return loadModel(options, {modelId: 'mapika-decider-2b-v12'} as unknown as ModelDescription);
}

export function mapika2bPackagedAsset(name: 'text.wasm' | 'text-bindings.json'): string {
  const bundleUrl = import.meta.url;
  return new URL('mapika2b/' + name, bundleUrl).href;
}

/** Load the vision implementation and URL modules without requesting any model asset. */
export async function warmMapika2bPackagedCode(): Promise<void> {
  await import('../../packages/web-decision/src/strands/qwen35-vision-artifacts.mts');
}
