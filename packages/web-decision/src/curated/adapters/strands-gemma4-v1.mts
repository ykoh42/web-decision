import type {AssetFileManifest} from '../../assets.js';
import {STRANDS_GEMMA4_V1_E2B_MODEL} from '../../strands/gemma4-v1-session.mjs';
import {loadGemma4V1, type Options, type RuntimeAssetUrls} from '../../strands/gemma4-v1-runtime.mjs';
import {WebDecisionError} from '../errors.ts';

/** Bind the verified E2B v1 artifacts without exposing a caller-supplied asset URL. */
export function createGemma4V1CuratedAdapter(urls: RuntimeAssetUrls, manifest: AssetFileManifest) {
  return {
    async loadModel(rawOptions: object, description: {modelId: string}) {
      if (description.modelId !== STRANDS_GEMMA4_V1_E2B_MODEL)
        throw new WebDecisionError('INVALID_ASSET', 'Gemma 4 v1 adapter received a different model descriptor.');
      const options = rawOptions as Options;
      if (
        (options.cache !== undefined && typeof options.cache !== 'boolean') ||
        (options.offline !== undefined && typeof options.offline !== 'boolean') ||
        (options.offline && options.cache === false) ||
        (options.onProgress !== undefined && typeof options.onProgress !== 'function')
      )
        throw new WebDecisionError('INVALID_INPUT', 'Invalid Gemma 4 v1 load options.');
      const session = await loadGemma4V1(urls, manifest, {...options, cache: options.cache ?? true});
      return {...session, modelId: STRANDS_GEMMA4_V1_E2B_MODEL, description};
    }
  };
}
