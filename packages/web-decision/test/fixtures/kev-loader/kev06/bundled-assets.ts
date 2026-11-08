import {getFakeKevLoaderState} from '../state.ts';

getFakeKevLoaderState().helperLoads++;
getFakeKevLoaderState().events.push('helper');

export const experimentalKevBundledUrls: Readonly<Record<string, string>> = Object.freeze({
  'model.wasm': 'data:application/wasm;base64,AA==',
  'tokenizer.json': 'data:application/json;base64,e30=',
  'tokenizer_config.json': 'data:application/json;base64,e30=',
  'model_config.json': 'data:application/json;base64,e30=',
  'weights/tensor-cache.json': 'data:application/json;base64,e30='
});
