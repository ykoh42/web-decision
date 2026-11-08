import {getFakeKevLoaderState} from '../state.ts';

getFakeKevLoaderState().stableHelperLoads++;
getFakeKevLoaderState().events.push('stable-helper');

export const stableKevBundledUrls: Readonly<Record<string, string>> = Object.freeze({
  'model.wasm': 'data:application/wasm;base64,BAUG',
  'tokenizer.json': 'data:application/json;base64,e30=',
  'tokenizer_config.json': 'data:application/json;base64,e30=',
  'model_config.json': 'data:application/json;base64,e30=',
  'weights/tensor-cache.json': 'data:application/json;base64,e30='
});
