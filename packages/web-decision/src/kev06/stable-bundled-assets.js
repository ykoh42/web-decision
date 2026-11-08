import modelWasmUrl from '../../assets/kev06-stable/model.wasm?url&inline';
import tokenizerUrl from '../../assets/experimental-kev-upstream/tokenizer.json?url&inline';
import tokenizerConfigUrl from '../../assets/experimental-kev-upstream/tokenizer_config.json?url&inline';
import modelConfigUrl from '../../assets/experimental-kev-upstream/model_config.json?url&inline';
import tensorCacheUrl from '../../assets/experimental-kev-upstream/weights/tensor-cache.json?url&inline';

export const stableKevBundledUrls = Object.freeze({
  'model.wasm': modelWasmUrl,
  'tokenizer.json': tokenizerUrl,
  'tokenizer_config.json': tokenizerConfigUrl,
  'model_config.json': modelConfigUrl,
  'weights/tensor-cache.json': tensorCacheUrl
});
