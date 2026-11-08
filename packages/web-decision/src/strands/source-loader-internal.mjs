import {WebDecisionError} from '../kev.js';
import {validatePrefetchBytes} from '../shard-prefetch.js';
import {requestStrandsAdapter} from './preflight.mjs';

/** Internal factory; dependencies are selected by the module, never by user options. */
export function createSourceLoader({openSource, openCache, loadBytes}) {
  return async function loadSource({
    assetsUrl,
    cache = true,
    offline = false,
    signal,
    createTokenizer,
    createRuntime,
    gpuApi,
    onProgress,
    maxPrefetchBytes
  } = {}) {
    if (
      typeof assetsUrl !== 'string' ||
      !assetsUrl ||
      typeof cache !== 'boolean' ||
      typeof offline !== 'boolean' ||
      (offline && !cache)
    )
      throw new WebDecisionError('INVALID_INPUT', 'Invalid Strands asset or cache options.');
    if (validatePrefetchBytes(maxPrefetchBytes) !== 0)
      throw new WebDecisionError('INVALID_INPUT', 'Strands does not support prefetching.');
    let base;
    try {
      base = new URL(assetsUrl, globalThis.location?.href);
    } catch {
      throw new WebDecisionError('INVALID_INPUT', 'Invalid asset URL.');
    }
    if (!['http:', 'https:'].includes(base.protocol) || base.search || base.hash || base.username || base.password)
      throw new WebDecisionError(
        'INVALID_INPUT',
        'The asset URL must be an HTTP(S) directory without credentials, query, or fragment.'
      );
    if (!base.pathname.endsWith('/')) base.pathname += '/';
    signal?.throwIfAborted();
    const preparedAdapter = await requestStrandsAdapter(gpuApi, signal);
    signal?.throwIfAborted();
    const storage = await openCache(cache, signal);
    signal?.throwIfAborted();
    const {bytes, transaction} = await openSource({base, cache: storage, offline, signal});
    return loadBytes(
      {
        base,
        manifestBytes: bytes.manifest,
        wasmBytes: bytes.wasm,
        tokenizerBytes: bytes.tokenizer,
        tokenizerConfigBytes: bytes.tokenizerConfig,
        createTokenizer,
        createRuntime,
        gpuApi,
        signal,
        onProgress,
        cache: false,
        offline: false
      },
      {
        preparedAdapter,
        readRange: (rangeBase, spec, options) => transaction.read(rangeBase, spec, options),
        finalize: (loadSignal) => transaction.commit({signal: loadSignal})
      }
    );
  };
}
