import {openAssetCache} from '../assets.js';
import {loadStrandsFromBytes} from './loader-core.mjs';
import {openPinnedRangeSource} from './range-source.mjs';
import {createSourceLoader} from './source-loader-internal.mjs';

/** Experimental source path with pinned ranges and optional offline cache. */
export const loadExperimentalStrandsFromSource = createSourceLoader({
  openSource: openPinnedRangeSource,
  openCache: openAssetCache,
  loadBytes: loadStrandsFromBytes
});
