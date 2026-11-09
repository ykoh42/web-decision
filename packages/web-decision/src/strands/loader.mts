import {openAssetCache} from '../assets.js';
import {loadStrandsFromBytes} from './loader-core.mjs';
import {openPinnedRangeSource} from './range-source.mjs';
import {createSourceLoader} from './source-loader-internal.mjs';
import type {StrandsSourceOptions} from './source-loader-internal.mjs';
import type {ExperimentalStrandsSession} from './session.mjs';

export type {StrandsSourceOptions} from './source-loader-internal.mjs';

/** Experimental source path with pinned ranges and optional offline cache. */
export const loadExperimentalStrandsFromSource: (
  options?: StrandsSourceOptions
) => Promise<ExperimentalStrandsSession> = createSourceLoader<ExperimentalStrandsSession>({
  openSource: openPinnedRangeSource,
  openCache: openAssetCache,
  loadBytes: loadStrandsFromBytes
});
