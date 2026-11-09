import {openAssetCache} from '../assets.js';
import {loadStrandsFromBytes} from './loader-core.mjs';
import {openPinnedRangeSource} from './range-source.mjs';
import {createSourceLoader} from './source-loader-internal.mjs';
import type {ExperimentalStrandsSession} from './session.mjs';
import type {ModelProgress} from '../curated/types.js';

export interface StrandsSourceOptions {
  readonly assetsUrl?: string;
  readonly cache?: boolean;
  readonly offline?: boolean;
  readonly signal?: AbortSignal;
  readonly createTokenizer?: (data: unknown, config: unknown) => unknown;
  readonly createRuntime?: (wasm: Uint8Array) => unknown;
  readonly gpuApi?: {requestAdapter(): Promise<unknown>};
  readonly onProgress?: (progress: ModelProgress) => void;
  readonly maxPrefetchBytes?: number;
}

/** Experimental source path with pinned ranges and optional offline cache. */
export const loadExperimentalStrandsFromSource = createSourceLoader({
  openSource: openPinnedRangeSource,
  openCache: openAssetCache,
  loadBytes: loadStrandsFromBytes
}) as (options?: StrandsSourceOptions) => Promise<ExperimentalStrandsSession>;
