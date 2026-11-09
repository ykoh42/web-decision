import {openAssetCache, readVerifiedAsset} from '../assets.js';
import type {AssetSourceOptions, ManifestSourceSpec} from '../asset-source-types.js';
import type {ModelProgress} from '../curated/types.js';
import {DECIMA_SMALL_SPEC} from './contract.mjs';

export interface DecimaSourceOptions extends AssetSourceOptions {
  readonly spec?: ManifestSourceSpec | undefined;
}

export interface DecimaSource {
  read(name: string): Promise<ArrayBuffer>;
}

export async function openDecimaSource({
  base,
  signal,
  cacheEnabled,
  offline,
  onProgress,
  spec = DECIMA_SMALL_SPEC
}: DecimaSourceOptions): Promise<DecimaSource> {
  const cache = await openAssetCache(cacheEnabled, signal);
  let loadedBytes = 0;
  return {
    async read(name: string): Promise<ArrayBuffer> {
      const buffer = await readVerifiedAsset(base, spec.manifest, name, {
        cache,
        signal,
        offline,
        maxAssetBytes: spec.maxAssetBytes,
        onBytes(bytes: number): void {
          loadedBytes += bytes;
          onProgress?.({
            loadedBytes,
            totalBytes: spec.totalBytes,
            fraction: loadedBytes / spec.totalBytes
          } satisfies ModelProgress);
        }
      });
      return buffer;
    }
  };
}
