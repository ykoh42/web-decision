import type {AssetSourceOptions, AssetSourceSpec} from '../asset-source-types.js';
import {openAssetCache, readVerifiedAsset} from '../assets.js';
import {ETTIN_PROFILE} from './contract.mjs';

export async function openEttinSource(
  {base, signal, cacheEnabled, offline, onProgress}: AssetSourceOptions,
  profile: AssetSourceSpec = ETTIN_PROFILE
) {
  const cache = await openAssetCache(cacheEnabled, signal);
  let loadedBytes = 0;
  return {
    async read(name: string): Promise<ArrayBuffer> {
      const options = {
        cache,
        signal,
        offline,
        onBytes(bytes: number) {
          loadedBytes += bytes;
          onProgress?.({loadedBytes, totalBytes: profile.totalBytes, fraction: loadedBytes / profile.totalBytes});
        }
      };
      const buffer: ArrayBuffer = await readVerifiedAsset(base, {files: profile.files}, name, options);
      return buffer;
    }
  };
}
