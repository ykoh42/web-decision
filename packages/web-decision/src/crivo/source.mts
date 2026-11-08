import type {AssetSourceOptions, AssetSourceSpec} from '../asset-source-types.js';
import {openAssetCache, readVerifiedAsset} from '../assets.js';

export async function openCrivoSource(
  {base, signal, cacheEnabled, offline, onProgress}: AssetSourceOptions,
  spec: AssetSourceSpec
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
          onProgress?.({loadedBytes, totalBytes: spec.totalBytes, fraction: loadedBytes / spec.totalBytes});
        }
      };
      const buffer: ArrayBuffer = await readVerifiedAsset(base, {files: spec.files}, name, options);
      return buffer;
    }
  };
}
