import type {AssetSourceOptions, ManifestSourceSpec} from '../asset-source-types.js';
import {openAssetCache, readVerifiedAsset} from '../assets.js';
import {SAFEDECIDE_SPEC} from './contract.mjs';

export async function openSafeDecideSource({
  base,
  signal,
  cacheEnabled,
  offline,
  onProgress,
  spec = SAFEDECIDE_SPEC
}: AssetSourceOptions & {spec?: ManifestSourceSpec}) {
  const cache = await openAssetCache(cacheEnabled, signal);
  let loadedBytes = 0;
  return {
    async read(name: string): Promise<ArrayBuffer> {
      const options = {
        cache,
        signal,
        offline,
        maxAssetBytes: spec.maxAssetBytes,
        onBytes(count: number) {
          loadedBytes += count;
          onProgress?.({loadedBytes, totalBytes: spec.totalBytes, fraction: loadedBytes / spec.totalBytes});
        }
      };
      const bytes: ArrayBuffer = await readVerifiedAsset(base, spec.manifest, name, options);
      return bytes;
    }
  };
}
