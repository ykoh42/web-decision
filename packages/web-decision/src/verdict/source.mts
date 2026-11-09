import type {AssetSourceOptions, ManifestSourceSpec} from '../asset-source-types.js';
import {openAssetCache, readVerifiedAsset} from '../assets.js';
import {VERDICT_SPEC} from './contract.mjs';

export async function openVerdictSource({
  base,
  signal,
  cacheEnabled,
  offline,
  onProgress,
  spec = VERDICT_SPEC
}: AssetSourceOptions & {spec?: ManifestSourceSpec | undefined}) {
  const cache = await openAssetCache(cacheEnabled, signal);
  let loadedBytes = 0;

  return {
    async read(name: string) {
      const bytes = await readVerifiedAsset(base, spec.manifest, name, {
        cache,
        signal,
        offline,
        maxAssetBytes: spec.maxAssetBytes,
        onBytes(count) {
          loadedBytes += count;
          onProgress?.({loadedBytes, totalBytes: spec.totalBytes, fraction: loadedBytes / spec.totalBytes});
        }
      });
      return bytes;
    }
  };
}
