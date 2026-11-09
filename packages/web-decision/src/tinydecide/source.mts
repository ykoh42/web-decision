import {openAssetCache, readVerifiedAsset, type AssetFileManifest} from '../assets.js';
import {TINYDECIDE_SPEC} from './contract.mjs';
import type {TinyDecideAssetName, TinyDecideAssetSource, TinyDecideSourceOptions} from './source-types.mjs';

export type {
  TinyDecideAssetFile,
  TinyDecideAssetManifestFile,
  TinyDecideAssetName,
  TinyDecideAssetSource,
  TinyDecideAssetSpec,
  TinyDecideProgress,
  TinyDecideSourceOptions
} from './source-types.mjs';

export async function openTinyDecideSource({
  base,
  signal,
  cacheEnabled,
  offline,
  onProgress,
  spec = TINYDECIDE_SPEC
}: TinyDecideSourceOptions): Promise<TinyDecideAssetSource> {
  const cache = await openAssetCache(cacheEnabled, signal);
  let loadedBytes = 0;
  return {
    async read(name: TinyDecideAssetName) {
      const buffer = await readVerifiedAsset(base, spec.manifest as AssetFileManifest, name, {
        cache,
        signal,
        offline,
        onBytes(bytes) {
          loadedBytes += bytes;
          onProgress?.({
            loadedBytes,
            totalBytes: spec.totalBytes,
            fraction: loadedBytes / spec.totalBytes
          });
        }
      });
      return buffer;
    }
  };
}
