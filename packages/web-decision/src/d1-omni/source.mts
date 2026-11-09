import {openAssetCache, readVerifiedAsset} from '../assets.js';
import {D1_OMNI_SPEC} from './contract.mjs';
import type {D1OmniSpec} from './contract.mjs';

interface D1OmniProgress {
  readonly loadedBytes: number;
  readonly totalBytes: number;
  readonly fraction: number;
}

interface D1OmniSourceOptions {
  readonly base: URL;
  readonly signal?: AbortSignal | null | undefined;
  readonly cacheEnabled: boolean | undefined;
  readonly offline: boolean | undefined;
  readonly onProgress?: ((progress: D1OmniProgress) => void) | undefined;
  readonly spec?: D1OmniSpec | undefined;
}

interface D1OmniSource {
  read(name: string): Promise<ArrayBuffer>;
}

export async function openD1OmniSource({
  base,
  signal,
  cacheEnabled,
  offline,
  onProgress,
  spec = D1_OMNI_SPEC
}: D1OmniSourceOptions): Promise<D1OmniSource> {
  const cache = await openAssetCache(cacheEnabled, signal);
  let loadedBytes = 0;

  return {
    async read(name) {
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
