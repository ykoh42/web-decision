import {openAssetCache, readVerifiedAsset} from '../assets.js';
import {TAMEV_MICRO_PROFILE} from './contract.mjs';
import type {TAMEV_PROFILES} from './contract.mjs';
import type {ModelProgress} from '../curated/types.js';

type TamevProfile = (typeof TAMEV_PROFILES)[keyof typeof TAMEV_PROFILES];
type TamevSignal = Pick<AbortSignal, 'throwIfAborted' | 'addEventListener' | 'removeEventListener'>;

interface TamevSourceOptions {
  readonly base: URL;
  readonly signal?: TamevSignal | null | undefined;
  readonly cacheEnabled: boolean;
  readonly offline: boolean;
  readonly onProgress?: ((progress: ModelProgress) => void) | undefined;
}

export async function openTamevSource(
  {base, signal, cacheEnabled, offline, onProgress}: TamevSourceOptions,
  profile: TamevProfile = TAMEV_MICRO_PROFILE
) {
  const cache = await openAssetCache(cacheEnabled, signal as AbortSignal | null | undefined);
  let loadedBytes = 0;
  return {
    async read(name: string): Promise<ArrayBuffer> {
      const buffer = await readVerifiedAsset(base, {files: profile.files}, name, {
        cache,
        signal: signal as AbortSignal | null | undefined,
        offline,
        onBytes(bytes) {
          loadedBytes += bytes;
          onProgress?.({
            loadedBytes,
            totalBytes: profile.totalBytes,
            fraction: loadedBytes / profile.totalBytes
          });
        }
      });
      return buffer;
    }
  };
}
