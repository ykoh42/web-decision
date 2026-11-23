import {openAssetCache, type AssetCache} from '../../assets.js';
import {GEMMA4_V2_MODEL, type Gemma4V2LoaderOptions} from '../../strands/gemma4-v2-loader.mjs';
import {pinnedGemma4V2OfflineAssets} from '../../strands/gemma4-v2-offline.mjs';
import {WebDecisionError} from '../errors.ts';
import type {ModelProgress} from '../types.js';

interface Session {
  systemOne: (...args: never[]) => unknown;
  releaseCachedMemory: () => unknown;
  prepareOffline: (...args: never[]) => unknown;
  isOfflineReady: (...args: never[]) => unknown;
  dispose: () => void | Promise<void>;
}
type RuntimeLoader<S extends Session> = (options: Gemma4V2LoaderOptions) => Promise<S>;
interface Options {
  readonly cache?: boolean;
  readonly offline?: boolean;
  readonly signal?: AbortSignal;
  readonly maxLength?: number;
  readonly onProgress?: (progress: ModelProgress) => void;
}
const sourceAssets = pinnedGemma4V2OfflineAssets([]);
const staticBytes = [
  sourceAssets.modelConfig,
  sourceAssets.decisionConfig,
  sourceAssets.processorConfig,
  sourceAssets.tokenizer,
  sourceAssets.tokenizerConfig,
  sourceAssets.readout
].reduce((sum, asset) => sum + asset.metadata.bytes, 0);

/** Internal adapter; registration requires a fully verified and pinned graph. */
export function createGemma4V2CuratedAdapter<S extends Session>(
  loadRuntime: RuntimeLoader<S>,
  cacheOpener: typeof openAssetCache = openAssetCache
) {
  return {
    async loadModel(rawOptions: object, description: {modelId: string}) {
      if (description.modelId !== GEMMA4_V2_MODEL)
        throw new WebDecisionError('INVALID_ASSET', 'Gemma 4 v2 adapter received a different model descriptor.');
      const options = rawOptions as Options;
      if (
        (options.cache !== undefined && typeof options.cache !== 'boolean') ||
        (options.offline !== undefined && typeof options.offline !== 'boolean') ||
        (options.offline && options.cache === false) ||
        (options.maxLength !== undefined && (!Number.isSafeInteger(options.maxLength) || options.maxLength < 1)) ||
        (options.onProgress !== undefined && typeof options.onProgress !== 'function')
      )
        throw new WebDecisionError('INVALID_INPUT', 'Invalid Gemma 4 v2 load options.');
      const cache: AssetCache | null = await cacheOpener(options.cache !== false, options.signal);
      let loadedBytes = 0;
      const session = await loadRuntime({
        cache,
        offline: options.offline ?? false,
        ...(options.maxLength === undefined ? {} : {maxLength: options.maxLength}),
        ...(options.signal === undefined ? {} : {signal: options.signal}),
        ...(options.onProgress === undefined
          ? {}
          : {
              onBytes(bytes: number) {
                loadedBytes += bytes;
                options.onProgress!({
                  loadedBytes,
                  totalBytes: staticBytes,
                  fraction: loadedBytes / staticBytes,
                  phase: 'model-assets'
                });
              }
            })
      });
      let disposal: Promise<void> | undefined;
      return {
        modelId: GEMMA4_V2_MODEL,
        description,
        systemOne: session.systemOne.bind(session) as S['systemOne'],
        releaseCachedMemory: session.releaseCachedMemory.bind(session) as S['releaseCachedMemory'],
        prepareOffline: session.prepareOffline.bind(session) as S['prepareOffline'],
        isOfflineReady: session.isOfflineReady.bind(session) as S['isOfflineReady'],
        dispose() {
          if (!disposal) disposal = Promise.resolve().then(() => session.dispose());
          return disposal;
        }
      };
    }
  };
}
