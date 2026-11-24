import {readVerifiedAsset, type AssetCache, type AssetFileMetadata} from '../assets.js';
import {waitForAbort} from '../abort.js';
import type {ModelProgress} from '../curated/types.js';
import {WebDecisionError} from '../kev.js';
import {createOfflinePreparation} from '../mapika2b/offline-preparation.mjs';
import {GEMMA4_V2_REPOSITORY, GEMMA4_V2_REVISION} from './gemma4-v2-source.mjs';

type Asset = Readonly<{base: URL; name: string; metadata: AssetFileMetadata}>;
export type Gemma4V2OfflineAssets = Readonly<{
  tokenizer: Asset;
  tokenizerConfig: Asset;
  readout: Asset;
  modelConfig: Asset;
  decisionConfig: Asset;
  processorConfig: Asset;
  artifacts: readonly Asset[];
}>;
type WeightSource = Readonly<{
  hasCompleteOfflineWeights(signal?: AbortSignal | null): Promise<boolean>;
  verifyOfflineWeights(signal?: AbortSignal | null): Promise<boolean>;
  prepareWeightsOffline(options?: {
    signal?: AbortSignal | null;
    onProgress?: (bytes: number, total: number) => void;
  }): Promise<void>;
}>;
type ImagePreparation = Readonly<{
  prepareOffline(signal?: AbortSignal): Promise<void>;
  isOfflineReady(signal?: AbortSignal): Promise<boolean>;
}>;

/** Bind source assets to the immutable checkpoint; graph pins come from the verified build. */
export function pinnedGemma4V2OfflineAssets(artifacts: readonly Asset[]): Gemma4V2OfflineAssets {
  const base = new URL(`https://huggingface.co/${GEMMA4_V2_REPOSITORY}/resolve/${GEMMA4_V2_REVISION}/`);
  return {
    tokenizer: {
      base,
      name: 'tokenizer.json',
      metadata: {bytes: 32169626, sha256: 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f'}
    },
    tokenizerConfig: {
      base,
      name: 'tokenizer_config.json',
      metadata: {bytes: 3082, sha256: '9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633'}
    },
    readout: {
      base,
      name: 'readout.safetensors',
      metadata: {bytes: 783472, sha256: '37b3f1c4670f1afb78007624e3ec2a2f55503d9b78602216e2f29a4cfb5111f6'}
    },
    modelConfig: {
      base,
      name: 'config.json',
      metadata: {bytes: 4954, sha256: '1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330'}
    },
    decisionConfig: {
      base,
      name: 'decision_config.json',
      metadata: {bytes: 5746, sha256: '3eafa83fff4e497b4283ef684e59b6928011584901ae2b273c78ccf975b59b00'}
    },
    processorConfig: {
      base,
      name: 'processor_config.json',
      metadata: {bytes: 1689, sha256: '32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c'}
    },
    artifacts
  };
}

/** Coordinate separately pinned checkpoint assets and model-specific graph artifacts. */
export function createGemma4V2OfflinePreparation(
  source: WeightSource,
  cache: AssetCache,
  assets: Gemma4V2OfflineAssets,
  images: ImagePreparation,
  offline = false
) {
  if (
    !cache ||
    !assets ||
    !Array.isArray(assets.artifacts) ||
    assets.artifacts.length === 0 ||
    !images ||
    typeof images.prepareOffline !== 'function' ||
    typeof images.isOfflineReady !== 'function'
  )
    throw new WebDecisionError('INVALID_ASSET', 'Gemma 4 v2 offline assets are incomplete.');
  const files = [
    assets.tokenizer,
    assets.tokenizerConfig,
    assets.readout,
    assets.modelConfig,
    assets.decisionConfig,
    assets.processorConfig,
    ...assets.artifacts
  ];
  if (
    files.some(
      (asset) =>
        !asset ||
        !(asset.base instanceof URL) ||
        typeof asset.name !== 'string' ||
        !asset.name ||
        !asset.metadata ||
        !Number.isSafeInteger(asset.metadata.bytes) ||
        asset.metadata.bytes < 1 ||
        !/^[0-9a-f]{64}$/u.test(asset.metadata.sha256)
    )
  )
    throw new WebDecisionError('INVALID_ASSET', 'Gemma 4 v2 offline asset metadata is invalid.');
  const pinned = files.map(({base, name, metadata}) => ({base: new URL(base.href), name, metadata: {...metadata}}));
  let disposed = false;
  const closed = new AbortController();
  const assertOpen = () => {
    if (disposed) throw new WebDecisionError('SESSION_CLOSED', 'The Gemma 4 v2 session is closed.');
  };

  async function readStaticAssets(offline: boolean, signal?: AbortSignal | null): Promise<void> {
    for (const asset of pinned) {
      let cached = false;
      await readVerifiedAsset(asset.base, {files: {[asset.name]: asset.metadata}}, asset.name, {
        cache,
        offline,
        signal,
        onCacheStatus: (value) => {
          cached = value;
        }
      });
      if (!cached) throw new WebDecisionError('ASSET_LOAD_FAILED', 'A Gemma 4 v2 asset could not be cached.');
    }
  }

  async function isOfflineReady(signal?: AbortSignal | null): Promise<boolean> {
    assertOpen();
    signal?.throwIfAborted();
    return waitForAbort(
      () =>
        waitForAbort(async () => {
          try {
            await readStaticAssets(true, signal);
            return (
              (await images.isOfflineReady(signal ?? undefined)) && (await source.hasCompleteOfflineWeights(signal))
            );
          } catch (cause) {
            signal?.throwIfAborted();
            if (cause instanceof WebDecisionError && cause.code === 'ASSET_LOAD_FAILED') return false;
            throw cause;
          }
        }, closed.signal),
      signal
    );
  }

  const preparation = createOfflinePreparation(false, async (signal, onProgress: (progress: ModelProgress) => void) => {
    await readStaticAssets(offline, signal);
    signal.throwIfAborted();
    const imagesReady = await images.isOfflineReady(signal);
    signal.throwIfAborted();
    if (!imagesReady) {
      if (offline) throw new WebDecisionError('ASSET_LOAD_FAILED', 'Image decoder is not available offline.');
      await images.prepareOffline(signal);
    }
    signal.throwIfAborted();
    const weightsReady = await source.verifyOfflineWeights(signal);
    signal.throwIfAborted();
    if (!weightsReady) {
      if (offline) throw new WebDecisionError('ASSET_LOAD_FAILED', 'Verified weights are not available offline.');
      await source.prepareWeightsOffline({
        signal,
        onProgress(loadedBytes, totalBytes) {
          onProgress({loadedBytes, totalBytes, fraction: loadedBytes / totalBytes, phase: 'upstream-source'});
        }
      });
    }
    if (!(await isOfflineReady(signal)))
      throw new WebDecisionError('ASSET_LOAD_FAILED', 'Gemma 4 v2 assets are not fully available offline.');
  });
  async function prepareOffline(options: Parameters<typeof preparation.prepareOffline>[0] = {}): Promise<void> {
    assertOpen();
    if (
      !options ||
      typeof options !== 'object' ||
      (options.onProgress !== undefined && typeof options.onProgress !== 'function')
    )
      throw new WebDecisionError('INVALID_INPUT', 'Invalid Gemma 4 v2 offline preparation options.');
    await waitForAbort(() => preparation.prepareOffline(options), closed.signal);
  }
  function dispose(): void {
    if (disposed) return;
    disposed = true;
    closed.abort(new WebDecisionError('SESSION_CLOSED', 'The Gemma 4 v2 session is closed.'));
    preparation.dispose();
  }
  return Object.freeze({prepareOffline, isOfflineReady, dispose});
}
