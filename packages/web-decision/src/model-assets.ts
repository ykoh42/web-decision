import {readVerifiedAsset} from './assets.js';
import {WebDecisionError} from './kev.js';
import {processPrefetchedShards} from './shard-prefetch.js';
import type {AssetCache, AssetFileManifest, ReadVerifiedAssetOptions} from './assets.js';
import type {ModelProgress} from './curated/types.js';

interface CreateModelAssetReaderOptions {
  readonly base: URL;
  readonly manifest: AssetFileManifest;
  readonly cache: AssetCache | null | undefined;
  readonly signal?: AbortSignal | undefined;
  readonly options: {
    readonly offline?: boolean | undefined;
    readonly onProgress?: ((progress: ModelProgress) => void) | undefined;
  };
  readonly fetchImpl?: ReadVerifiedAssetOptions['fetchImpl'];
  readonly progressPhase?: string | undefined;
}

interface ModelTensorRecord {
  readonly byteOffset: number;
  readonly dtype: string;
  readonly name: string;
  readonly nbytes: number;
  readonly shape: number[];
}

interface ModelShardRecord {
  readonly dataPath: string;
  readonly nbytes: number;
  readonly records: readonly ModelTensorRecord[];
}

interface ModelAssetTensor {
  copyFromRawBytes(bytes: Uint8Array): void;
}

interface ModelAssetDevice {
  sync(): Promise<void>;
}

interface ModelAssetRuntime {
  empty(shape: number[], dtype: string, device: ModelAssetDevice): ModelAssetTensor;
}

interface UploadModelShardsOptions {
  readonly records: readonly ModelShardRecord[];
  readonly read: (name: string, signal?: AbortSignal | undefined) => Promise<ArrayBuffer>;
  readonly runtime: ModelAssetRuntime;
  readonly device: ModelAssetDevice;
  readonly weights: ModelAssetTensor[];
  readonly signal?: AbortSignal | undefined;
  readonly maxPrefetchBytes: number | undefined;
  readonly uncaptured: readonly string[];
}

/** Read verified assets while tracking progress and whether all files were cached. */
export function createModelAssetReader({
  base,
  manifest,
  cache,
  signal,
  options,
  fetchImpl,
  progressPhase
}: CreateModelAssetReaderOptions) {
  const totalBytes = Object.values(manifest.files).reduce((sum, entry) => sum + entry.bytes, 0);
  let loadedBytes = 0,
    cacheComplete = true,
    currentProgressPhase = progressPhase;
  const read = (name: string, readSignal: AbortSignal | undefined = signal) =>
    readVerifiedAsset(base, manifest, name, {
      cache,
      signal: readSignal,
      offline: options.offline === true,
      fetchImpl,
      onCacheStatus: (cached) => {
        if (!cached) cacheComplete = false;
      },
      onBytes: (count) => {
        loadedBytes += count;
        options.onProgress?.({
          loadedBytes,
          totalBytes,
          fraction: totalBytes ? loadedBytes / totalBytes : 1,
          ...(currentProgressPhase ? {phase: currentProgressPhase as Exclude<ModelProgress['phase'], undefined>} : {})
        });
      }
    });
  return {
    read,
    isCacheComplete: () => cacheComplete,
    setProgressPhase(phase: string | undefined) {
      currentProgressPhase = phase;
    }
  };
}

/** Upload each verified tensor shard in order, with optional bounded read-ahead. */
export async function uploadModelShards({
  records,
  read,
  runtime,
  device,
  weights,
  signal,
  maxPrefetchBytes,
  uncaptured
}: UploadModelShardsOptions): Promise<void> {
  const upload = async (shard: ModelShardRecord, data: ArrayBuffer): Promise<void> => {
    const bytes = new Uint8Array(data);
    for (const record of shard.records) {
      const index = Number(record.name.slice('param_'.length));
      const tensor = runtime.empty(record.shape, record.dtype, device);
      tensor.copyFromRawBytes(bytes.subarray(record.byteOffset, record.byteOffset + record.nbytes));
      weights[index] = tensor;
    }
    await device.sync();
    if (uncaptured.length) throw new WebDecisionError('INFERENCE_ERROR', uncaptured[0]!);
  };
  if ((maxPrefetchBytes ?? 0) > 0) {
    await processPrefetchedShards(
      records,
      (shard, readSignal) => read(`weights/${shard.dataPath}`, readSignal),
      upload,
      signal,
      maxPrefetchBytes
    );
  } else {
    for (const shard of records) await upload(shard, await read(`weights/${shard.dataPath}`, signal));
  }
}
