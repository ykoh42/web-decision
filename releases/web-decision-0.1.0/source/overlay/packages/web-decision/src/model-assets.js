import {readVerifiedAsset} from './assets.js';
import {WebDecisionError} from './kev.js';
import {processPrefetchedShards} from './shard-prefetch.js';

/** Read verified assets while tracking progress and whether all files were cached. */
export function createModelAssetReader({base, manifest, cache, signal, options}) {
  const totalBytes = Object.values(manifest.files).reduce((sum, entry) => sum + entry.bytes, 0);
  let loadedBytes = 0,
    cacheComplete = true;
  const read = (name, readSignal = signal) =>
    readVerifiedAsset(base, manifest, name, {
      cache,
      signal: readSignal,
      offline: options.offline === true,
      onCacheStatus: (cached) => {
        if (!cached) cacheComplete = false;
      },
      onBytes: (count) => {
        loadedBytes += count;
        options.onProgress?.({loadedBytes, totalBytes, fraction: totalBytes ? loadedBytes / totalBytes : 1});
      }
    });
  return {read, isCacheComplete: () => cacheComplete};
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
}) {
  const upload = async (shard, data) => {
    const bytes = new Uint8Array(data);
    for (const record of shard.records) {
      const index = Number(record.name.slice('param_'.length));
      const tensor = runtime.empty(record.shape, record.dtype, device);
      tensor.copyFromRawBytes(bytes.subarray(record.byteOffset, record.byteOffset + record.nbytes));
      weights[index] = tensor;
    }
    await device.sync();
    if (uncaptured.length) throw new WebDecisionError('INFERENCE_ERROR', uncaptured[0]);
  };
  if (maxPrefetchBytes > 0) {
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
