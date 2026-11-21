import {requestGpuAdapter, requestGpuDevice} from '../abortable-gpu-device.js';
import type {AssetCache} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {buildQwen35PerImageVisionGeometry} from '../strands/qwen35-vision-geometry.mts';
import {createQwen35VisionRuntimeRunnerFromSource} from '../strands/qwen35-vision-runtime.mts';
import {MAPIKA2B_VISION, type Mapika2bImageAsset} from './contract.mts';
import {cacheMapika2bImageCodecs, createMapika2bImageCodecs} from './image-codecs.mts';
import {preprocessMapika2bImage} from './images.mts';
import type {Mapika2bVisionFeatures} from './session.mts';
import type {Mapika2bSource} from './source.mts';

const WORKSPACE_BYTES = 1024 ** 3;
const WORKGROUP_STORAGE_BYTES = 8 * 1024;
const VISION_GEOMETRY = Object.freeze({
  numPositionEmbeddings: 2304,
  spatialMergeSize: MAPIKA2B_VISION.mergeSize,
  maxImages: 1
});

export interface OpenMapika2bVisionOptions {
  readonly source: Pick<Mapika2bSource, 'readTensor'>;
  readonly cache?: AssetCache | null | undefined;
  readonly offline?: boolean | undefined;
  readonly signal?: AbortSignal | null | undefined;
  readonly onCacheStatus?: ((cached: boolean) => void) | undefined;
}

export interface Mapika2bVision {
  encodeImage(asset: Mapika2bImageAsset): Promise<Mapika2bVisionFeatures>;
  dispose(): Promise<void>;
}

function deviceDescriptor(adapter: GPUAdapter): GPUDeviceDescriptor {
  const {
    maxBufferSize,
    maxStorageBufferBindingSize,
    maxComputeWorkgroupStorageSize,
    maxComputeInvocationsPerWorkgroup,
    maxStorageBuffersPerShaderStage
  } = adapter.limits;
  if (Math.min(maxBufferSize, maxStorageBufferBindingSize) < WORKSPACE_BYTES)
    throw new WebDecisionError(
      'DEVICE_RESOURCE_LIMIT',
      `Mapika vision needs WebGPU buffers and bindings of at least ${WORKSPACE_BYTES} bytes. The adapter provides ${maxBufferSize} and ${maxStorageBufferBindingSize} bytes.`
    );
  if (maxComputeWorkgroupStorageSize < WORKGROUP_STORAGE_BYTES)
    throw new WebDecisionError(
      'DEVICE_RESOURCE_LIMIT',
      `Mapika vision needs ${WORKGROUP_STORAGE_BYTES} bytes of WebGPU workgroup storage. The adapter provides ${maxComputeWorkgroupStorageSize} bytes.`
    );
  const requiredFeatures: GPUFeatureName[] = ['shader-f16', 'subgroups'];
  return {
    requiredFeatures: requiredFeatures.filter((feature) => adapter.features.has(feature)),
    requiredLimits: {
      maxBufferSize,
      maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: WORKGROUP_STORAGE_BYTES,
      maxComputeInvocationsPerWorkgroup,
      maxStorageBuffersPerShaderStage
    }
  };
}

async function requestVisionDevice(gpu: GPU | undefined, signal?: AbortSignal): Promise<GPUDevice> {
  if (!gpu) throw new WebDecisionError('UNSUPPORTED_ENVIRONMENT', 'Mapika vision requires WebGPU.');
  const adapter = await requestGpuAdapter(gpu, signal);
  if (!adapter)
    throw new WebDecisionError('UNSUPPORTED_ENVIRONMENT', 'No WebGPU adapter is available for Mapika vision.');
  return requestGpuDevice(adapter, deviceDescriptor(adapter), signal);
}

/** Fill the packaged graph cache while the native weights are prefetched separately. */
export async function cacheMapika2bVisionAssets({
  cache,
  offline,
  signal,
  onCacheStatus
}: Pick<OpenMapika2bVisionOptions, 'cache' | 'offline' | 'signal' | 'onCacheStatus'>): Promise<void> {
  signal?.throwIfAborted();
  const {openPinnedQwen35VisionArtifacts} = await import('../strands/qwen35-vision-artifacts.mts');
  await Promise.all([
    openPinnedQwen35VisionArtifacts({cache, offline, signal, onCacheStatus}),
    cacheMapika2bImageCodecs({cache, offline, signal, onCacheStatus})
  ]);
}

/** Run the identical pinned Qwen3.5 visual tower with Mapika's image processor and checkpoint. */
export async function openMapika2bVision({
  source,
  cache,
  offline,
  signal
}: OpenMapika2bVisionOptions): Promise<Mapika2bVision> {
  signal?.throwIfAborted();
  const [tvmjs, {openPinnedQwen35VisionArtifacts}] = await Promise.all([
    import('../runtime/tvm.generated.mjs'),
    import('../strands/qwen35-vision-artifacts.mts')
  ]);
  const artifacts = await openPinnedQwen35VisionArtifacts({cache, offline, signal});
  const gpu = typeof navigator === 'undefined' ? undefined : navigator.gpu;
  // TVM owns the GPU device, so vision requests its own adapter and device.
  const gpuDevice = await requestVisionDevice(gpu, signal ?? undefined);
  const runner = await createQwen35VisionRuntimeRunnerFromSource({
    tvmjs,
    gpuDevice,
    artifacts,
    signal,
    readWeight: ({sourceName}, readSignal) => source.readTensor('vision', sourceName, {signal: readSignal})
  });

  const codecs = createMapika2bImageCodecs({cache, offline});
  return Object.freeze({
    async encodeImage(asset: Mapika2bImageAsset): Promise<Mapika2bVisionFeatures> {
      const {imageGridThw: grid, pixelShape, pixelValues} = await preprocessMapika2bImage(asset, codecs);
      const [embeddings] = await runner.run({
        imageCount: 1,
        imageGridThw: Int32Array.from(grid),
        imageInput: {imageGridThw: [grid], pixelShape, pixelValues},
        visionGeometry: buildQwen35PerImageVisionGeometry([grid], VISION_GEOMETRY)
      });
      if (!embeddings) throw new WebDecisionError('INFERENCE_ERROR', 'TVM returned no Mapika vision features.');
      return {grid, embeddings};
    },
    dispose: runner.dispose
  });
}
