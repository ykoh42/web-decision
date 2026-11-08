import {requestFp16Device} from '../fp16-device.js';
import {MAPIKA_LARGEST_BUFFER} from './contract.js';

/** Request the larger buffer limits needed by the pinned Mapika FP16 graph. */
export function requestMapikaDevice(gpu: GPU, signal?: AbortSignal): Promise<GPUDevice> {
  return requestFp16Device(gpu, signal, 'Mapika', {
    maxBufferSize: MAPIKA_LARGEST_BUFFER,
    maxStorageBufferBindingSize: MAPIKA_LARGEST_BUFFER,
    maxStorageBuffersPerShaderStage: 8
  });
}
