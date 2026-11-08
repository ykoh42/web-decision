import {requestFp16Device} from '../fp16-device.js';
import {MAPIKA_Q4_LARGEST_BUFFER} from './contract.js';

/** Request shader-f16 and the packed embedding's maximum storage-binding size. */
export function requestMapikaQ4Device(gpu, signal) {
  return requestFp16Device(gpu, signal, 'Mapika Q4', {
    maxBufferSize: MAPIKA_Q4_LARGEST_BUFFER,
    maxStorageBufferBindingSize: MAPIKA_Q4_LARGEST_BUFFER,
    maxStorageBuffersPerShaderStage: 8
  });
}
