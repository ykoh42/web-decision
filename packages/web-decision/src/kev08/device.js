import {requestFp16Device} from '../fp16-device.js';
import {KEV08_LARGEST_BUFFER} from './contract.js';

export function requestKev08Device(gpu, signal) {
  return requestFp16Device(gpu, signal, 'Kev 0.8B', {
    maxBufferSize: KEV08_LARGEST_BUFFER,
    maxStorageBufferBindingSize: KEV08_LARGEST_BUFFER,
    maxStorageBuffersPerShaderStage: 8
  });
}
