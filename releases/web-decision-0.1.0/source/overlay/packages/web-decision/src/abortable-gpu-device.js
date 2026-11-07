import {waitForAbort} from './abort.js';
import {retainStrandsResources} from './strands/disposal-guard.mjs';

export function destroyGpuDevice(device) {
  if (!device) return;
  try {
    device.destroy();
  } catch {
    retainStrandsResources(undefined, device);
  }
}

/** Stop waiting for an adapter when the caller cancels. */
export async function requestGpuAdapter(gpu, signal) {
  try {
    signal?.throwIfAborted();
    const adapter = await waitForAbort(() => gpu.requestAdapter({powerPreference: 'high-performance'}), signal);
    signal?.throwIfAborted();
    return adapter;
  } catch (cause) {
    signal?.throwIfAborted();
    throw cause;
  }
}

/** Destroy a device that arrives after cancellation, retaining it if destroy fails. */
export async function requestGpuDevice(adapter, descriptor, signal) {
  let device;
  try {
    signal?.throwIfAborted();
    device = await waitForAbort(() => adapter.requestDevice(descriptor), signal, destroyGpuDevice);
    signal?.throwIfAborted();
    return device;
  } catch (cause) {
    destroyGpuDevice(device);
    signal?.throwIfAborted();
    throw cause;
  }
}
