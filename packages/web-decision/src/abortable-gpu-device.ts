import {waitForAbort} from './abort.js';
import {retainStrandsResources} from './strands/disposal-guard.mjs';

interface Destroyable {
  destroy(): void;
}

export function destroyGpuDevice(device: Destroyable | null | undefined): void {
  if (!device) return;
  try {
    device.destroy();
  } catch {
    retainStrandsResources(undefined, device);
  }
}

/** Stop waiting for an adapter when the caller cancels. */
export async function requestGpuAdapter<T>(
  gpu: {requestAdapter(options: {powerPreference: 'high-performance'}): T | PromiseLike<T>},
  signal?: AbortSignal
): Promise<Awaited<T>> {
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
export async function requestGpuDevice<T extends Destroyable, D>(
  adapter: {requestDevice(descriptor: D): T | PromiseLike<T>},
  descriptor: D,
  signal?: AbortSignal
): Promise<Awaited<T>> {
  let device: Awaited<T> | undefined;
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
