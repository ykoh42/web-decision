import {waitForAbort} from './assets.js';
import {WebDecisionError} from './kev.js';

/** Check a required feature before model reads without requesting a device.
 * Do not return the adapter: it can expire while model files are downloading.
 */
export async function requestF16Adapter(gpu: GPU, signal?: AbortSignal): Promise<void> {
  signal?.throwIfAborted();
  let adapter;
  try {
    adapter = await waitForAbort(() => gpu.requestAdapter({powerPreference: 'high-performance'}), signal);
    signal?.throwIfAborted();
    if (!adapter) throw new Error('No compatible WebGPU adapter was found.');
  } catch (cause) {
    signal?.throwIfAborted();
    throw new WebDecisionError('MODEL_LOAD_FAILED', 'Could not initialize a compatible WebGPU device.', {cause});
  }
  if (!adapter.features.has('shader-f16')) {
    throw new WebDecisionError('UNSUPPORTED_ENVIRONMENT', 'This model requires WebGPU shader-f16 support.');
  }
}
