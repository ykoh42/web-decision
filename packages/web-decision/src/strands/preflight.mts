import {waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {checkFusedAdapter} from './manifest.mjs';

interface StrandsGpuApi {
  requestAdapter(): PromiseLike<unknown>;
}

/** Obtain and validate the actual adapter before opening the asset source. */
export async function requestStrandsAdapter(gpuApi: unknown, signal?: AbortSignal | null): Promise<unknown> {
  signal?.throwIfAborted();
  const candidate = gpuApi as StrandsGpuApi | null | undefined;
  if (typeof candidate?.requestAdapter !== 'function')
    throw new WebDecisionError(
      'UNSUPPORTED_ENVIRONMENT',
      'WebGPU is unavailable. Use a supported browser on HTTPS or localhost.'
    );
  if (typeof globalThis.crypto?.subtle?.digest !== 'function')
    throw new WebDecisionError('UNSUPPORTED_ENVIRONMENT', 'Web Crypto is unavailable. Use HTTPS or localhost.');
  let adapter: unknown;
  try {
    adapter = await waitForAbort(() => candidate.requestAdapter(), signal);
  } catch (cause) {
    signal?.throwIfAborted();
    throw new WebDecisionError('MODEL_LOAD_FAILED', 'Could not request a WebGPU adapter.', {cause});
  }
  signal?.throwIfAborted();
  checkFusedAdapter(adapter);
  signal?.throwIfAborted();
  return adapter;
}
