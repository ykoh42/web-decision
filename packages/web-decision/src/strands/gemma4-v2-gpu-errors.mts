import {WebDecisionError} from '../kev.js';
import {assertGemma4V2CompiledContext} from './gemma4-v2-context.mjs';
import type {Gemma4V2OwnedRunner} from './gemma4-v2-session.mjs';

// The pinned TVM JS FFI reads the WebGPU allocator's int64 argument as a signed int32.
const TVM_WEBGPU_MAX_ALLOCATION_BYTES = 0x7fffffff;

function assertAttentionBufferLimit(limits: GPUSupportedLimits, tokens: number): void {
  // Full attention binds FP32 query and output tensors of shape [1, tokens, 8, 512].
  const bytes = tokens * 8 * 512 * Float32Array.BYTES_PER_ELEMENT;
  if (bytes > TVM_WEBGPU_MAX_ALLOCATION_BYTES)
    throw new WebDecisionError(
      'DEVICE_RESOURCE_LIMIT',
      `Gemma 4 v2 full attention for ${tokens} tokens requires a ${bytes}-byte allocation. The pinned TVM WebGPU runtime supports signed 32-bit allocation sizes.`
    );
  const {maxBufferSize, maxStorageBufferBindingSize} = limits;
  if (bytes > Math.min(maxBufferSize, maxStorageBufferBindingSize))
    throw new WebDecisionError(
      'DEVICE_RESOURCE_LIMIT',
      `Gemma 4 v2 full attention for ${tokens} tokens requires a ${bytes}-byte WebGPU buffer and storage binding. This device provides ${maxBufferSize} and ${maxStorageBufferBindingSize} bytes, respectively.`
    );
}

/** Surface scoped WebGPU errors instead of returning results after a failed dispatch. */
export async function withGemma4V2GpuErrors<T>(device: GPUDevice, operation: () => Promise<T>): Promise<T> {
  device.pushErrorScope('internal');
  device.pushErrorScope('out-of-memory');
  device.pushErrorScope('validation');
  let output: T | undefined;
  let failure: unknown;
  let failed = false;
  try {
    output = await operation();
  } catch (error) {
    failure = error;
    failed = true;
  }
  let gpuError: GPUError | null = null;
  let outOfMemory = false;
  for (let index = 0; index < 3; index++) {
    try {
      const current = await device.popErrorScope();
      if (!gpuError && current) {
        gpuError = current;
        outOfMemory = index === 1;
      }
    } catch (error) {
      if (!failed) {
        failure = error;
        failed = true;
      }
    }
  }
  if (failed) throw failure;
  if (gpuError)
    throw new WebDecisionError(
      outOfMemory ? 'DEVICE_RESOURCE_LIMIT' : 'INFERENCE_ERROR',
      `WebGPU execution failed: ${gpuError.message}`
    );
  return output as T;
}

/** Attempt every cleanup action, and report the first failure if any. */
export function disposeGemma4V2Runtime(
  runtime: {endScope(): void; dispose(): void} | undefined,
  device: {destroy(): void},
  scoped: boolean,
  alreadyDisposed = false
): void {
  let failure: unknown;
  let failed = false;
  const actions = [
    () => {
      if (scoped && !alreadyDisposed) runtime?.endScope();
    },
    () => {
      if (!alreadyDisposed) runtime?.dispose();
    },
    () => device.destroy()
  ];
  for (const action of actions) {
    try {
      action();
    } catch (error) {
      if (!failed) {
        failure = error;
        failed = true;
      }
    }
  }
  if (failed) throw failure;
}

/** Track the public GPUDevice loss signal without modifying the TVM runtime. */
export function watchGemma4V2Device(device: {lost: PromiseLike<unknown>}) {
  let loss: unknown;
  const controller = new AbortController();
  let lost = false;
  let closed = false;
  const onLoss = (cause: unknown) => {
    if (!closed) {
      lost = true;
      loss = cause;
      controller.abort(new WebDecisionError('INFERENCE_ERROR', 'The GPU device was lost.', {cause}));
    }
  };
  device.lost.then(onLoss, onLoss);
  return {
    signal: controller.signal,
    get lost() {
      return lost;
    },
    get cause() {
      return loss;
    },
    close() {
      closed = true;
    }
  };
}

/** Combine a device-loss signal with caller cancellation without changing either source. */
export function combineGemma4V2Signals(device: AbortSignal, caller?: AbortSignal | null) {
  const controller = new AbortController();
  const onCaller = () => controller.abort(caller?.reason);
  const onDevice = () => controller.abort(caller?.aborted ? caller.reason : device.reason);
  if (caller?.aborted) onCaller();
  else if (device.aborted) onDevice();
  else {
    caller?.addEventListener('abort', onCaller, {once: true});
    device.addEventListener('abort', onDevice, {once: true});
  }
  return {
    signal: controller.signal,
    close() {
      caller?.removeEventListener('abort', onCaller);
      device.removeEventListener('abort', onDevice);
    }
  };
}

/** Bind per-request cancellation to both the caller and the owned WebGPU device. */
export function guardGemma4V2Runner(
  device: GPUDevice,
  lifecycle: ReturnType<typeof watchGemma4V2Device>,
  runner: Gemma4V2OwnedRunner
): Gemma4V2OwnedRunner {
  return {
    async run(input, caller) {
      caller?.throwIfAborted();
      if (lifecycle.lost)
        throw new WebDecisionError('INFERENCE_ERROR', 'The GPU device was lost.', {cause: lifecycle.cause});
      const tokens = input.stateIds.length + input.questionIds.length;
      assertGemma4V2CompiledContext(tokens);
      assertAttentionBufferLimit(device.limits, tokens);
      const combined = combineGemma4V2Signals(lifecycle.signal, caller);
      try {
        const result = await withGemma4V2GpuErrors(device, () => runner.run(input, combined.signal));
        caller?.throwIfAborted();
        if (lifecycle.lost)
          throw new WebDecisionError('INFERENCE_ERROR', 'The GPU device was lost.', {cause: lifecycle.cause});
        return result;
      } catch (error) {
        caller?.throwIfAborted();
        if (lifecycle.lost) throw new WebDecisionError('INFERENCE_ERROR', 'The GPU device was lost.', {cause: error});
        throw error;
      } finally {
        combined.close();
      }
    },
    releaseCachedMemory: runner.releaseCachedMemory.bind(runner),
    dispose: runner.dispose.bind(runner)
  };
}
