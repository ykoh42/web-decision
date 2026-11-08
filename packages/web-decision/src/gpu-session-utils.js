import {WebDecisionError} from './kev.js';
import {trySyncBeforeCleanup} from './runtime-lifecycle.js';

const noFailure = Symbol('no failure');

export function checkGpuDevice(session) {
  if (session.lifecycle.loss || session.lifecycle.disposed) throw new Error('The GPU device was lost.');
  if (session.uncaptured.length) throw new Error(session.uncaptured[0]);
}

function closeRuntimeScope(session, open, failure) {
  if (!open || session.lifecycle.disposed) return failure;
  try {
    session.runtime.endScope();
  } catch (cause) {
    return failure === noFailure ? cause : failure;
  }
  return failure;
}

async function closeScopes(session, gpuScopes, open, failure) {
  while (gpuScopes > 0) {
    try {
      const error = await session.gpu.popErrorScope();
      if (error && failure === noFailure) failure = error;
    } catch (cause) {
      if (failure === noFailure) failure = cause;
    }
    gpuScopes--;
  }
  return closeRuntimeScope(session, open, failure);
}

/** Scope temporary tensors and surface captured WebGPU errors after synchronization. */
export async function withGpuScope(session, message, operation) {
  let failure = noFailure,
    result,
    open = false,
    gpuScopes = 0;
  try {
    checkGpuDevice(session);
    session.runtime.beginScope();
    open = true;
    session.gpu.pushErrorScope('out-of-memory');
    gpuScopes++;
    session.gpu.pushErrorScope('validation');
    gpuScopes++;
    result = await operation();
    checkGpuDevice(session);
  } catch (cause) {
    failure = cause;
    // Work may already have submitted a readback before throwing. Give the
    // device a chance to finish it while the temporary WASM scope is alive.
    // A rejected sync cannot prove all work settled; preserve the first error.
    if (open) await trySyncBeforeCleanup(session.device, session.lifecycle);
  } finally {
    failure = await closeScopes(session, gpuScopes, open, failure);
  }
  if (failure !== noFailure) throw new WebDecisionError('INFERENCE_ERROR', message, {cause: failure});
  if (session.uncaptured.length) throw new WebDecisionError('INFERENCE_ERROR', session.uncaptured[0]);
  // Device loss can arrive while asynchronous GPU error scopes are closing.
  if (session.lifecycle.loss || session.lifecycle.disposed)
    throw new WebDecisionError('INFERENCE_ERROR', message, {cause: new Error('The GPU device was lost.')});
  return result;
}

function halfTensor(session, shape, bits) {
  const tensor = session.runtime.empty(shape, 'float16', session.device);
  // WebGPU writes require four-byte alignment; pad odd FP16 element counts
  // into the runtime's physically rounded allocation, without changing shape.
  const raw = new Uint8Array(Math.ceil((bits.length * 2) / 4) * 4),
    view = new DataView(raw.buffer);
  for (let i = 0; i < bits.length; i++) view.setUint16(i * 2, bits[i], true);
  const context = session.runtime.lib?.webGPUContext;
  if (!context || typeof context.copyRawBytesToBuffer !== 'function' || typeof tensor.byteOffset !== 'number')
    throw new Error('The pinned TVM aligned upload API is unavailable.');
  context.copyRawBytesToBuffer(raw, tensor.getDataPtr(), tensor.byteOffset, raw.length);
  return tensor;
}

/** Construct an unpadded text row and its causal full-attention mask. */
export function createCausalInputs(session, tokenIds) {
  const n = tokenIds.length;
  const ids = session.runtime.empty([1, n], 'int32', session.device).copyFrom(Int32Array.from(tokenIds));
  const positions = session.runtime
    .empty([1, n], 'int32', session.device)
    .copyFrom(Int32Array.from({length: n}, (_, i) => i));
  const maskBits = new Uint16Array(n * n);
  for (let row = 0; row < n; row++) for (let col = row + 1; col < n; col++) maskBits[row * n + col] = 0xfbff;
  const mask = halfTensor(session, [1, 1, n, n], maskBits);
  const valid = halfTensor(session, [1, n], new Uint16Array(n).fill(0x3c00));
  return {ids, positions, mask, valid};
}
