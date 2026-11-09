import {WebDecisionError} from './kev.js';
import {trySyncBeforeCleanup} from './runtime-lifecycle.js';

interface GpuTensorLike {
  copyFrom(value: unknown): GpuTensorLike;
  toArray(): unknown;
}

interface GpuUploadTensor extends GpuTensorLike {
  getDataPtr(): unknown;
  readonly byteOffset: number;
}

interface GpuWebGpuContext {
  copyRawBytesToBuffer(bytes: Uint8Array, destination: unknown, byteOffset: number, byteLength: number): unknown;
}

interface GpuRuntime {
  empty(shape: number[], dtype: string, device: unknown): GpuTensorLike;
  beginScope(): unknown;
  endScope(): unknown;
  readonly lib?: {readonly webGPUContext?: GpuWebGpuContext | null} | null;
}

type GpuInputRuntime = Pick<GpuRuntime, 'empty' | 'lib'>;

interface GpuDevice {
  sync(): unknown;
}

interface GpuErrorScopes {
  pushErrorScope(filter: 'out-of-memory' | 'validation'): unknown;
  popErrorScope(): PromiseLike<unknown>;
}

interface GpuLifecycle {
  readonly loss: unknown;
  readonly disposed: boolean;
}

interface GpuScopeSession {
  readonly runtime: GpuRuntime | null;
  readonly device: GpuDevice | null;
  readonly gpu: GpuErrorScopes | null;
  readonly lifecycle: GpuLifecycle | null;
  readonly uncaptured: readonly unknown[];
}

interface GpuDeviceCheckSession {
  readonly lifecycle: unknown;
  readonly uncaptured: readonly unknown[] | undefined;
}

interface GpuInputSession {
  readonly runtime: GpuInputRuntime | null;
  readonly device: unknown;
}

const noFailure = Symbol('no failure');

export function checkGpuDevice(session: GpuDeviceCheckSession): void {
  if ((session.lifecycle as GpuLifecycle).loss || (session.lifecycle as GpuLifecycle).disposed)
    throw new Error('The GPU device was lost.');
  if ((session.uncaptured as readonly unknown[]).length)
    throw new Error((session.uncaptured as readonly unknown[])[0] as string);
}

function closeRuntimeScope(
  session: Pick<GpuScopeSession, 'runtime' | 'lifecycle'>,
  open: boolean,
  failure: unknown
): unknown {
  if (!open || (session.lifecycle as GpuLifecycle).disposed) return failure;
  try {
    (session.runtime as GpuRuntime).endScope();
  } catch (cause) {
    return failure === noFailure ? cause : failure;
  }
  return failure;
}

async function closeScopes(
  session: Pick<GpuScopeSession, 'gpu' | 'runtime' | 'lifecycle'>,
  gpuScopes: number,
  open: boolean,
  failure: unknown
): Promise<unknown> {
  while (gpuScopes > 0) {
    try {
      const error = await (session.gpu as GpuErrorScopes).popErrorScope();
      if (error && failure === noFailure) failure = error;
    } catch (cause) {
      if (failure === noFailure) failure = cause;
    }
    gpuScopes--;
  }
  return closeRuntimeScope(session, open, failure);
}

/** Scope temporary tensors and surface captured WebGPU errors after synchronization. */
export async function withGpuScope<Value>(
  session: GpuScopeSession,
  message: string,
  operation: () => Value | PromiseLike<Value>
): Promise<Awaited<Value>> {
  let failure: unknown | typeof noFailure = noFailure,
    result!: Awaited<Value>,
    open = false,
    gpuScopes = 0;
  try {
    checkGpuDevice(session);
    (session.runtime as GpuRuntime).beginScope();
    open = true;
    (session.gpu as GpuErrorScopes).pushErrorScope('out-of-memory');
    gpuScopes++;
    (session.gpu as GpuErrorScopes).pushErrorScope('validation');
    gpuScopes++;
    result = await operation();
    checkGpuDevice(session);
  } catch (cause) {
    failure = cause;
    // Work may already have submitted a readback before throwing. Give the
    // device a chance to finish it while the temporary WASM scope is alive.
    // A rejected sync cannot prove all work settled; preserve the first error.
    if (open) await trySyncBeforeCleanup(session.device as GpuDevice, session.lifecycle as GpuLifecycle);
  } finally {
    failure = await closeScopes(session, gpuScopes, open, failure);
  }
  if (failure !== noFailure) throw new WebDecisionError('INFERENCE_ERROR', message, {cause: failure});
  if ((session.uncaptured as readonly unknown[]).length)
    throw new WebDecisionError('INFERENCE_ERROR', (session.uncaptured as readonly unknown[])[0] as string);
  // Device loss can arrive while asynchronous GPU error scopes are closing.
  if ((session.lifecycle as GpuLifecycle).loss || (session.lifecycle as GpuLifecycle).disposed)
    throw new WebDecisionError('INFERENCE_ERROR', message, {cause: new Error('The GPU device was lost.')});
  return result;
}

function halfTensor(session: GpuInputSession, shape: number[], bits: Uint16Array): GpuTensorLike {
  const tensor = (session.runtime as GpuInputRuntime).empty(shape, 'float16', session.device) as GpuUploadTensor;
  // WebGPU writes require four-byte alignment; pad odd FP16 element counts
  // into the runtime's physically rounded allocation, without changing shape.
  const raw = new Uint8Array(Math.ceil((bits.length * 2) / 4) * 4),
    view = new DataView(raw.buffer);
  for (let i = 0; i < bits.length; i++) view.setUint16(i * 2, bits[i]!, true);
  const context = (session.runtime as GpuInputRuntime).lib?.webGPUContext;
  if (!context || typeof context.copyRawBytesToBuffer !== 'function' || typeof tensor.byteOffset !== 'number')
    throw new Error('The pinned TVM aligned upload API is unavailable.');
  context.copyRawBytesToBuffer(raw, tensor.getDataPtr(), tensor.byteOffset, raw.length);
  return tensor;
}

/** Construct an unpadded text row and its causal full-attention mask. */
export function createCausalInputs(
  session: GpuInputSession,
  tokenIds: ArrayLike<number>
): {
  ids: GpuTensorLike;
  positions: GpuTensorLike;
  mask: GpuTensorLike;
  valid: GpuTensorLike;
} {
  const n = tokenIds.length;
  const ids = (session.runtime as GpuInputRuntime)
    .empty([1, n], 'int32', session.device)
    .copyFrom(Int32Array.from(tokenIds));
  const positions = (session.runtime as GpuInputRuntime)
    .empty([1, n], 'int32', session.device!)
    .copyFrom(Int32Array.from({length: n}, (_, i) => i));
  const maskBits = new Uint16Array(n * n);
  for (let row = 0; row < n; row++) for (let col = row + 1; col < n; col++) maskBits[row * n + col] = 0xfbff;
  const mask = halfTensor(session, [1, 1, n, n], maskBits);
  const valid = halfTensor(session, [1, n], new Uint16Array(n).fill(0x3c00));
  return {ids, positions, mask, valid};
}
