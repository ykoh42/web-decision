import {trackRuntime} from '../runtime-lifecycle.js';
import {waitForAbort} from '../abort.js';
import {guardStrandsDisposal, retainStrandsResources} from '../strands/disposal-guard.mjs';
import {withNariScope} from './scope.mjs';
import {createIncrementalSha256} from '../h2o/hasher.mjs';
import {uploadParameters} from './upload.mjs';
import {observeH2oTeardown, H2O_TEARDOWN_TIMEOUT_MS} from '../h2o/teardown.mjs';

const sha = async (bytes) =>
  [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
    .map((x) => x.toString(16).padStart(2, '0'))
    .join('');
const rotaryBase64 =
  'AACAPyuzGj8b+Lo+NfhhPneNCD5WCaU9P3ZHPXcR8Tw5rZE8UhAwPBXK1Dt9loA7DWkbO+3TuzriAWM6Ai4JOmDLpTnBYEg54izyOH5YkjhS3zA4QcTVN6stgTfEHxw3wbC8NsYMZDZKzwk2TI6mNVdMSTWdSfM0jgSTNESvMTQ=';
const ROTARY_HASH = 'ec4437c15ead01576c3e6c7412c11daba363d17cee8bfce5c5d1c54e2e09d2d0';
const retained = new Set();

export async function fixedRotary(signal) {
  signal?.throwIfAborted();
  const raw = Uint8Array.from(atob(rotaryBase64), (x) => x.charCodeAt(0));
  if (raw.length !== 128) throw new Error('rotary pin mismatch');
  const hash = await waitForAbort(() => sha(raw), signal);
  signal?.throwIfAborted();
  if (hash !== ROTARY_HASH) throw new Error('rotary pin mismatch');
  const data = new DataView(raw.buffer),
    array = Float32Array.from({length: 32}, (_, i) => data.getFloat32(i * 4, true));
  return {'body.rotary_emb.inv_freq': array, 'body.rotary_emb.original_inv_freq': Float32Array.from(array)};
}

/** Install the disposal guard before TVM's device-loss callback. */
function initializeRuntime(runtime, gpu) {
  let guard, owner;
  try {
    const teardownObserver = observeH2oTeardown(gpu);
    const lifecycle = trackRuntime(runtime, gpu);
    guard = guardStrandsDisposal(runtime, gpu, lifecycle);
    let held = false;
    const underlying = runtime.dispose;
    runtime.dispose = (...args) => {
      if (held) {
        guard.retain();
        return;
      }
      return Reflect.apply(underlying, runtime, args);
    };
    const uncaptured = [];
    if (typeof gpu.addEventListener !== 'function') throw new Error('missing device error listener');
    gpu.addEventListener('uncapturederror', (event) => {
      if (uncaptured.length < 10) uncaptured.push(event.error?.message || 'device error');
    });
    runtime.initWebGPU(gpu);
    runtime.beginScope();
    const device = runtime.webgpu(),
      fn = runtime.createVirtualMachine(device).getFunction('main');
    if (typeof fn !== 'function') throw new Error('missing graph entry');
    owner = {
      runtime,
      gpu,
      device,
      fn,
      lifecycle,
      uncaptured,
      disposalGuard: guard,
      teardownObserver,
      teardownTimeoutMs: H2O_TEARDOWN_TIMEOUT_MS,
      phase: 'prepared',
      resources: [],
      get blocked() {
        return guard.blocked;
      },
      get lost() {
        return lifecycle.loss || lifecycle.disposed;
      },
      retain() {
        guard.retain();
        retained.add(owner);
      },
      hold() {
        if (held || guard.blocked || lifecycle.loss || lifecycle.disposed) throw new Error('owner unavailable');
        held = true;
        return {
          release() {
            if (!held) throw new Error('upload lease not held');
            held = false;
          }
        };
      }
    };
    return owner;
  } catch (error) {
    if (guard) guard.retain();
    else retainStrandsResources(runtime, gpu);
    throw error;
  }
}

/** Own a device across the initialization handoff, including partial failure. */
export function runtimeOwnership() {
  let attempted = false,
    owner;
  return {
    initialize(runtime, gpu) {
      if (attempted) throw new Error('runtime initialization already attempted');
      attempted = true;
      owner = initializeRuntime(runtime, gpu);
      return owner;
    },
    fail(runtime, gpu) {
      if (owner) owner.retain();
      else if (!attempted) {
        if (runtime)
          try {
            runtime.dispose();
          } catch {}
        if (gpu)
          try {
            gpu.destroy();
          } catch {}
      }
      // initializeRuntime retains both resources on any initialization failure.
    }
  };
}

export function makeUploadGuard(guard, signal, clock = () => performance.now()) {
  if (typeof guard !== 'function' || typeof clock !== 'function') throw new Error('missing upload resource guard');
  let nextGuard = 0;
  return async () => {
    if (clock() < nextGuard) return;
    await guard('running');
    signal?.throwIfAborted();
    nextGuard = clock() + 5000;
  };
}

/** Uses actual TVM allocation size and settles every queued upload. */
export async function uploadFromSource(
  owner,
  source,
  {signal, HashClass, guard, diagnostic, expectedBindings = 730} = {}
) {
  const beforeUpload = makeUploadGuard(guard, signal);
  const context = owner.runtime.lib?.webGPUContext;
  if (typeof context?.copyRawBytesToBuffer !== 'function' || typeof context?.gpuBufferFromPtr !== 'function')
    throw new Error('unsupported TVM WebGPU raw buffer interface');
  const rotary = await fixedRotary(signal);
  return uploadParameters({
    owner,
    plan: source.plan,
    inventory: source.inventory,
    rotary,
    readRange: source.readRange.bind(source),
    beforeUpload,
    sha256: sha,
    makeHasher: () => createIncrementalSha256(HashClass),
    signal,
    diagnostic,
    expectedBindings,
    allocate(entry, needed) {
      const tensor = owner.runtime.empty(entry.shape, entry.dtype, owner.device);
      owner.resources.push(tensor);
      if (typeof tensor.getDataPtr !== 'function' || !Number.isSafeInteger(tensor.byteOffset) || tensor.byteOffset % 4)
        throw new Error('invalid TVM GPU tensor');
      const buffer = context.gpuBufferFromPtr(tensor.getDataPtr());
      if (!Number.isSafeInteger(buffer?.size) || buffer.size < tensor.byteOffset + needed)
        throw new Error('GPU allocation too small');
      return {tensor, capacityBytes: buffer.size - tensor.byteOffset};
    },
    async writeAndSettle(tensor, offset, bytes) {
      await withNariScope(owner, 'Nari parameter upload failed.', async (lease) => {
        context.copyRawBytesToBuffer(bytes, tensor.getDataPtr(), tensor.byteOffset + offset, bytes.byteLength);
        await lease.syncWithoutReadback();
      });
    }
  });
}
