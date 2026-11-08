function wasmArrayBuffer(bytes) {
  if (bytes instanceof ArrayBuffer) return bytes;
  if (!ArrayBuffer.isView(bytes))
    throw new TypeError('Kanana TVM WASM bytes must be an ArrayBuffer or an ArrayBuffer view.');

  if (bytes.buffer instanceof ArrayBuffer && bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength)
    return bytes.buffer;

  return new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength).slice().buffer;
}

function releaseRuntime(runtime, scopeOpen) {
  const errors = [];
  if (scopeOpen) {
    try {
      runtime.endScope();
    } catch (error) {
      errors.push(error);
    }
  }

  try {
    runtime.dispose();
  } catch (error) {
    errors.push(error);
  }

  if (errors.length === 1) throw errors[0];
  if (errors.length > 1) throw new AggregateError(errors, 'Kanana TVM runtime cleanup failed.', {cause: errors[0]});
}

/**
 * Initialize a TVM WebGPU runtime and VM without loading model parameters or invoking model functions.
 * The caller owns the returned setup and must call dispose when finished.
 */
export async function setupKananaTvmRuntime({instantiate, createPolyfillWASI, device, bytes}) {
  if (typeof instantiate !== 'function') throw new TypeError('TVM instantiate() is required.');
  if (typeof createPolyfillWASI !== 'function') throw new TypeError('TVM createPolyfillWASI() is required.');
  if (device === null || typeof device !== 'object') throw new TypeError('A WebGPU device is required.');
  const wasmBytes = wasmArrayBuffer(bytes);

  let runtime;
  let scopeOpen = false;
  try {
    runtime = await instantiate(wasmBytes, createPolyfillWASI(), () => {});
    runtime.initWebGPU(device);
    runtime.beginScope();
    scopeOpen = true;
    const vm = runtime.createVirtualMachine(runtime.webgpu());
    let disposed = false;

    return {
      runtime,
      vm,
      dispose() {
        if (disposed) return;
        disposed = true;
        const shouldEndScope = scopeOpen;
        scopeOpen = false;
        releaseRuntime(runtime, shouldEndScope);
      }
    };
  } catch (error) {
    if (!runtime) throw error;
    try {
      releaseRuntime(runtime, scopeOpen);
    } catch (cleanupError) {
      const failure = new Error('Kanana TVM setup and cleanup both failed.', {cause: error});
      failure.cleanupError = cleanupError;
      throw failure;
    }
    throw error;
  }
}
