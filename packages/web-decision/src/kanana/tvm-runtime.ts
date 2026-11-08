export type WasmByteInput = ArrayBuffer | ArrayBufferView;

export interface TvmRuntimeLike<VirtualMachine = unknown> {
  initWebGPU(device: object): void;
  beginScope(): void;
  endScope(): void;
  dispose(): void;
  webgpu(): object;
  createVirtualMachine(device: object): VirtualMachine;
}

export interface KananaTvmSetupOptions<ImportObject, Runtime extends TvmRuntimeLike> {
  instantiate(
    bytes: ArrayBuffer,
    importObject: ImportObject,
    logger: (message: string) => void
  ): Runtime | PromiseLike<Runtime>;
  createPolyfillWASI(): ImportObject;
  device: object;
  bytes: WasmByteInput;
}

export interface KananaTvmSetup<Runtime extends TvmRuntimeLike> {
  runtime: Runtime;
  vm: ReturnType<Runtime['createVirtualMachine']>;
  dispose(): void;
}

function wasmArrayBuffer(bytes: WasmByteInput): ArrayBuffer {
  if (bytes instanceof ArrayBuffer) return bytes;
  if (!ArrayBuffer.isView(bytes))
    throw new TypeError('Kanana TVM WASM bytes must be an ArrayBuffer or an ArrayBuffer view.');

  if (bytes.buffer instanceof ArrayBuffer && bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength)
    return bytes.buffer;

  return new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength).slice().buffer as ArrayBuffer;
}

function releaseRuntime(runtime: TvmRuntimeLike, scopeOpen: boolean): void {
  const errors: unknown[] = [];
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
export async function setupKananaTvmRuntime<ImportObject, Runtime extends TvmRuntimeLike>({
  instantiate,
  createPolyfillWASI,
  device,
  bytes
}: KananaTvmSetupOptions<ImportObject, Runtime>): Promise<KananaTvmSetup<Runtime>> {
  if (typeof instantiate !== 'function') throw new TypeError('TVM instantiate() is required.');
  if (typeof createPolyfillWASI !== 'function') throw new TypeError('TVM createPolyfillWASI() is required.');
  if (device === null || typeof device !== 'object') throw new TypeError('A WebGPU device is required.');
  const wasmBytes = wasmArrayBuffer(bytes);

  let runtime: Runtime | undefined;
  let scopeOpen = false;
  try {
    const initializedRuntime = await instantiate(wasmBytes, createPolyfillWASI(), () => {});
    runtime = initializedRuntime;
    initializedRuntime.initWebGPU(device);
    initializedRuntime.beginScope();
    scopeOpen = true;
    const vm = initializedRuntime.createVirtualMachine(initializedRuntime.webgpu()) as ReturnType<
      Runtime['createVirtualMachine']
    >;
    let disposed = false;

    return {
      runtime: initializedRuntime,
      vm,
      dispose() {
        if (disposed) return;
        disposed = true;
        const shouldEndScope = scopeOpen;
        scopeOpen = false;
        releaseRuntime(initializedRuntime, shouldEndScope);
      }
    };
  } catch (error) {
    if (!runtime) throw error;
    try {
      releaseRuntime(runtime, scopeOpen);
    } catch (cleanupError) {
      const failure = Object.assign(new Error('Kanana TVM setup and cleanup both failed.', {cause: error}), {
        cleanupError
      });
      throw failure;
    }
    throw error;
  }
}
