export interface DLDevice {
  readonly deviceType: number;
  readonly deviceId: number;
  sync(): Promise<void>;
}

export interface TVMLibraryProvider {
  readonly imports: Record<string, unknown>;
  start(instance: WebAssembly.Instance): void;
}

export type TVMImportObject = Record<string, unknown> | TVMLibraryProvider;

export interface Tensor {
  readonly shape: number[];
  readonly dtype: string;
  readonly device: DLDevice;
  copyFrom(
    data:
      | Tensor
      | number[]
      | Float32Array
      | Float64Array
      | Int32Array
      | Int8Array
      | Uint8Array
      | Uint32Array
      | Uint8ClampedArray
  ): this;
  copyFromRawBytes(bytes: Uint8Array): this;
  toRawBytes(): Uint8Array;
  toArray(): Float32Array | Float64Array | Int32Array | Int8Array | Uint8Array;
}

export type PackedFunc = (...args: unknown[]) => unknown;

export interface VirtualMachine {
  getFunction(name: string): PackedFunc;
}

export interface Instance {
  dispose(): void;
  initWebGPU(device: GPUDevice): void;
  webgpu(): DLDevice;
  beginScope(): void;
  endScope(): void;
  cpu(): DLDevice;
  createVirtualMachine(device: DLDevice): VirtualMachine;
  empty(shape: number[] | number, dtype: string, device: DLDevice): Tensor;
  getGlobalFunc(name: string): PackedFunc;
  systemLib(): {getFunction(name: string): PackedFunc};
}
