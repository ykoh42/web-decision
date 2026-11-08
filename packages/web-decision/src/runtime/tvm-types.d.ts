export interface DLDevice {
  readonly deviceType: number;
  readonly deviceId: number;
  sync(): Promise<void>;
}

export interface Tensor {
  copyFrom(data: Int32Array | Tensor): this;
  copyFromRawBytes(bytes: Uint8Array): this;
  toRawBytes(): Uint8Array;
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
  getGlobalFunc(name: string): (...args: unknown[]) => unknown;
}
