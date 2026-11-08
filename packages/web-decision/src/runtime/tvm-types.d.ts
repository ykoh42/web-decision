export interface DLDevice {
  readonly deviceType: number;
  readonly deviceId: number;
}

export interface Tensor {
  copyFromRawBytes(bytes: Uint8Array): this;
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
  createVirtualMachine(device: DLDevice): VirtualMachine;
  empty(shape: number[] | number, dtype: string, device: DLDevice): Tensor;
}
