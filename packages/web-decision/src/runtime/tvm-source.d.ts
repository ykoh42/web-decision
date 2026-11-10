declare module '*/.toolchains/apache-tvm-src/web/src/runtime.ts?web-decision-runtime' {
  export function instantiate(
    bufferSource: ArrayBuffer,
    importObject: object
  ): Promise<import('./tvm-types.js').Instance>;
}
declare module '*/.toolchains/apache-tvm-src/web/src/compact.ts?web-decision-runtime' {
  export function createPolyfillWASI(): object;
}
declare module '*/.toolchains/apache-tvm-src/web/src/webgpu.ts?web-decision-runtime' {
  export interface GPUDeviceDetectOutput {
    adapter: GPUAdapter;
    adapterInfo: GPUAdapterInfo;
    device: GPUDevice;
  }
  export function detectGPUDevice(
    powerPreference?: 'low-power' | 'high-performance'
  ): Promise<GPUDeviceDetectOutput | undefined>;
}
