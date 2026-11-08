declare module '*/.toolchains/apache-tvm-src/web/src/runtime.ts?web-decision-runtime' {
  export function instantiate(
    bufferSource: ArrayBuffer,
    importObject: object
  ): Promise<import('./tvm-types.js').Instance>;
}
declare module '*/.toolchains/apache-tvm-src/web/src/compact.ts?web-decision-runtime' {
  export function createPolyfillWASI(): object;
}
