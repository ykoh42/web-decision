// Keep the pinned upstream sources outside this project's strict typecheck.
// The Vite resolver removes the marker before bundling the actual TVM modules.
export {instantiate} from '../../../../.toolchains/apache-tvm-src/web/src/runtime.ts?web-decision-runtime';
export {createPolyfillWASI} from '../../../../.toolchains/apache-tvm-src/web/src/compact.ts?web-decision-runtime';
export type {DLDevice, Instance, PackedFunc, Tensor} from './tvm-types.js';
