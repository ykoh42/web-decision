import {setTimeout as delay} from 'node:timers/promises';
import {withStrandsScope} from '../src/strands/scope.mjs';

const mode = process.argv[2] || 'preexisting';
let rejectUnexpected!: (reason?: unknown) => void;
let resolveOriginal!: () => void;
let unhandled = false;
const unexpected = new Promise((_resolve, reject) => {
  rejectUnexpected = reject;
});
const original = new Promise<void>((resolve) => {
  resolveOriginal = resolve;
});
const context = {pendingGPUToCPUCopy: mode === 'preexisting' ? unexpected : null};
if (mode === 'nonwritable') Object.defineProperty(context, 'pendingGPUToCPUCopy', {value: unexpected, writable: false});
const runner = {
  runtime: {lib: {webGPUContext: context}, beginScope() {}, endScope() {}},
  gpu: {
    pushErrorScope() {},
    async popErrorScope() {
      return null;
    }
  },
  device: {async sync() {}},
  lifecycle: {loss: null, disposed: false},
  uncaptured: []
};
process.on('unhandledRejection', () => {
  unhandled = true;
});
const work = withStrandsScope(runner, 'fixture', async (lease) => {
  if (mode === 'graph') context.pendingGPUToCPUCopy = unexpected;
  lease.prepareReadback();
  if (mode === 'replacement') {
    context.pendingGPUToCPUCopy = original;
    lease.captureReadback();
    context.pendingGPUToCPUCopy = unexpected;
    await lease.sync();
  }
});
if (mode === 'replacement') {
  // The replaced promise rejects while the captured readback is still pending.
  queueMicrotask(() => rejectUnexpected(Error('unexpected pending readback')));
  setTimeout(resolveOriginal, 10);
}
try {
  await work;
} catch {}
if (mode !== 'replacement') rejectUnexpected(Error('unexpected pending readback'));
await delay(30);
if (!(runner as typeof runner & {poisoned?: unknown}).poisoned || unhandled) process.exitCode = 1;
