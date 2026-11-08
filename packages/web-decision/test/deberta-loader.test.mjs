import assert from 'node:assert/strict';
import * as nodeModule from 'node:module';
import test from 'node:test';

// Run the actual loader and lifecycle with a small fake asset source and runtime.
// Synchronous module hooks are available in Node 24, as in the package CI.
const supportsHooks = typeof nodeModule.registerHooks === 'function';
let loadExperimentalDeberta;
if (supportsHooks) {
  const loaderUrl = new URL('../src/deberta/loader.mjs', import.meta.url).href;
  const fakeSource = `
    const state = () => globalThis.__fakeDebertaLoader;
    export class Tokenizer {
      encode(text) { return {ids: [{'[CLS]': 1, '[SEP]': 2, '[STATE]': 128001, '[Q]': 128002, '[OPT]': 128003}[text]]}; }
    }
    export function createPolyfillWASI() { return {}; }
    export function instantiate() { state().started(); return state().pending; }
    export async function detectDebertaGPUDevice() { return {device: state().gpu, adapter: {features: new Set(['shader-f16'])}}; }
    export async function prepareModelSource(options) { return {signal: options.signal, response: {}, modelOptions: 'default'}; }
    export async function readDebertaManifestResponse() { return {}; }
    export function createModelAssetReader() { return {read() {}, isCacheComplete() { return false; }}; }
    export async function readAssetBatch() {
      const bytes = (value) => new TextEncoder().encode(JSON.stringify(value));
      return [bytes({}), bytes({}), bytes({vocab_size: 128100, open_jev: {max_len: 512, max_state_tokens: 256, temperature: 1.05}}), bytes({}), bytes({records: []}), bytes({}), bytes({})];
    }
    export function validateDebertaTensorCache() {}
    export function validateDebertaSplit() { return {encoderIndices: [], headIndices: []}; }
    export async function uploadModelShards() {}
    export function validateDebertaReadback() {}
    export function runWithPackedReadback() {}
    export class ExperimentalDebertaSession { constructor(options) { Object.assign(this, options); state().sessions++; } }
  `;
  const fakeModule = `data:text/javascript;base64,${Buffer.from(fakeSource).toString('base64')}`;
  const fakeImports = new Set([
    '@huggingface/tokenizers',
    './device.mjs',
    './model.mjs',
    './packed-readback.mjs',
    './readback-mode.mjs',
    './contract.mjs',
    '../asset-batch.js',
    '../model-source.js',
    '../model-assets.js'
  ]);
  const hooks = nodeModule.registerHooks({
    resolve(specifier, context, nextResolve) {
      if (
        context.parentURL === loaderUrl &&
        (fakeImports.has(specifier) ||
          specifier.endsWith('/.toolchains/apache-tvm-src/web/src/runtime.ts') ||
          specifier.endsWith('/.toolchains/apache-tvm-src/web/src/compact.ts'))
      ) {
        return {url: fakeModule, shortCircuit: true};
      }
      return nextResolve(specifier, context);
    }
  });
  try {
    ({loadExperimentalDeberta} = await import('../src/deberta/loader.mjs'));
  } finally {
    hooks.deregister();
  }
}

function setup(t) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, '__fakeDebertaLoader');
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, '__fakeDebertaLoader', previous);
    else delete globalThis.__fakeDebertaLoader;
  });
  let resolveRuntime, rejectRuntime, started, lose;
  const pending = new Promise((resolve, reject) => {
    resolveRuntime = resolve;
    rejectRuntime = reject;
  });
  const ready = new Promise((resolve) => {
    started = resolve;
  });
  const lost = new Promise((resolve) => {
    lose = resolve;
  });
  const state = {pending, started, sessions: 0, disposed: 0, destroyed: 0, initialized: 0, ended: 0};
  state.gpu = {
    lost,
    addEventListener() {},
    destroy() {
      state.destroyed++;
    }
  };
  const runtime = {
    initWebGPU() {
      state.initialized++;
    },
    webgpu() {
      return {};
    },
    beginScope() {},
    endScope() {
      state.ended++;
    },
    createVirtualMachine() {
      return {
        getFunction() {
          return () => {};
        }
      };
    },
    dispose() {
      state.disposed++;
    }
  };
  Object.defineProperty(globalThis, '__fakeDebertaLoader', {configurable: true, value: state});
  return {state, runtime, ready, lose, resolveRuntime, rejectRuntime};
}

test(
  'pending instantiation stops on caller abort or device loss and disposes a late runtime',
  {skip: !supportsHooks},
  async (t) => {
    for (const kind of ['abort', 'loss']) {
      await t.test(kind, async (t) => {
        const f = setup(t),
          controller = new AbortController(),
          reason = new Error('caller cancelled');
        const loading = loadExperimentalDeberta({signal: controller.signal});
        await f.ready;
        if (kind === 'abort') controller.abort(reason);
        else f.lose({message: 'fake loss'});
        await assert.rejects(loading, (error) =>
          kind === 'abort'
            ? error === reason
            : error.code === 'MODEL_LOAD_FAILED' && /device was lost/.test(error.message)
        );
        assert.equal(f.state.destroyed, 1);
        assert.equal(f.state.disposed, 0);
        assert.equal(f.state.initialized, 0);
        assert.equal(f.state.sessions, 0);
        f.resolveRuntime(f.runtime);
        await new Promise((resolve) => setImmediate(resolve));
        assert.equal(f.state.disposed, 1);
        assert.equal(f.state.ended, 0);
      });
    }
  }
);

test('a late instantiation rejection after cancellation is handled', {skip: !supportsHooks}, async (t) => {
  const f = setup(t),
    controller = new AbortController(),
    reason = new Error('caller cancelled');
  const loading = loadExperimentalDeberta({signal: controller.signal});
  await f.ready;
  controller.abort(reason);
  await assert.rejects(loading, (error) => error === reason);
  f.rejectRuntime(new Error('late rejection'));
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(f.state.destroyed, 1);
  assert.equal(f.state.disposed, 0);
});

test('successful instantiation transfers runtime ownership to the session', {skip: !supportsHooks}, async (t) => {
  const f = setup(t),
    controller = new AbortController();
  const loading = loadExperimentalDeberta({signal: controller.signal});
  await f.ready;
  f.resolveRuntime(f.runtime);
  const session = await loading;
  assert.equal(session.runtime, f.runtime);
  assert.equal(f.state.initialized, 1);
  assert.equal(f.state.sessions, 1);
  assert.equal(f.state.disposed, 0);
  assert.equal(f.state.destroyed, 0);
  controller.abort(new Error('after load'));
  await Promise.resolve();
  assert.equal(f.state.disposed, 0);
  session.runtime.dispose();
  assert.equal(f.state.disposed, 1);
});
