import assert from 'node:assert/strict';
import test from 'node:test';
import {createSourceLoader} from '../src/strands/source-loader-internal.mjs';
import type {StrandsSourceOptions} from '../src/strands/source-loader-internal.mjs';

const LARGEST_BINDING = 1_017_118_720;
type TestFactoryDependencies = Parameters<typeof createSourceLoader<string>>[0];

function makeAdapter() {
  return {
    features: new Set(['shader-f16']),
    limits: {
      maxBufferSize: LARGEST_BINDING,
      maxStorageBufferBindingSize: LARGEST_BINDING,
      maxStorageBuffersPerShaderStage: 8
    }
  };
}

function makeBytes(events?: string[]) {
  const manifest = new Uint8Array([1]);
  const wasm = new Uint8Array([2]);
  const tokenizer = new Uint8Array([3]);
  const tokenizerConfig = new Uint8Array([4]);
  return {
    get manifest() {
      events?.push('bytes.manifest');
      return manifest;
    },
    get wasm() {
      events?.push('bytes.wasm');
      return wasm;
    },
    get tokenizer() {
      events?.push('bytes.tokenizer');
      return tokenizer;
    },
    get tokenizerConfig() {
      events?.push('bytes.tokenizerConfig');
      return tokenizerConfig;
    }
  };
}

test('preserves dependency, option getter, and source callback order', async () => {
  const events: string[] = [];
  const adapter = makeAdapter();
  let gpuGetterCalls = 0;
  const gpuApi = {
    get requestAdapter() {
      gpuGetterCalls++;
      events.push(`gpu.get:${gpuGetterCalls}`);
      return async () => {
        events.push('gpu.call');
        return adapter;
      };
    }
  };
  const signal = new AbortController().signal;
  const createTokenizer = (_data: unknown, _config: unknown) => ({kind: 'tokenizer'});
  const createRuntime = (_wasm: Uint8Array) => ({kind: 'runtime'});
  const onProgress = (_progress: {readonly loadedBytes: number}) => {};
  const readResult = {bytes: new Uint8Array([9]).buffer, cachedByHash: false};
  const sourceBytes = makeBytes(events);
  const transaction = {
    async read(base: URL, spec: {readonly path: string}, options: {readonly signal: AbortSignal | undefined}) {
      events.push('transaction.read');
      assert.equal(base.href, 'https://example.test/assets/');
      assert.equal(spec.path, 'weights.bin');
      assert.equal(options.signal, signal);
      return readResult;
    },
    async commit(options: {readonly signal: AbortSignal | undefined}) {
      events.push('transaction.commit');
      assert.equal(options.signal, signal);
      return true;
    }
  };
  const load = createSourceLoader<string>({
    get openSource() {
      events.push('dependency.openSource');
      return async (options: Parameters<TestFactoryDependencies['openSource']>[0]) => {
        events.push('openSource');
        assert.equal(options.base.href, 'https://example.test/assets/');
        assert.equal(options.cache, null);
        assert.equal(options.offline, false);
        assert.equal(options.signal, signal);
        const sourceResult = {
          get bytes() {
            events.push('sourceResult.bytes');
            return sourceBytes;
          },
          get transaction() {
            events.push('sourceResult.transaction');
            return transaction;
          }
        };
        return sourceResult;
      };
    },
    get openCache() {
      events.push('dependency.openCache');
      return async (
        enabled: Parameters<TestFactoryDependencies['openCache']>[0],
        actualSignal: Parameters<TestFactoryDependencies['openCache']>[1]
      ) => {
        events.push('openCache');
        assert.equal(enabled, true);
        assert.equal(actualSignal, signal);
        return null;
      };
    },
    get loadBytes() {
      events.push('dependency.loadBytes');
      return async (
        options: Parameters<TestFactoryDependencies['loadBytes']>[0],
        dependencies: Parameters<TestFactoryDependencies['loadBytes']>[1]
      ) => {
        events.push('loadBytes');
        assert.equal(options.base.href, 'https://example.test/assets/');
        assert.deepEqual(
          [
            options.manifestBytes?.[0],
            options.wasmBytes?.[0],
            options.tokenizerBytes?.[0],
            options.tokenizerConfigBytes?.[0]
          ],
          [1, 2, 3, 4]
        );
        assert.equal(options.createTokenizer, createTokenizer);
        assert.equal(options.createRuntime, createRuntime);
        assert.equal(options.gpuApi, gpuApi);
        assert.equal(options.signal, signal);
        assert.equal(options.onProgress, onProgress);
        assert.equal(options.cache, false);
        assert.equal(options.offline, false);
        assert.equal(dependencies.preparedAdapter, adapter);
        assert.equal(
          await dependencies.readRange(
            options.base,
            {path: 'weights.bin', totalBytes: 16, offset: 0, length: 4, sha256: 'a'.repeat(64)},
            {signal}
          ),
          readResult
        );
        await dependencies.finalize(signal);
        return 'loaded';
      };
    }
  });

  const options = {
    get assetsUrl() {
      events.push('options.assetsUrl');
      return 'https://example.test/assets';
    },
    get cache() {
      events.push('options.cache');
      return true;
    },
    get offline() {
      events.push('options.offline');
      return false;
    },
    get signal() {
      events.push('options.signal');
      return signal;
    },
    get createTokenizer() {
      events.push('options.createTokenizer');
      return createTokenizer;
    },
    get createRuntime() {
      events.push('options.createRuntime');
      return createRuntime;
    },
    get gpuApi() {
      events.push('options.gpuApi');
      return gpuApi;
    },
    get onProgress() {
      events.push('options.onProgress');
      return onProgress;
    },
    get maxPrefetchBytes() {
      events.push('options.maxPrefetchBytes');
      return 0;
    }
  } satisfies StrandsSourceOptions;

  assert.equal(await load(options), 'loaded');
  assert.deepEqual(events, [
    'dependency.openSource',
    'dependency.openCache',
    'dependency.loadBytes',
    'options.assetsUrl',
    'options.cache',
    'options.offline',
    'options.signal',
    'options.createTokenizer',
    'options.createRuntime',
    'options.gpuApi',
    'options.onProgress',
    'options.maxPrefetchBytes',
    'gpu.get:1',
    'gpu.get:2',
    'gpu.call',
    'openCache',
    'openSource',
    'sourceResult.bytes',
    'sourceResult.transaction',
    'bytes.manifest',
    'bytes.wasm',
    'bytes.tokenizer',
    'bytes.tokenizerConfig',
    'loadBytes',
    'transaction.read',
    'transaction.commit'
  ]);
  assert.equal(gpuGetterCalls, 2);
});

test('stops before opening the source when cache setup aborts the load', async () => {
  const controller = new AbortController();
  const reason = new Error('cancel during cache setup');
  const events: string[] = [];
  const gpuApi = {requestAdapter: async () => makeAdapter()};
  const load = createSourceLoader<string>({
    async openSource() {
      events.push('openSource');
      return {
        bytes: makeBytes(),
        transaction: {read: async () => ({bytes: new ArrayBuffer(0), cachedByHash: false}), commit: async () => true}
      };
    },
    async openCache(_enabled, signal) {
      events.push('openCache');
      assert.equal(signal, controller.signal);
      controller.abort(reason);
      return null;
    },
    async loadBytes() {
      events.push('loadBytes');
      return 'unreachable';
    }
  });

  await assert.rejects(
    load({assetsUrl: 'https://example.test/assets/', gpuApi, signal: controller.signal}),
    (error: unknown) => error === reason
  );
  assert.deepEqual(events, ['openCache']);
});

test('retains the source loader callback after openSource resolves during cancellation', async () => {
  const controller = new AbortController();
  const reason = new Error('cancel while opening source');
  const events: string[] = [];
  const gpuApi = {requestAdapter: async () => makeAdapter()};
  const load = createSourceLoader<string>({
    async openSource() {
      events.push('openSource');
      controller.abort(reason);
      return {
        bytes: makeBytes(),
        transaction: {
          read: async () => ({bytes: new ArrayBuffer(0), cachedByHash: false}),
          commit: async () => true
        }
      };
    },
    async openCache() {
      events.push('openCache');
      return null;
    },
    async loadBytes(options) {
      events.push('loadBytes');
      assert.equal(options.signal, controller.signal);
      assert.equal(options.signal.aborted, true);
      return 'called-after-source-abort';
    }
  });

  assert.equal(
    await load({assetsUrl: 'https://example.test/assets/', gpuApi, signal: controller.signal}),
    'called-after-source-abort'
  );
  assert.equal(controller.signal.reason, reason);
  assert.deepEqual(events, ['openCache', 'openSource', 'loadBytes']);
});
