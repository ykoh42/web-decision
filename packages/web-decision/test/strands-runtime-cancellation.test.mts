import test from 'node:test';
import assert from 'node:assert/strict';
import {loadStrandsFromBytes} from '../src/strands/loader-core.mjs';

function deferred<T = void>() {
  let resolve!: (value: T) => void, reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return {promise, resolve, reject};
}

function fixture() {
  const started = deferred(),
    pending = deferred<{dispose(): void; [key: string]: unknown}>(),
    lost = deferred<{reason: string}>();
  const events: string[] = [];
  const gpu = {
    lost: lost.promise,
    addEventListener() {},
    pushErrorScope() {},
    async popErrorScope() {
      return null;
    },
    destroy() {
      events.push('gpu.destroy');
    }
  };
  const adapter: {
    features: Set<string>;
    limits: {maxBufferSize: number; maxStorageBufferBindingSize: number; maxStorageBuffersPerShaderStage: number};
    requestDevice(): Promise<unknown>;
  } = {
    features: new Set(['shader-f16']),
    limits: {maxBufferSize: 1017118720, maxStorageBufferBindingSize: 1017118720, maxStorageBuffersPerShaderStage: 8},
    async requestDevice() {
      return gpu;
    }
  };
  const runtime: {dispose(): void; [key: string]: unknown} = {
    dispose() {
      events.push('runtime.dispose');
    }
  };
  const options = {
    base: 'https://example.test/model/',
    gpuApi: {requestAdapter: async () => adapter},
    createTokenizer: () => ({}),
    createRuntime: () => {
      started.resolve();
      return pending.promise;
    }
  };
  const dependencies: {
    preparedAdapter: typeof adapter;
    verifyAssets: () => Promise<unknown>;
    uploadWeights: (
      manifest: unknown,
      options: {allocate: (input: {shape: number[]; dtype: string}) => unknown}
    ) => Promise<unknown>;
    retainLateDevice?: (device: unknown) => unknown;
  } = {
    preparedAdapter: adapter,
    verifyAssets: async () => ({
      manifest: {},
      wasm: new Uint8Array(),
      tokenizerJson: new TextEncoder().encode('{}'),
      tokenizerConfig: new TextEncoder().encode('{}')
    }),
    uploadWeights: async () => {
      throw new Error('Unexpected upload');
    }
  };
  return {started, pending, lost, events, runtime, options, dependencies};
}

test('abort during device request returns promptly and disposes a late device', async () => {
  const f = fixture(),
    pending = deferred<{destroy(): void}>(),
    started = deferred(),
    controller = new AbortController();
  const reason = new Error('device request cancelled');
  f.dependencies.preparedAdapter.requestDevice = () => {
    started.resolve();
    return pending.promise;
  };
  const load = loadStrandsFromBytes(
    {...f.options, signal: controller.signal},
    f.dependencies as unknown as Parameters<typeof loadStrandsFromBytes>[1]
  );
  const outcome = load.then(
    () => 'resolved',
    (error) => error
  );
  await started.promise;
  controller.abort(reason);
  const observed = await Promise.race([outcome, new Promise((resolve) => setTimeout(() => resolve('pending'), 50))]);
  pending.resolve({
    destroy() {
      f.events.push('late.destroy');
    }
  });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(observed, reason);
  assert.deepEqual(f.events, ['late.destroy']);
});

test('late device rejection is handled and a failed late destroy is retained', async () => {
  for (const mode of ['reject', 'destroy']) {
    const f = fixture(),
      pending = deferred<{destroy(): void}>(),
      started = deferred(),
      controller = new AbortController();
    const reason = new Error('device request cancelled'),
      retained: unknown[] = [];
    f.dependencies.preparedAdapter.requestDevice = () => {
      started.resolve();
      return pending.promise;
    };
    f.dependencies.retainLateDevice = (device) => retained.push(device);
    const load = loadStrandsFromBytes(
      {...f.options, signal: controller.signal},
      f.dependencies as unknown as Parameters<typeof loadStrandsFromBytes>[1]
    );
    const outcome = load.then(
      () => 'resolved',
      (error) => error
    );
    await started.promise;
    controller.abort(reason);
    const observed = await Promise.race([outcome, new Promise((resolve) => setTimeout(() => resolve('pending'), 50))]);
    const device = {
      destroy() {
        throw new Error('cannot destroy');
      }
    };
    if (mode === 'reject') pending.reject(new Error('late request failure'));
    else pending.resolve(device);
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(observed, reason);
    assert.deepEqual(retained, mode === 'destroy' ? [device] : []);
  }
});

test('caller abort settles pending runtime creation and disposes a late runtime', {timeout: 1000}, async () => {
  const f = fixture(),
    controller = new AbortController(),
    reason = new Error('cancel');
  const load = loadStrandsFromBytes(
    {...f.options, signal: controller.signal},
    f.dependencies as unknown as Parameters<typeof loadStrandsFromBytes>[1]
  );
  await f.started.promise;
  controller.abort(reason);
  await assert.rejects(load, (error) => error === reason);
  assert.deepEqual(f.events, ['gpu.destroy']);
  f.pending.resolve(f.runtime);
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(f.events, ['gpu.destroy', 'runtime.dispose']);
});

test('device loss settles pending runtime creation and disposes a late runtime', {timeout: 1000}, async () => {
  const f = fixture();
  const load = loadStrandsFromBytes(f.options, f.dependencies as unknown as Parameters<typeof loadStrandsFromBytes>[1]);
  await f.started.promise;
  f.lost.resolve({reason: 'unknown'});
  await assert.rejects(
    load,
    (error) =>
      (error as {code: string}).code === 'MODEL_LOAD_FAILED' && /lost/.test((error as {message: string}).message)
  );
  assert.deepEqual(f.events, ['gpu.destroy']);
  f.pending.resolve(f.runtime);
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(f.events, ['gpu.destroy', 'runtime.dispose']);
});

test('late runtime rejection after abort is handled', {timeout: 1000}, async () => {
  const f = fixture(),
    controller = new AbortController(),
    reason = new Error('cancel');
  const load = loadStrandsFromBytes(
    {...f.options, signal: controller.signal},
    f.dependencies as unknown as Parameters<typeof loadStrandsFromBytes>[1]
  );
  await f.started.promise;
  controller.abort(reason);
  await assert.rejects(load, (error) => error === reason);
  f.pending.reject(new Error('late failure'));
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(f.events, ['gpu.destroy']);
});

test('successful runtime creation transfers ownership to the session', {timeout: 1000}, async () => {
  const f = fixture();
  Object.assign(f.runtime, {
    initWebGPU(gpu: unknown) {
      assert.ok(gpu);
    },
    beginScope() {
      f.events.push('runtime.begin');
    },
    endScope() {
      f.events.push('runtime.end');
    },
    webgpu() {
      return {};
    },
    createVirtualMachine() {
      return {getFunction: () => () => {}};
    }
  });
  f.dependencies.uploadWeights = async () => Array.from({length: 328}, () => ({}));
  const load = loadStrandsFromBytes(f.options, f.dependencies as unknown as Parameters<typeof loadStrandsFromBytes>[1]);
  await f.started.promise;
  f.pending.resolve(f.runtime);
  const session = await load;
  assert.deepEqual(f.events, ['runtime.begin']);
  await session.dispose();
  assert.deepEqual(f.events, ['runtime.begin', 'runtime.end', 'runtime.dispose']);
});

function setupLoadRuntime(f: ReturnType<typeof fixture>, sync: () => Promise<void>) {
  Object.assign(f.runtime, {
    initWebGPU() {
      f.events.push('runtime.init');
    },
    beginScope() {
      f.events.push('runtime.begin');
    },
    endScope() {
      f.events.push('runtime.end');
    },
    webgpu() {
      return {sync};
    },
    empty() {
      throw Error('allocation failed');
    }
  });
}

test('device loss during a pending load retains runtime until the realm is discarded', {timeout: 1000}, async () => {
  const f = fixture(),
    upload = deferred<unknown>(),
    entered = deferred();
  setupLoadRuntime(f, async () => {});
  f.dependencies.uploadWeights = async () => {
    entered.resolve();
    return upload.promise;
  };
  const load = loadStrandsFromBytes(f.options, f.dependencies as unknown as Parameters<typeof loadStrandsFromBytes>[1]);
  await f.started.promise;
  f.pending.resolve(f.runtime);
  await entered.promise;
  f.lost.resolve({reason: 'fixture-loss'});
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(f.events, ['runtime.init', 'runtime.begin']);
  upload.reject(Error('upload stopped'));
  await assert.rejects(load, {code: 'MODEL_LOAD_FAILED'});
  assert.deepEqual(f.events, ['runtime.init', 'runtime.begin']);
});

test('unconfirmed load fallback retains runtime while a confirmed fallback cleans up', {timeout: 1000}, async () => {
  for (const rejectSync of [true, false]) {
    const f = fixture();
    setupLoadRuntime(f, async () => {
      if (rejectSync) throw Error('fallback failed');
    });
    f.dependencies.uploadWeights = async (_manifest, {allocate}) => allocate({shape: [1], dtype: 'float32'});
    const load = loadStrandsFromBytes(
      f.options,
      f.dependencies as unknown as Parameters<typeof loadStrandsFromBytes>[1]
    );
    await f.started.promise;
    f.pending.resolve(f.runtime);
    await assert.rejects(load, {code: 'MODEL_LOAD_FAILED'});
    assert.deepEqual(
      f.events,
      rejectSync
        ? ['runtime.init', 'runtime.begin']
        : ['runtime.init', 'runtime.begin', 'runtime.end', 'runtime.dispose', 'gpu.destroy']
    );
  }
});

test('an exception registering the tracker or guard retains the runtime and device', {timeout: 1000}, async () => {
  for (const failingRegistration of [2, 3]) {
    const f = fixture();
    let registrations = 0;
    const lost = new Proxy(new Promise(() => {}), {
      get(target, key) {
        if (key !== 'then') return Reflect.get(target, key);
        return (...args: unknown[]) => {
          if (++registrations === failingRegistration) throw Error('cannot observe loss');
          return Reflect.apply(Promise.prototype.then, target, args);
        };
      }
    });
    f.dependencies.preparedAdapter = {
      features: new Set(['shader-f16']),
      limits: {
        maxBufferSize: 1017118720,
        maxStorageBufferBindingSize: 1017118720,
        maxStorageBuffersPerShaderStage: 8
      },
      async requestDevice() {
        return {
          lost,
          addEventListener() {},
          destroy() {
            f.events.push('gpu.destroy');
          }
        };
      }
    };
    const load = loadStrandsFromBytes(
      f.options,
      f.dependencies as unknown as Parameters<typeof loadStrandsFromBytes>[1]
    );
    await f.started.promise;
    f.pending.resolve(f.runtime);
    await assert.rejects(load, {code: 'MODEL_LOAD_FAILED'});
    assert.deepEqual(f.events, []);
  }
});

test('partial WebGPU initialization or scope setup retains the runtime and device', {timeout: 1000}, async () => {
  for (const failurePhase of ['initWebGPU', 'beginScope']) {
    const f = fixture();
    setupLoadRuntime(f, async () => {});
    f.runtime[failurePhase] = () => {
      throw Error('partial setup');
    };
    const load = loadStrandsFromBytes(
      f.options,
      f.dependencies as unknown as Parameters<typeof loadStrandsFromBytes>[1]
    );
    await f.started.promise;
    f.pending.resolve(f.runtime);
    await assert.rejects(load, {code: 'MODEL_LOAD_FAILED'});
    assert.equal(f.events.includes('runtime.dispose'), false);
    assert.equal(f.events.includes('gpu.destroy'), false);
  }
});
