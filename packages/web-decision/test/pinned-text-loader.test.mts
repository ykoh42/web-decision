import assert from 'node:assert/strict';
import {createHash, webcrypto} from 'node:crypto';
import * as nodeModule from 'node:module';
import test, {type TestContext} from 'node:test';
import {Tokenizer} from '@huggingface/tokenizers';
import type {Instance, Tensor} from '../src/runtime/tvm-types.js';
import {processPrefetchedShards} from '../src/shard-prefetch.js';

type ResolveContext = {readonly parentURL?: string | undefined; readonly conditions: readonly string[]};
type ResolveResult = {
  readonly url: string;
  readonly shortCircuit?: boolean | undefined;
  readonly format?: string | undefined;
};
type ResolveNext = (specifier: string, context: ResolveContext) => ResolveResult;
type RegisterHooks = (hooks: {
  readonly resolve: (specifier: string, context: ResolveContext, nextResolve: ResolveNext) => ResolveResult;
}) => {deregister(): void};
const hookModule = nodeModule as typeof nodeModule & {readonly registerHooks?: RegisterHooks};

// Run the actual shared loader and asset/cache path with a fake model
// descriptor, tokenizer, runtime and device. Node 20 has no synchronous module
// hooks; the repository's Node 24 CI executes these boundary tests.
const supportsHooks = typeof hookModule.registerHooks === 'function';
let loadPinnedTextModel!: typeof import('../src/pinned-text-loader.js').loadPinnedTextModel;
if (supportsHooks) {
  const loaderUrl = new URL('../src/pinned-text-loader.ts', import.meta.url).href;
  const runtimeBridgeUrl = new URL('../src/runtime/tvm.generated.mjs', import.meta.url).href;
  const fakeSource = `
    export class Tokenizer { constructor() {} }
    export function createPolyfillWASI() { return {}; }
    export async function instantiate() {
      globalThis.__fakePinnedLoader.onInstantiate?.();
      return globalThis.__fakePinnedLoader.runtime;
    }
  `;
  const fakeModule = `data:text/javascript;base64,${Buffer.from(fakeSource).toString('base64')}`;
  const hooks = hookModule.registerHooks!({
    resolve(specifier, context, nextResolve) {
      if (
        (context.parentURL === loaderUrl && specifier === '@huggingface/tokenizers') ||
        (context.parentURL === runtimeBridgeUrl &&
          (specifier.endsWith('/.toolchains/apache-tvm-src/web/src/runtime.ts') ||
            specifier.endsWith('/.toolchains/apache-tvm-src/web/src/compact.ts')))
      )
        return {url: fakeModule, shortCircuit: true};
      return nextResolve(specifier, context);
    }
  });
  try {
    ({loadPinnedTextModel} = await import('../src/pinned-text-loader.js'));
  } finally {
    hooks.deregister();
  }
}

interface TestState {
  readonly pushes: string[];
  pops: number;
  syncs: number;
  ended: number;
  disposed: number;
  destroyed: number;
  sessions: number;
}

interface FakeManifest {
  readonly parameterCount: number;
  readonly files: Record<string, {readonly bytes: number; readonly sha256: string}>;
}

interface FakeUploadArgs {
  readonly device: {sync(): Promise<void>};
  readonly read: (name: string, signal?: AbortSignal | undefined) => Promise<ArrayBuffer>;
  readonly signal: AbortSignal;
  readonly runtime: unknown;
  readonly index: unknown;
  readonly extras: unknown;
  readonly weights: (Tensor | undefined)[];
  readonly maxPrefetchBytes: number;
  readonly lifecycle: unknown;
  readonly uncaptured: string[];
}

interface FakeSession {
  readonly fake: true;
}

interface FakeModel {
  readonly name: string;
  readonly offlineMissingMessage: string;
  readManifest(response: Response): Promise<FakeManifest>;
  validateIndex(manifest: FakeManifest, value: unknown): unknown;
  prepare(data: unknown, tokenizer: InstanceType<typeof Tokenizer>, manifest: FakeManifest): unknown;
  requestDevice(): Promise<GPUDevice>;
  upload(args: FakeUploadArgs): Promise<unknown>;
  createSession(args: unknown): FakeSession;
}

interface Fixture {
  readonly state: TestState;
  readonly model: FakeModel;
  readonly runtime: Instance;
  readonly entries: Map<string, Response>;
  readonly prefix: string;
  lose(info: unknown): void;
}

interface SetupOptions {
  readonly upload?: ((args: FakeUploadArgs) => Promise<unknown>) | undefined;
  readonly pop?: ((index: number) => unknown | PromiseLike<unknown>) | undefined;
  readonly push?: ((name: string) => void) | undefined;
  readonly sync?: (() => unknown | PromiseLike<unknown>) | undefined;
}

declare global {
  var __fakePinnedLoader: {runtime: Instance | Promise<Instance>; onInstantiate?: (() => void) | undefined};
}

interface LoadError {
  readonly code?: unknown;
  readonly cause?: unknown;
}

type LoadOutcome = {readonly ok: true} | {readonly ok: false; readonly error: unknown};

function loadError(error: unknown): LoadError {
  return error as LoadError;
}

function setup(
  t: TestContext,
  {upload = async () => {}, pop = async () => null, push = () => {}, sync = async () => {}}: SetupOptions = {}
): Fixture {
  const globalNames = ['fetch', 'navigator', 'caches', 'crypto', '__fakePinnedLoader'] as const;
  type GlobalName = (typeof globalNames)[number];
  const previous = new Map<GlobalName, PropertyDescriptor | undefined>(
    globalNames.map((name) => [name, Object.getOwnPropertyDescriptor(globalThis, name)] as const)
  );
  t.after(() => {
    for (const [name, descriptor] of previous) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else delete (globalThis as unknown as Record<GlobalName, unknown>)[name];
    }
  });
  const state: TestState = {pushes: [], pops: 0, syncs: 0, ended: 0, disposed: 0, destroyed: 0, sessions: 0};
  const device = {
    async sync() {
      state.syncs++;
      return sync();
    }
  };
  const runtimeLike = {
    initWebGPU() {},
    webgpu() {
      return device;
    },
    beginScope() {},
    endScope() {
      state.ended++;
    },
    dispose() {
      state.disposed++;
    },
    createVirtualMachine() {
      return {
        getFunction() {
          return () => {};
        }
      };
    }
  };
  const runtime = runtimeLike as unknown as Instance;
  let lose!: (info: unknown) => void;
  const lost = new Promise<unknown>((resolve) => {
    lose = resolve;
  });
  const gpuLike = {
    lost,
    addEventListener() {},
    pushErrorScope(name: GPUErrorFilter) {
      state.pushes.push(name);
      push(name);
    },
    async popErrorScope() {
      state.pops++;
      return pop(state.pops);
    },
    destroy() {
      state.destroyed++;
    }
  };
  const gpu = gpuLike as unknown as GPUDevice;
  const texts = {
    'tokenizer.json': '{}',
    'tokenizer_config.json': '{}',
    'model.wasm': 'fake-wasm',
    'weights/index.json': '{}',
    'weights/fake-shard.bin': 'tiny-weight'
  };
  const prefix = 'https://fixture.invalid/model/',
    bytes = new Map<string, Uint8Array<ArrayBuffer>>(),
    files: FakeManifest['files'] = {};
  for (const [name, value] of Object.entries(texts)) {
    const data = new TextEncoder().encode(value);
    bytes.set(prefix + name, data);
    files[name] = {bytes: data.length, sha256: createHash('sha256').update(data).digest('hex')};
  }
  const manifest = {parameterCount: 1, files};
  bytes.set(prefix + 'manifest.json', new TextEncoder().encode(JSON.stringify(manifest)));
  const entries = new Map<string, Response>();
  const cache = {
    async match(key: RequestInfo) {
      return entries.get(String(key))?.clone();
    },
    async put(key: RequestInfo, response: Response) {
      entries.set(String(key), response.clone());
    }
  };
  Object.defineProperty(globalThis, '__fakePinnedLoader', {configurable: true, value: {runtime}});
  Object.defineProperty(globalThis, 'crypto', {configurable: true, value: webcrypto});
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: {
      gpu: {
        async requestAdapter() {
          return {features: new Set(['shader-f16'])};
        }
      }
    }
  });
  Object.defineProperty(globalThis, 'caches', {
    configurable: true,
    value: {
      async open() {
        return cache;
      }
    }
  });
  Object.defineProperty(globalThis, 'fetch', {
    configurable: true,
    value: async (url: RequestInfo) => {
      const data = bytes.get(String(url));
      if (!data) throw new Error('Unexpected fake asset URL');
      return new Response(data);
    }
  });
  const model: FakeModel = {
    name: 'fake',
    offlineMissingMessage: 'offline',
    async readManifest(response): Promise<FakeManifest> {
      return (await response.json()) as FakeManifest;
    },
    validateIndex(_manifest, value) {
      return value;
    },
    prepare(_data, _tokenizer, _manifest) {
      return {};
    },
    async requestDevice(): Promise<GPUDevice> {
      return gpu;
    },
    upload,
    createSession(): FakeSession {
      state.sessions++;
      return {fake: true};
    }
  };
  return {state, model, runtime, entries, prefix, lose};
}

function assertFailedLoad(
  fixture: Fixture,
  {pushes = ['out-of-memory', 'validation'], pops = 2}: {readonly pushes?: string[]; readonly pops?: number} = {}
): void {
  assert.deepEqual(fixture.state.pushes, pushes);
  assert.equal(fixture.state.pops, pops);
  assert.equal(fixture.state.ended, 1);
  assert.equal(fixture.state.disposed, 1);
  assert.equal(fixture.state.destroyed, 1);
  assert.equal(fixture.state.sessions, 0);
  assert.equal(fixture.entries.has(fixture.prefix + 'manifest.json'), false);
}

test(
  'runtime creation cancellation and device loss release the GPU and dispose a late runtime',
  {skip: !supportsHooks},
  async (t) => {
    for (const reason of ['abort', 'loss']) {
      await t.test(reason, async (t) => {
        const fixture = setup(t);
        let resolveRuntime!: (runtime: Instance) => void;
        let markStarted!: () => void;
        const runtimePending = new Promise<Instance>((resolve) => {
          resolveRuntime = resolve;
        });
        const started = new Promise<void>((resolve) => {
          markStarted = resolve;
        });
        globalThis.__fakePinnedLoader.runtime = runtimePending;
        globalThis.__fakePinnedLoader.onInstantiate = markStarted;
        const controller = new AbortController(),
          abortReason = new Error('caller cancelled');
        const loading = loadPinnedTextModel({assetsUrl: fixture.prefix, signal: controller.signal}, fixture.model);
        await started;
        if (reason === 'abort') controller.abort(abortReason);
        else fixture.lose({message: 'fake device loss'});
        await assert.rejects(loading, (error: unknown) =>
          reason === 'abort' ? error === abortReason : loadError(error).code === 'MODEL_LOAD_FAILED'
        );
        assert.equal(fixture.state.destroyed, 1);
        assert.equal(fixture.state.sessions, 0);
        assert.equal(fixture.state.disposed, 0);
        resolveRuntime(fixture.runtime);
        await new Promise((resolve) => setImmediate(resolve));
        assert.equal(fixture.state.disposed, 1);
        assert.equal(fixture.state.ended, 0);
      });
    }
  }
);

test('runtime creation rejection after cancellation is handled', {skip: !supportsHooks}, async (t) => {
  const fixture = setup(t);
  let rejectRuntime!: (error: unknown) => void;
  let markStarted!: () => void;
  globalThis.__fakePinnedLoader.runtime = new Promise<Instance>((_, reject) => {
    rejectRuntime = reject;
  });
  const started = new Promise<void>((resolve) => {
    markStarted = resolve;
  });
  globalThis.__fakePinnedLoader.onInstantiate = markStarted;
  const controller = new AbortController(),
    abortReason = new Error('caller cancelled');
  const loading = loadPinnedTextModel({assetsUrl: fixture.prefix, signal: controller.signal}, fixture.model);
  await started;
  controller.abort(abortReason);
  await assert.rejects(loading, (error: unknown) => error === abortReason);
  rejectRuntime(new Error('late runtime rejection'));
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(fixture.state.destroyed, 1);
  assert.equal(fixture.state.disposed, 0);
});

test(
  'upload failure waits for fallback sync before GPU scope and runtime cleanup',
  {skip: !supportsHooks},
  async (t) => {
    let finish!: () => void;
    let started!: () => void;
    const wait = new Promise<void>((resolve) => {
      finish = resolve;
    });
    const syncing = new Promise<void>((resolve) => {
      started = resolve;
    });
    const cause = new Error('upload failed after submitting work');
    const fixture = setup(t, {
      upload: async () => {
        throw cause;
      },
      sync: async () => {
        started();
        await wait;
      }
    });
    const loading = loadPinnedTextModel({assetsUrl: fixture.prefix}, fixture.model).then<LoadOutcome, LoadOutcome>(
      () => ({ok: true}),
      (error: unknown) => ({ok: false, error})
    );
    try {
      await Promise.race([syncing, loading]);
      assert.equal(fixture.state.syncs, 1);
      assert.equal(fixture.state.pops, 0);
      assert.equal(fixture.state.ended, 0);
      assert.equal(fixture.state.disposed, 0);
    } finally {
      finish();
    }
    const outcome = await loading;
    assert.equal(outcome.ok, false);
    if (!outcome.ok) assert.equal(loadError(outcome.error).cause, cause);
    assertFailedLoad(fixture);
  }
);

test('caller cancellation during load fallback keeps its reason after cleanup', {skip: !supportsHooks}, async (t) => {
  let finish!: () => void;
  let started!: () => void;
  const wait = new Promise<void>((resolve) => {
    finish = resolve;
  });
  const syncing = new Promise<void>((resolve) => {
    started = resolve;
  });
  const fixture = setup(t, {
    upload: async () => {
      throw Error('upload failed');
    },
    sync: async () => {
      started();
      await wait;
    }
  });
  const controller = new AbortController(),
    reason = new Error('caller cancelled');
  const loading = loadPinnedTextModel({assetsUrl: fixture.prefix, signal: controller.signal}, fixture.model).then<
    LoadOutcome,
    LoadOutcome
  >(
    () => ({ok: true}),
    (error: unknown) => ({ok: false, error})
  );
  try {
    await Promise.race([syncing, loading]);
    assert.equal(fixture.state.syncs, 1);
    controller.abort(reason);
    assert.equal(fixture.state.disposed, 0);
  } finally {
    finish();
  }
  const outcome = await loading;
  assert.equal(outcome.ok, false);
  if (!outcome.ok) assert.equal(outcome.error, reason);
  assertFailedLoad(fixture);
});

test('a rejected load fallback preserves a falsy original failure', {skip: !supportsHooks}, async (t) => {
  const fixture = setup(t, {
    upload: async () => {
      throw false;
    },
    sync: async () => {
      throw Error('fallback failed');
    }
  });
  await assert.rejects(
    loadPinnedTextModel({assetsUrl: fixture.prefix}, fixture.model),
    (error: unknown) =>
      loadError(error).code === 'MODEL_LOAD_FAILED' &&
      Object.hasOwn(error as object, 'cause') &&
      loadError(error).cause === false
  );
  assert.equal(fixture.state.syncs, 1);
  assertFailedLoad(fixture);
});

test('successful upload does not get an extra fallback sync', {skip: !supportsHooks}, async (t) => {
  const fixture = setup(t, {
    upload: async ({device}) => {
      await device.sync();
    }
  });
  assert.deepEqual(await loadPinnedTextModel({assetsUrl: fixture.prefix}, fixture.model), {fake: true});
  assert.equal(fixture.state.syncs, 1);
  assert.equal(fixture.state.disposed, 0);
});

test(
  'known device loss skips load fallback and avoids ending a lost runtime scope',
  {skip: !supportsHooks},
  async (t) => {
    let fixture!: Fixture;
    const cause = new Error('upload failed after device loss');
    fixture = setup(t, {
      upload: async () => {
        fixture.lose({reason: 'unknown'});
        await Promise.resolve();
        throw cause;
      }
    });
    await assert.rejects(
      loadPinnedTextModel({assetsUrl: fixture.prefix}, fixture.model),
      (error: unknown) => loadError(error).code === 'MODEL_LOAD_FAILED' && loadError(error).cause === cause
    );
    assert.equal(fixture.state.syncs, 0);
    assert.equal(fixture.state.ended, 0);
    assert.equal(fixture.state.disposed, 1);
    assert.equal(fixture.state.destroyed, 1);
    assert.equal(fixture.state.sessions, 0);
  }
);

test('falsy upload rejections fail the real loader and clean up', {skip: !supportsHooks}, async (t) => {
  for (const reason of [undefined, null, false, 0, '']) {
    await t.test(`upload rejects ${String(reason)}`, async (subtest) => {
      const fixture = setup(subtest, {
        upload: async () => {
          throw reason;
        }
      });
      await assert.rejects(
        loadPinnedTextModel({assetsUrl: fixture.prefix}, fixture.model),
        (error: unknown) =>
          loadError(error).code === 'MODEL_LOAD_FAILED' &&
          Object.hasOwn(error as object, 'cause') &&
          Object.is(loadError(error).cause, reason)
      );
      assertFailedLoad(fixture);
    });
  }
});

test('falsy GPU scope rejection fails after successful upload', {skip: !supportsHooks}, async (t) => {
  for (const reason of [undefined, null, false, 0, '']) {
    await t.test(`scope rejects ${String(reason)}`, async (subtest) => {
      const fixture = setup(subtest, {
        pop: async (index) => {
          if (index === 1) throw reason;
          return null;
        }
      });
      await assert.rejects(
        loadPinnedTextModel({assetsUrl: fixture.prefix}, fixture.model),
        (error: unknown) =>
          loadError(error).code === 'MODEL_LOAD_FAILED' &&
          Object.hasOwn(error as object, 'cause') &&
          Object.is(loadError(error).cause, reason)
      );
      assertFailedLoad(fixture);
    });
  }
});

test('falsy scope creation failures close only scopes that were opened', {skip: !supportsHooks}, async (t) => {
  for (const name of ['out-of-memory', 'validation']) {
    for (const reason of [undefined, null, false, 0, '']) {
      await t.test(`${name} throws ${String(reason)}`, async (subtest) => {
        const fixture = setup(subtest, {
          push: (opened) => {
            if (opened === name) throw reason;
          }
        });
        await assert.rejects(
          loadPinnedTextModel({assetsUrl: fixture.prefix}, fixture.model),
          (error: unknown) =>
            loadError(error).code === 'MODEL_LOAD_FAILED' &&
            Object.hasOwn(error as object, 'cause') &&
            Object.is(loadError(error).cause, reason)
        );
        assertFailedLoad(fixture, name === 'out-of-memory' ? {pushes: ['out-of-memory'], pops: 0} : {pops: 1});
      });
    }
  }
});

test(
  'a falsy progress callback failure while reading a shard prevents a successful session',
  {skip: !supportsHooks},
  async (t) => {
    let progress = 0;
    const fixture = setup(t, {upload: ({read, signal}) => read('weights/fake-shard.bin', signal)});
    await assert.rejects(
      loadPinnedTextModel(
        {
          assetsUrl: fixture.prefix,
          onProgress() {
            if (++progress === 5) throw false;
          }
        },
        fixture.model
      ),
      (error: unknown) => loadError(error).code === 'MODEL_LOAD_FAILED' && loadError(error).cause === false
    );
    assert.equal(progress, 5);
    assertFailedLoad(fixture);
  }
);

test('an undefined shard progress failure remains the loader cause', {skip: !supportsHooks}, async (t) => {
  let progress = 0;
  const fixture = setup(t, {
    upload: ({read, signal}) =>
      processPrefetchedShards(
        [{nbytes: 11}],
        (_shard, innerSignal) => read('weights/fake-shard.bin', innerSignal),
        async () => {},
        signal,
        0
      )
  });
  await assert.rejects(
    loadPinnedTextModel(
      {
        assetsUrl: fixture.prefix,
        onProgress() {
          if (++progress === 5) throw undefined;
        }
      },
      fixture.model
    ),
    (error: unknown) =>
      loadError(error).code === 'MODEL_LOAD_FAILED' &&
      Object.hasOwn(error as object, 'cause') &&
      loadError(error).cause === undefined
  );
  assert.equal(progress, 5);
  assertFailedLoad(fixture);
});

test('the first falsy upload failure wins over later GPU scope errors', {skip: !supportsHooks}, async (t) => {
  const later = new Error('later GPU scope error');
  const fixture = setup(t, {
    upload: async () => {
      throw null;
    },
    pop: async () => later
  });
  await assert.rejects(
    loadPinnedTextModel({assetsUrl: fixture.prefix}, fixture.model),
    (error: unknown) =>
      loadError(error).code === 'MODEL_LOAD_FAILED' &&
      Object.hasOwn(error as object, 'cause') &&
      loadError(error).cause === null
  );
  assertFailedLoad(fixture);
});

test(
  'a successful upload still creates a session and commits its complete cache',
  {skip: !supportsHooks},
  async (t) => {
    const fixture = setup(t, {upload: ({read, signal}) => read('weights/fake-shard.bin', signal)});
    assert.deepEqual(await loadPinnedTextModel({assetsUrl: fixture.prefix}, fixture.model), {fake: true});
    assert.equal(fixture.state.sessions, 1);
    assert.equal(fixture.entries.has(fixture.prefix + 'manifest.json'), true);
    assert.equal(fixture.state.pops, 2);
    assert.equal(fixture.state.disposed, 0);
    fixture.runtime.dispose();
  }
);

test('cancellation during upload retains the caller reason and cleans up', {skip: !supportsHooks}, async (t) => {
  let started!: () => void;
  const ready = new Promise<void>((resolve) => {
    started = resolve;
  });
  const fixture = setup(t, {
    upload: async ({signal}) => {
      started();
      await new Promise((_resolve, reject) =>
        signal.addEventListener('abort', () => reject(signal.reason), {once: true})
      );
    }
  });
  const controller = new AbortController();
  const pending = loadPinnedTextModel({assetsUrl: fixture.prefix, signal: controller.signal}, fixture.model);
  await ready;
  controller.abort(null);
  await assert.rejects(pending, (reason: unknown) => reason === null);
  assertFailedLoad(fixture);
});
