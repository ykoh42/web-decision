import assert from 'node:assert/strict';
import test from 'node:test';
import {setImmediate as nextTurn} from 'node:timers/promises';
import {loadFakeNariCandidate, loadNariCandidate, NARI_LIMITS} from '../src/nari/loader.mjs';
import {NARI_MODEL_ID, NARI_MIXED_MODEL_ID, validateNariOptions} from '../src/nari/options.mjs';
import {NARI_MIXED} from '../src/nari/variants.mjs';
import {WebDecisionError} from '../src/kev.js';

type FakeDevice = {lost: Promise<unknown>; destroy(): void};
type FakeAdapter = {
  features: Set<string>;
  limits: {maxStorageBuffersPerShaderStage: number; maxStorageBufferBindingSize: number; maxBufferSize: number};
  requestDevice(options?: unknown): Promise<FakeDevice>;
};
type FakeSource = {
  plan: {ordered: {source: string; file: {bytes: number}}[]; letterFile: {bytes: number}};
  readAsset(name: string, options?: {signal?: AbortSignal | undefined}): Promise<Uint8Array>;
  close(): void;
};
type FakeOwner = {phase: string; blocked: boolean; lost: boolean; uncaptured: unknown[]; retain(): void};
type FakeRunner = {run(): Promise<void>; releaseCachedMemory(): Promise<void>; dispose(): Promise<void>};
type FakeDependencies = {
  fake: true;
  gpu: {requestAdapter(): Promise<FakeAdapter>};
  ownership: () => {initialize(): FakeOwner; fail(runtime: unknown, gpu?: FakeDevice): void};
  openSource(base: URL, context: {signal?: AbortSignal | undefined; variant: string}): Promise<FakeSource>;
  tokenizer(): {tokenizer: {encode(): void}; controlIds: number[]};
  instantiate(): Promise<{dispose(): void}>;
  upload(
    owner: unknown,
    source: FakeSource,
    context: {expectedBindings: number; signal: AbortSignal}
  ): Promise<{parameters: unknown[]; letterRows: unknown}>;
  runner(context: {expectedBindings: number}): FakeRunner;
};
type FakeOptions = {model: string; assetsUrl: string; signal?: AbortSignal | undefined};

function fixture(): {
  events: string[];
  adapter: FakeAdapter;
  source: FakeSource;
  device: FakeDevice;
  deps: FakeDependencies;
  options: FakeOptions;
} {
  const events: string[] = [];
  const device = {
    lost: new Promise<unknown>(() => {}),
    destroy() {
      events.push('destroy');
    }
  };
  const adapter = {
    features: new Set(['shader-f16']),
    limits: {...NARI_LIMITS},
    async requestDevice() {
      events.push('device');
      return device;
    }
  };
  const source = {
    plan: {
      ordered: Array.from({length: 728}, (_, i) => ({source: 'planned-asset', file: {bytes: i ? 4 : 134184960}})),
      letterFile: {bytes: 4}
    },
    async readAsset(name: string) {
      events.push(name);
      return new Uint8Array([1]);
    },
    close() {
      events.push('close');
    }
  };
  const owner = {
    phase: 'prepared',
    blocked: false,
    lost: false,
    uncaptured: [] as unknown[],
    retain() {
      events.push('retain');
    }
  };
  const deps: FakeDependencies = {
    fake: true,
    gpu: {
      async requestAdapter() {
        events.push('adapter');
        return adapter;
      }
    },
    ownership: () => ({
      initialize() {
        events.push('initialize');
        return owner;
      },
      fail(_runtime, gpu) {
        events.push('fail');
        if (events.includes('initialize')) owner.retain();
        else gpu?.destroy();
      }
    }),
    async openSource() {
      events.push('source');
      return source;
    },
    tokenizer() {
      return {tokenizer: {encode() {}}, controlIds: [1]};
    },
    async instantiate() {
      events.push('instantiate');
      return {
        dispose() {
          events.push('late-dispose');
        }
      };
    },
    async upload() {
      events.push('upload');
      return {parameters: [], letterRows: {}};
    },
    runner() {
      events.push('runner');
      return {
        async run() {},
        async releaseCachedMemory() {},
        async dispose() {
          events.push('dispose');
        }
      };
    }
  };
  return {
    events,
    adapter,
    source,
    device,
    deps,
    options: {model: NARI_MODEL_ID, assetsUrl: 'https://example.test/assets/'}
  };
}

test('checks adapter before fetching and transfers initialized runner to session', async () => {
  const {events, deps, options} = fixture();
  const session = await loadFakeNariCandidate(options, deps);
  assert.ok(events.indexOf('adapter') < events.indexOf('source'));
  assert.ok(events.indexOf('config.json') < events.indexOf('device'));
  assert.ok(events.indexOf('upload') < events.indexOf('runner'));
  assert.equal(events.at(-1), 'close');
  await session.dispose();
  assert.equal(events.at(-1), 'dispose');
});

test('mixed route propagates its binding contract and rejects a full-Q4 source', async () => {
  const {deps, options, source} = fixture();
  options.model = NARI_MIXED_MODEL_ID;
  const open = deps.openSource;
  deps.openSource = async (base, context) => {
    assert.equal(context.variant, NARI_MIXED);
    return open(base, context);
  };
  await assert.rejects(loadFakeNariCandidate(options, deps), {code: 'INVALID_ASSET'});
  source.plan.ordered.length = 604;
  deps.upload = async (_owner, _source, context) => {
    assert.equal(context.expectedBindings, 606);
    return {parameters: [], letterRows: {}};
  };
  deps.runner = (context) => {
    assert.equal(context.expectedBindings, 606);
    return {async run() {}, async releaseCachedMemory() {}, async dispose() {}};
  };
  const session = await loadFakeNariCandidate(options, deps);
  await session.dispose();
});

test('real loader rejects a model ID different from the routed registration before GPU access', () => {
  for (const [model, expectedModel] of [
    [NARI_MODEL_ID, NARI_MIXED_MODEL_ID],
    [NARI_MIXED_MODEL_ID, NARI_MODEL_ID]
  ]) {
    assert.throws(
      () =>
        loadNariCandidate(
          {model, assetsUrl: 'https://example.test/'},
          {
            expectedModel,
            Tokenizer() {},
            instantiate() {},
            createPolyfillWASI() {},
            gpu: {
              requestAdapter() {
                throw Error('GPU must not be accessed');
              }
            } as unknown as GPU
          }
        ),
      {code: 'INVALID_INPUT'}
    );
  }
});

test('rejects unsupported options and insufficient adapters without asset fetches', async () => {
  const {events, deps, options, adapter} = fixture();
  for (const change of [{cache: true}, {offline: true}, {onProgress() {}}, {maxPrefetchBytes: 1}, {model: 'other'}])
    await assert.rejects(loadFakeNariCandidate({...options, ...change}, deps), {code: 'INVALID_INPUT'});
  assert.deepEqual(events, [] as string[]);
  adapter.limits.maxBufferSize--;
  await assert.rejects(loadFakeNariCandidate(options, deps), {code: 'UNSUPPORTED_ENVIRONMENT'});
  assert.equal(events.includes('source'), false);
  assert.throws(
    () =>
      validateNariOptions({
        ...options,
        get cache() {
          throw Error('getter');
        }
      }),
    {code: 'INVALID_INPUT'}
  );
});

test('abort while device request is pending destroys the late device', async () => {
  const {events, deps, options, adapter, device} = fixture();
  let resolveDevice!: (value: FakeDevice) => void, entered!: () => void;
  const started = new Promise<void>((resolve) => {
    entered = resolve;
  });
  adapter.requestDevice = () => {
    entered();
    return new Promise<FakeDevice>((resolve) => {
      resolveDevice = resolve;
    });
  };
  const controller = new AbortController();
  const job = loadFakeNariCandidate({...options, signal: controller.signal}, deps);
  await started;
  controller.abort();
  await assert.rejects(job, {name: 'AbortError'});
  assert.ok(events.includes('close'));
  assert.equal(events.includes('destroy'), false);
  resolveDevice(device);
  await nextTurn();
  assert.ok(events.includes('destroy'));
  assert.equal(events.includes('instantiate'), false);
});

test('abort during device request observes a late rejection', async () => {
  const {events, deps, options, adapter} = fixture();
  let rejectDevice!: (reason?: unknown) => void, entered!: () => void;
  const started = new Promise<void>((resolve) => {
    entered = resolve;
  });
  adapter.requestDevice = () => {
    entered();
    return new Promise<FakeDevice>((_resolve, reject) => {
      rejectDevice = reject;
    });
  };
  const controller = new AbortController();
  const reason = Error('cancel device request');
  const job = loadFakeNariCandidate({...options, signal: controller.signal}, deps);
  await started;
  controller.abort(reason);
  await assert.rejects(job, (error: unknown) => error === reason);
  assert.ok(events.includes('close'));
  rejectDevice(Error('late device failure'));
  await nextTurn();
  assert.equal(events.includes('instantiate'), false);
});

test('abort during small asset reads closes the source and observes late failures', async () => {
  const {events, deps, options, source} = fixture();
  const pending: Array<{resolve(value: Uint8Array): void; reject(reason?: unknown): void}> = [];
  source.readAsset = () =>
    new Promise<Uint8Array>((resolve, reject) => {
      pending.push({resolve, reject});
    });
  const controller = new AbortController();
  const reason = Error('cancel small assets');
  const job = loadFakeNariCandidate({...options, signal: controller.signal}, deps);
  while (pending.length < 4) await nextTurn();
  controller.abort(reason);
  await assert.rejects(job, (error: unknown) => error === reason);
  assert.ok(events.includes('close'));
  assert.ok(events.includes('fail'));
  assert.equal(events.includes('device'), false);
  pending[0]!.resolve(new Uint8Array([1]));
  for (const item of pending.slice(1)) item.reject(Error('late asset failure'));
  await nextTurn();
});

test('a late source is closed after cancellation', async () => {
  const {events, deps, options, source} = fixture();
  let resolveSource!: (value: FakeSource) => void, entered!: () => void;
  const started = new Promise<void>((resolve) => {
    entered = resolve;
  });
  deps.openSource = () => {
    entered();
    return new Promise<FakeSource>((resolve) => {
      resolveSource = resolve;
    });
  };
  const controller = new AbortController();
  const job = loadFakeNariCandidate({...options, signal: controller.signal}, deps);
  await started;
  controller.abort();
  await assert.rejects(job, {name: 'AbortError'});
  resolveSource(source);
  await nextTurn();
  assert.ok(events.includes('close'));
});

test('upload failure retains the initialized owner and closes source', async () => {
  const {events, deps, options} = fixture();
  deps.upload = async () => {
    throw Error('upload failed');
  };
  await assert.rejects(loadFakeNariCandidate(options, deps), {code: 'MODEL_LOAD_FAILED'});
  assert.ok(events.includes('retain'));
  assert.ok(events.includes('close'));
  assert.equal(events.includes('runner'), false);
});

test('GPU upload failures during load use the model-load error code and preserve the cause', async () => {
  const {events, deps, options} = fixture();
  const uploadError = new WebDecisionError('INFERENCE_ERROR', 'GPU lost during weight upload.');
  deps.upload = async () => {
    events.push('upload-failed');
    throw uploadError;
  };
  await assert.rejects(loadFakeNariCandidate(options, deps), (error: unknown) => {
    assert.equal((error as WebDecisionError).code, 'MODEL_LOAD_FAILED');
    assert.equal((error as Error & {cause?: unknown}).cause, uploadError);
    return true;
  });
  assert.ok(events.includes('retain'));
  assert.ok(events.includes('close'));
  assert.equal(events.includes('runner'), false);
});

test('device loss during weight upload preserves the load-time device error', async () => {
  const {events, deps, options, device} = fixture();
  let lose!: (value: unknown) => void;
  device.lost = new Promise<unknown>((resolve) => {
    lose = resolve;
  });
  let uploadStarted!: () => void;
  const started = new Promise<void>((resolve) => {
    uploadStarted = resolve;
  });
  deps.upload = (_owner, _source, {signal}) => {
    events.push('upload');
    uploadStarted();
    return new Promise<{parameters: unknown[]; letterRows: unknown}>((_resolve, reject) => {
      signal.addEventListener(
        'abort',
        () => reject(new WebDecisionError('INFERENCE_ERROR', 'Nari parameter upload failed.')),
        {once: true}
      );
    });
  };
  const job = loadFakeNariCandidate(options, deps);
  await started;
  const loss = {reason: 'destroyed', message: 'Device was destroyed.'};
  lose(loss);
  await assert.rejects(job, (error: unknown) => {
    assert.equal((error as WebDecisionError).code, 'MODEL_LOAD_FAILED');
    assert.equal((error as Error & {cause?: unknown}).message, 'The GPU device was lost while loading the model.');
    assert.equal((error as Error & {cause?: unknown}).cause, loss);
    return true;
  });
  assert.ok(events.includes('retain'));
  assert.ok(events.includes('close'));
  assert.equal(events.includes('runner'), false);
});

test('caller cancellation during weight upload preserves the caller reason', async () => {
  const {events, deps, options} = fixture();
  let uploadStarted!: () => void;
  const started = new Promise<void>((resolve) => {
    uploadStarted = resolve;
  });
  deps.upload = (_owner, _source, {signal}) => {
    events.push('upload');
    uploadStarted();
    return new Promise<{parameters: unknown[]; letterRows: unknown}>((_resolve, reject) => {
      signal.addEventListener('abort', () => reject(Error('upload interrupted')), {once: true});
    });
  };
  const controller = new AbortController();
  const reason = Error('cancel during Nari upload');
  const job = loadFakeNariCandidate({...options, signal: controller.signal}, deps);
  await started;
  controller.abort(reason);
  await assert.rejects(job, (error: unknown) => error === reason);
  assert.ok(events.includes('retain'));
  assert.ok(events.includes('close'));
  assert.equal(events.includes('runner'), false);
});

test('device loss during runtime creation disposes the late uninitialized runtime', async () => {
  const {events, deps, options, device} = fixture();
  let lose!: (value: unknown) => void, finish!: (value: {dispose(): void}) => void, entered!: () => void;
  device.lost = new Promise<unknown>((resolve) => {
    lose = resolve;
  });
  const started = new Promise<void>((resolve) => {
    entered = resolve;
  });
  deps.instantiate = () => {
    entered();
    return new Promise<{dispose(): void}>((resolve) => {
      finish = resolve;
    });
  };
  const job = loadFakeNariCandidate(options, deps);
  await started;
  lose({reason: 'unknown'});
  await assert.rejects(job, {code: 'MODEL_LOAD_FAILED'});
  finish({
    dispose() {
      events.push('late-dispose');
    }
  });
  await nextTurn();
  assert.ok(events.includes('late-dispose'));
  assert.ok(events.includes('close'));
  assert.equal(events.includes('initialize'), false);
});
