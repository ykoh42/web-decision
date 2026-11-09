import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {runInNewContext} from 'node:vm';
import {readFile} from 'node:fs/promises';
import {initializeH2oRuntime, attachH2oRunner} from '../src/h2o/owner.mjs';
import {H2oTvmRunner} from '../src/h2o/runner.mjs';
import {createH2oSession} from '../src/h2o/session.mjs';
import {uploadFakeH2oParameters, uploadH2oParameters} from '../src/h2o/upload.mjs';
import type {H2oGpu, H2oRuntime, H2oRuntimeOwner, H2oTensor} from '../src/h2o/owner.mjs';
import type {H2oOpenedSource} from '../src/h2o/source.mjs';

type Deferred<Value> = {
  readonly promise: Promise<Value>;
  resolve(value?: Value): void;
  reject(reason: unknown): void;
};

type H2oIntInput = {dtype: 'int32'; shape: number[]; data: Int32Array};
type H2oHalfInput = {dtype: 'float16'; shape: number[]; data: Uint16Array};
type H2oInput = H2oIntInput | H2oHalfInput;
type H2oInputTuple = [H2oIntInput, H2oIntInput, H2oHalfInput, H2oHalfInput, H2oIntInput];
type H2oFileParameter = {
  kind: 'file';
  path: string;
  bytes: number;
  sha256: string;
  dtype: string;
  shape: number[];
};
type H2oRotaryParameter = {
  kind: 'rotary';
  target: string;
  dtype: string;
  shape: number[];
  values: number[];
};
type H2oParameter = H2oFileParameter | H2oRotaryParameter;
type H2oRangeChunk = {offset: number; length: number; sha256: string};
type H2oInventory = Record<string, H2oRangeChunk[] | undefined>;
type H2oRangeReader = H2oOpenedSource['readRange'];
type H2oRangeResponse = Awaited<ReturnType<H2oRangeReader>>;
type H2oTestContext = {
  pendingGPUToCPUCopy: Promise<void> | null;
  copyRawBytesToBuffer(bytes: Uint8Array, pointer: unknown, offset: number, length: number): void;
};
type H2oTestTensor = H2oTensor & {
  readonly dtype: string;
  readonly shape: number[];
  readonly byteOffset: number;
  value?: Float32Array | number[];
  copyFrom(value: unknown): H2oTestTensor;
  toArray(): unknown;
};
type H2oTestDevice = {sync(): Promise<void>};
type H2oTestRuntime = Omit<H2oRuntime, 'lib' | 'webgpu' | 'empty'> & {
  readonly lib: {readonly webGPUContext: H2oTestContext};
  webgpu(): H2oTestDevice;
  cpu(): string;
  getGlobalFunc(name: string): () => void;
  empty(shape: number[], dtype: string, device: unknown): H2oTestTensor;
  createVirtualMachine(device: unknown): {getFunction(name: string): (...args: unknown[]) => unknown};
};
type H2oTestGpu = H2oGpu & {
  addEventListener(
    type: 'uncapturederror',
    listener: (event: {error?: {message?: unknown} | null | undefined}) => void
  ): void;
  pushErrorScope(): void;
  popErrorScope(): Promise<null>;
};
type H2oTestFixture = {
  gpu: H2oTestGpu;
  device: H2oTestDevice;
  runtime: H2oTestRuntime;
  context: H2oTestContext;
  lost: Deferred<{reason: string}>;
  events: string[];
  writes: Array<{bytes: number[]; pointer: unknown; offset: number; length: number}>;
  graphCalls: unknown[][];
};

function deferred<Value = void>(): Deferred<Value> {
  let resolvePromise!: (value: Value | PromiseLike<Value>) => void;
  let rejectPromise!: (reason?: unknown) => void;
  const promise = new Promise<Value>((a, b) => {
    resolvePromise = a;
    rejectPromise = b;
  });
  return {
    promise,
    resolve: (value?: Value) => resolvePromise(value as Value),
    reject: (reason) => rejectPromise(reason)
  };
}
function fixture({
  syncImpl,
  disposeMode = 'destroyed'
}: {
  syncImpl?: (pending: Promise<void> | null) => Promise<void> | void;
  disposeMode?: string;
} = {}): H2oTestFixture {
  const lost = deferred<{reason: string}>(),
    events: string[] = [],
    writes: H2oTestFixture['writes'] = [],
    graphCalls: unknown[][] = [];
  const context: H2oTestContext = {
    pendingGPUToCPUCopy: null,
    copyRawBytesToBuffer(bytes: Uint8Array, pointer: unknown, offset: number, length: number) {
      writes.push({bytes: [...bytes], pointer, offset, length});
    }
  };
  let uncaptured: ((event: {error?: {message?: unknown} | null | undefined}) => void) | undefined;
  const gpu: H2oTestGpu = {
    lost: lost.promise,
    addEventListener(name, callback) {
      if (name === 'uncapturederror') uncaptured = callback;
    },
    pushErrorScope() {
      events.push('push');
    },
    async popErrorScope() {
      events.push('pop');
      return null;
    }
  };
  const device: H2oTestDevice = {
    async sync() {
      events.push('sync');
      const pending = context.pendingGPUToCPUCopy;
      context.pendingGPUToCPUCopy = null;
      if (syncImpl) return syncImpl(pending);
      if (pending) await pending;
    }
  };
  let pointer = 0;
  const runtime: H2oTestRuntime = {
    lib: {webGPUContext: context},
    initWebGPU(g) {
      g.lost.then(
        () => this.dispose(),
        () => {}
      );
    },
    beginScope() {
      events.push('begin');
    },
    endScope() {
      events.push('end');
    },
    dispose() {
      events.push('dispose');
      if (disposeMode === 'rejected') lost.reject(Error('loss rejected'));
      else if (disposeMode === 'uncaptured') {
        uncaptured!({error: Error('late validation')});
        lost.resolve({reason: 'destroyed'});
      } else if (disposeMode === 'uncaptured-microtask') {
        lost.resolve({reason: 'destroyed'});
        queueMicrotask(() => uncaptured!({error: Error('late validation')}));
      } else if (disposeMode !== 'pending') lost.resolve({reason: disposeMode});
    },
    webgpu() {
      return device;
    },
    cpu() {
      return 'cpu';
    },
    createVirtualMachine() {
      return {
        getFunction() {
          return (...args: unknown[]) => {
            graphCalls.push(args);
            return {dtype: 'float32', shape: [1, 1, 255]};
          };
        }
      };
    },
    getGlobalFunc() {
      return () => events.push('clear');
    },
    empty(shape, dtype, where) {
      const id = ++pointer;
      const tensor: H2oTestTensor = {
        shape,
        dtype,
        byteOffset: 0,
        getDataPtr: () => id,
        copyFrom(value: unknown) {
          if (where === 'cpu') {
            context.pendingGPUToCPUCopy = Promise.resolve();
            this.value = new Float32Array(255);
            this.value[1] = 2;
          } else this.value = Array.from(value as ArrayLike<number>);
          return this;
        },
        toArray(): unknown {
          return this.value;
        }
      };
      return tensor;
    }
  };
  return {gpu, device, runtime, context, lost, events, writes, graphCalls};
}

test('runtime context setup runs after WebGPU initialization and before VM allocation', () => {
  const f = fixture();
  const events: string[] = [];
  f.runtime.initWebGPU = () => events.push('init');
  f.runtime.beginScope = () => events.push('scope');
  f.runtime.webgpu = () => {
    events.push('device');
    return f.device;
  };
  f.runtime.createVirtualMachine = () => {
    events.push('vm');
    return {getFunction: () => () => {}};
  };

  const owner = initializeH2oRuntime(f.runtime, f.gpu, {
    setupWebGPUContext(context) {
      assert.equal(context, f.context);
      events.push('setup');
    }
  });

  assert.deepEqual(events, ['init', 'setup', 'scope', 'device', 'vm']);
  assert.equal(owner.phase, 'prepared');
});
const inputs = (n: number): H2oInputTuple => [
  {dtype: 'int32', shape: [1, n], data: Int32Array.from({length: n}, (_, i) => i + 1)},
  {dtype: 'int32', shape: [1, n], data: Int32Array.from({length: n}, (_, i) => i)},
  {dtype: 'float16', shape: [1, 1, n, n], data: new Uint16Array(n * n)},
  {dtype: 'float16', shape: [1, n], data: new Uint16Array(n).fill(0x3c00)},
  {dtype: 'int32', shape: [1], data: Int32Array.of(n - 1)}
];
const digest = (data: Uint8Array): string => createHash('sha256').update(data).digest('hex');
const makeHasher = () => {
  const h = createHash('sha256');
  return {update: (bytes: unknown) => h.update(bytes as Uint8Array), digest: () => h.digest('hex')};
};
const rotaryTargets = ['body.rotary_emb.inv_freq', 'body.rotary_emb.original_inv_freq'];
const rotaryInputs = (): H2oRotaryParameter[] =>
  rotaryTargets.map((target) => ({
    kind: 'rotary',
    target,
    dtype: 'float32',
    shape: [32],
    values: Array<number>(32).fill(1)
  }));

function repeatedRunnerWeights(): H2oTensor[] {
  // The runner contract checks ordered truthy placeholders here; these mocks
  // intentionally remain empty objects as in the original JavaScript test.
  return Array(682).fill({} as unknown as H2oTensor);
}

function distinctRunnerWeights(): H2oTensor[] {
  // Keep Array.from's distinct placeholder identity for the original cases.
  return Array.from({length: 682}, () => ({}) as unknown as H2oTensor);
}

function replaceInputData(input: H2oInput, data: unknown): void {
  (input as unknown as {data: unknown}).data = data;
}

function rangeResponse(bytes: unknown): H2oRangeResponse {
  // The test only exercises response bytes; the real range source also returns
  // a cache marker that these legacy mocks intentionally do not provide.
  return {bytes} as H2oRangeResponse;
}

for (const mode of ['unknown', 'rejected', 'pending', 'uncaptured', 'uncaptured-microtask']) {
  test(`H2O teardown retains owner and rejects ${mode} loss or error`, async () => {
    const f = fixture({disposeMode: mode});
    const owner = initializeH2oRuntime(f.runtime, f.gpu, {teardownTimeoutMs: 15});
    const runner = new H2oTvmRunner({...owner, weights: repeatedRunnerWeights()});
    const disposal = runner.dispose();
    assert.equal(runner.dispose(), disposal);
    await assert.rejects(disposal, {code: 'INFERENCE_ERROR'});
    assert.equal(owner.disposalGuard.blocked, true);
    assert.equal(runner.weights.length, 682);
    assert.equal(runner.runtime, f.runtime);
    assert.equal(f.events.filter((x) => x === 'end').length, 1);
    assert.equal(f.events.filter((x) => x === 'dispose').length, 1);
    assert.equal(runner.dispose(), disposal);
    if (mode === 'pending') {
      f.lost.resolve({reason: 'destroyed'});
      await Promise.resolve();
      await assert.rejects(runner.dispose(), {code: 'INFERENCE_ERROR'});
      assert.equal(owner.disposalGuard.blocked, true);
    }
  });
}

test('H2O normal destroyed teardown waits for its outcome and is idempotent', async () => {
  const f = fixture(),
    owner = initializeH2oRuntime(f.runtime, f.gpu);
  const runner = new H2oTvmRunner({...owner, weights: repeatedRunnerWeights()});
  const disposal = runner.dispose();
  assert.equal(runner.dispose(), disposal);
  await disposal;
  assert.equal(owner.disposalGuard.blocked, false);
  assert.equal(owner.lifecycle.disposed, true);
  assert.equal(runner.weights.length, 0);
  assert.equal(runner.runtime, null);
});

test('H2O refuses a destroyed loss already observed before disposal', async () => {
  const f = fixture(),
    owner = initializeH2oRuntime(f.runtime, f.gpu);
  const runner = new H2oTvmRunner({...owner, weights: repeatedRunnerWeights()});
  f.lost.resolve({reason: 'destroyed'});
  await assert.rejects(runner.dispose(), {code: 'INFERENCE_ERROR'});
  assert.equal(f.events.includes('end'), false);
  assert.equal(f.events.includes('dispose'), false);
  assert.equal(owner.disposalGuard.blocked, true);
});

test('H2O does not call runtime dispose after loss or uncaptured error during scope release', async () => {
  for (const mode of ['loss', 'uncaptured']) {
    const f = fixture();
    let owner: H2oRuntimeOwner<H2oTestRuntime, H2oTestGpu>;
    f.runtime.endScope = () => {
      f.events.push('end');
      if (mode === 'loss') f.lost.resolve({reason: 'unknown'});
      else owner.uncaptured.push('synthetic validation');
    };
    owner = initializeH2oRuntime(f.runtime, f.gpu);
    const runner = new H2oTvmRunner({...owner, weights: repeatedRunnerWeights()});
    await assert.rejects(runner.dispose(), {code: 'INFERENCE_ERROR'});
    assert.deepEqual(f.events, ['begin', 'end']);
    assert.equal(owner.disposalGuard.blocked, true);
    assert.equal(runner.weights.length, 682);
  }
});

test('H2O cancellation leaves actual runner readback queued before disposal', async () => {
  const readback = deferred(),
    started = deferred();
  const f = fixture({
    syncImpl: async (pending) => {
      started.resolve();
      await pending;
    }
  });
  const empty = f.runtime.empty;
  f.runtime.empty = function (shape, dtype, where) {
    const tensor = empty.call(this, shape, dtype, where);
    if (where === 'cpu') {
      const copy = tensor.copyFrom;
      tensor.copyFrom = function (...args) {
        const result = copy.apply(this, args);
        f.context.pendingGPUToCPUCopy = readback.promise;
        return result;
      };
    }
    return tensor;
  };
  const owner = initializeH2oRuntime(f.runtime, f.gpu);
  const runner = new H2oTvmRunner({...owner, weights: repeatedRunnerWeights()});
  const plan = JSON.parse(await readFile(new URL('./fixtures/h2o-labels.json', import.meta.url), 'utf8')) as {
    labels: string[];
    labelIds: number[];
  };
  const session = await createH2oSession({
    tokenizer: {encode: () => ({ids: [1, 2, 3]})},
    plan,
    addedTokens: ['<reserved>'],
    runner
  });
  const controller = new AbortController(),
    reason = Error('cancel');
  const decision = session.decide(
    {state: 'record', questions: [{instructions: 'Choose', options: ['first', 'second']}]},
    {signal: controller.signal}
  );
  await started.promise;
  controller.abort(reason);
  await assert.rejects(decision, (error) => error === reason);
  const disposal = session.dispose();
  assert.equal(session.dispose(), disposal);
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(f.events.includes('end'), false);
  assert.equal(f.events.includes('dispose'), false);
  readback.resolve();
  await disposal;
  assert.equal(f.events.filter((x) => x === 'end').length, 2);
  assert.equal(f.events.filter((x) => x === 'dispose').length, 1);
});

test('fake TVM uploads 680 files plus rotary in order and reads FP32 head', async () => {
  const f = fixture(),
    owner = initializeH2oRuntime(f.runtime, f.gpu),
    bytes = Uint8Array.of(1, 2, 3, 4);
  const files: H2oFileParameter[] = Array.from({length: 680}, (_, i) => ({
    kind: 'file',
    path: `tensor-${i}.pending`,
    bytes: 4,
    sha256: digest(bytes),
    dtype: 'uint32',
    shape: [1]
  }));
  const parameters: H2oParameter[] = [...files, ...rotaryInputs()];
  const inventory: H2oInventory = Object.fromEntries(
    files.map((x) => [x.path, [{offset: 0, length: 4, sha256: digest(bytes)}]])
  );
  const seen: string[] = [];
  const weights = await uploadFakeH2oParameters({
    fake: true,
    owner,
    plan: {parameters},
    inventory,
    makeHasher,
    async readRange(_base, range) {
      seen.push(range.path);
      return rangeResponse(bytes.buffer);
    }
  });
  assert.deepEqual(
    seen,
    files.map((x) => x.path)
  );
  assert.equal(weights.length, 682);
  assert.ok(Object.isFrozen(weights));
  assert.throws(() => attachH2oRunner(owner, [...weights]), /not available/);
  const runner = attachH2oRunner(owner, weights);
  assert.throws(() => attachH2oRunner(owner, weights), /not available/);
  const scores = await runner.run(inputs(3));
  assert.ok(scores instanceof Float32Array);
  assert.equal(scores.length, 255);
  assert.equal(scores[1], 2);
  assert.equal(f.graphCalls[0]!.length, 687);
  assert.deepEqual(f.graphCalls[0]!.slice(5), weights);
  assert.deepEqual(
    f.writes.slice(-2).map((x) => x.length),
    [20, 8]
  );
  assert.deepEqual(f.writes.at(-1)!.bytes, [0, 60, 0, 60, 0, 60, 0, 0]);
  await runner.releaseCachedMemory();
  await runner.dispose();
  assert.ok(f.events.includes('clear'));
  assert.equal(f.events.filter((x) => x === 'dispose').length, 1);
  await assert.rejects(uploadH2oParameters(), {code: 'INVALID_ASSET'});
});

test('invalid inputs and wrong readback shapes fail before or during graph work', async () => {
  const f = fixture(),
    owner = initializeH2oRuntime(f.runtime, f.gpu);
  const runner = new H2oTvmRunner({...owner, weights: distinctRunnerWeights()});
  const bad = inputs(3);
  bad[2].data[1] = 1;
  await assert.rejects(runner.run(bad), {code: 'INVALID_INPUT'});
  assert.equal(f.graphCalls.length, 0);
  runner.fn = () => ({dtype: 'float32', shape: [1, 255]});
  await assert.rejects(runner.run(inputs(3)), {code: 'INFERENCE_ERROR'});
  await runner.dispose();
});

test('runner validates and uses one snapshot despite changing getters and hostile iterators', async () => {
  const f = fixture(),
    owner = initializeH2oRuntime(f.runtime, f.gpu),
    runner = new H2oTvmRunner({...owner, weights: distinctRunnerWeights()});
  const values = inputs(3),
    first = values[0].data;
  const counts = {dtype: 0, shape: 0, data: 0};
  const originalShape = values[0].shape;
  Object.defineProperties(values[0], {
    dtype: {
      get() {
        counts.dtype++;
        return counts.dtype === 1 ? 'int32' : 'float16';
      }
    },
    shape: {
      get() {
        counts.shape++;
        return counts.shape === 1 ? originalShape : [999];
      }
    },
    data: {
      get() {
        counts.data++;
        return counts.data === 1 ? first : Int32Array.of(-1, -1, -1);
      }
    }
  });
  values[Symbol.iterator] = () => {
    throw new Error('input iterator used');
  };
  originalShape[Symbol.iterator] = () => {
    throw new Error('shape iterator used');
  };
  first[Symbol.iterator] = () => {
    throw new Error('data iterator used');
  };
  first.slice = () => Int32Array.of(-1, -1, -1);
  Object.defineProperty(values[1], 'dtype', {
    get() {
      first[0] = -1;
      return 'int32';
    }
  });
  const mask = values[2].data;
  Object.defineProperty(values[2].shape, '0', {
    get() {
      mask[0] = 1;
      return 1;
    }
  });
  await runner.run(values);
  assert.deepEqual(counts, {dtype: 1, shape: 1, data: 1});
  assert.deepEqual((f.graphCalls[0]![0] as H2oTestTensor).value, [1, 2, 3]);
  assert.equal(f.writes.at(-2)!.bytes[0], 0);
  assert.equal(mask[0], 1);
  await runner.dispose();
});

test('runner rejects sparse inputs and shared buffers before graph work', async () => {
  const f = fixture(),
    owner = initializeH2oRuntime(f.runtime, f.gpu),
    runner = new H2oTvmRunner({...owner, weights: distinctRunnerWeights()});
  const sparse = inputs(3) as unknown[];
  delete sparse[1];
  await assert.rejects(runner.run(sparse), {code: 'INVALID_INPUT'});
  if (typeof SharedArrayBuffer === 'function') {
    for (let i = 0; i < 5; i++) {
      const values = inputs(3),
        original = values[i]!.data;
      const sharedBuffer = new SharedArrayBuffer(original.byteLength);
      const shared = original instanceof Int32Array ? new Int32Array(sharedBuffer) : new Uint16Array(sharedBuffer);
      shared.set(original);
      replaceInputData(values[i]!, shared);
      await assert.rejects(runner.run(values), {code: 'INVALID_INPUT'});
    }
  }
  assert.equal(f.graphCalls.length, 0);
  await runner.dispose();
});

test('detached buffers in every descriptor are rejected as invalid input before graph work', async () => {
  const f = fixture(),
    owner = initializeH2oRuntime(f.runtime, f.gpu),
    runner = new H2oTvmRunner({...owner, weights: distinctRunnerWeights()});
  for (let i = 0; i < 5; i++) {
    const values = inputs(3),
      buffer = values[i]!.data.buffer;
    structuredClone(buffer, {transfer: [buffer]});
    await assert.rejects(runner.run(values), {name: 'WebDecisionError', code: 'INVALID_INPUT'});
  }
  assert.equal(f.graphCalls.length, 0);
  await runner.dispose();
});

test('prototype-swapped arrays cannot impersonate the required typed arrays', async () => {
  const f = fixture(),
    owner = initializeH2oRuntime(f.runtime, f.gpu),
    runner = new H2oTvmRunner({...owner, weights: distinctRunnerWeights()});
  for (let i = 0; i < 5; i++) {
    const values = inputs(3),
      original = values[i]!.data;
    const forged = new Uint32Array(original);
    if (i === 2) forged[0] = 65536;
    Object.setPrototypeOf(forged, i === 2 || i === 3 ? Uint16Array.prototype : Int32Array.prototype);
    Object.defineProperty(forged, Symbol.toStringTag, {value: i === 2 || i === 3 ? 'Uint16Array' : 'Int32Array'});
    replaceInputData(values[i]!, forged);
    await assert.rejects(runner.run(values), {code: 'INVALID_INPUT'});
  }
  assert.equal(f.graphCalls.length, 0);
  const valid = inputs(3);
  Object.defineProperty(valid[0].data, Symbol.toStringTag, {value: 'spoofed'});
  await runner.run(valid);
  assert.equal(f.graphCalls.length, 1);
  await runner.dispose();
});

test('genuine cross-realm typed arrays retain their intrinsic brands', async () => {
  const f = fixture(),
    owner = initializeH2oRuntime(f.runtime, f.gpu),
    runner = new H2oTvmRunner({...owner, weights: distinctRunnerWeights()});
  const values = inputs(3);
  for (const value of values) {
    const type = value.dtype === 'int32' ? 'Int32Array' : 'Uint16Array';
    const foreignData: unknown = runInNewContext(`new ${type}(${JSON.stringify(Array.from(value.data))})`);
    replaceInputData(value, foreignData);
  }
  await runner.run(values);
  assert.deepEqual((f.graphCalls[0]![0] as H2oTestTensor).value, [1, 2, 3]);
  await runner.dispose();
});

test('invalid first data snapshot cannot be replaced by a later valid getter result', async () => {
  const f = fixture(),
    owner = initializeH2oRuntime(f.runtime, f.gpu),
    runner = new H2oTvmRunner({...owner, weights: distinctRunnerWeights()});
  const values = inputs(3),
    good = values[0].data;
  let reads = 0;
  Object.defineProperty(values[0], 'data', {
    get() {
      reads++;
      return reads === 1 ? Int32Array.of(-1, 2, 3) : good;
    }
  });
  await assert.rejects(runner.run(values), {code: 'INVALID_INPUT'});
  assert.equal(reads, 1);
  assert.equal(f.graphCalls.length, 0);
  await runner.dispose();
});

test('sync failure retains readback scope until settlement and poisons on undrainable queue', async () => {
  const readback = deferred();
  const f = fixture({
    syncImpl() {
      return Promise.reject(new Error('sync failed'));
    }
  });
  const owner = initializeH2oRuntime(f.runtime, f.gpu),
    runner = new H2oTvmRunner({...owner, weights: distinctRunnerWeights()});
  const empty = f.runtime.empty;
  f.runtime.empty = (shape, dtype, where) => {
    const tensor = empty(shape, dtype, where);
    if (where === 'cpu')
      tensor.copyFrom = () => {
        f.context.pendingGPUToCPUCopy = readback.promise;
        return tensor;
      };
    return tensor;
  };
  let done = false;
  const result = runner.run(inputs(2)).catch((e) => {
    done = true;
    throw e;
  });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(done, false);
  assert.equal(f.events.filter((x) => x === 'end').length, 0);
  readback.resolve();
  await assert.rejects(result, {code: 'INFERENCE_ERROR'});
  await assert.rejects(runner.dispose(), {code: 'INFERENCE_ERROR'});
  assert.equal(f.events.includes('dispose'), false);
  await assert.rejects(runner.run(inputs(2)), {code: 'INFERENCE_ERROR'});
});

test('device loss guard blocks TVM automatic disposal and reuse', async () => {
  const f = fixture(),
    owner = initializeH2oRuntime(f.runtime, f.gpu);
  const runner = new H2oTvmRunner({...owner, weights: distinctRunnerWeights()});
  f.lost.resolve({reason: 'fixture'});
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(owner.disposalGuard.blocked, true);
  assert.equal(f.events.includes('dispose'), false);
  await assert.rejects(runner.run(inputs(2)), {code: 'INFERENCE_ERROR'});
  await assert.rejects(runner.dispose(), {code: 'INFERENCE_ERROR'});
});

test('failed fake range verification retains the runtime and never opens a session', async () => {
  const f = fixture(),
    owner = initializeH2oRuntime(f.runtime, f.gpu),
    bytes = Uint8Array.of(1, 2, 3, 4);
  const entry: H2oFileParameter = {
    kind: 'file',
    path: 'a.pending',
    bytes: 4,
    sha256: digest(bytes),
    dtype: 'uint32',
    shape: [1]
  };
  const parameters: H2oParameter[] = Array.from({length: 680}, (_, i): H2oFileParameter => ({
    ...entry,
    path: i ? `t-${i}.pending` : 'a.pending'
  }));
  parameters.push(...rotaryInputs());
  const inventory: H2oInventory = {'a.pending': [{offset: 0, length: 4, sha256: digest(bytes)}]};
  await assert.rejects(
    uploadFakeH2oParameters({
      fake: true,
      owner,
      plan: {parameters},
      inventory,
      makeHasher,
      async readRange() {
        return rangeResponse(Uint8Array.of(0, 0, 0, 0).buffer);
      }
    }),
    {code: 'INVALID_ASSET'}
  );
  assert.equal(owner.phase, 'failed');
  assert.equal(owner.disposalGuard.blocked, true);
  assert.equal(f.events.includes('dispose'), false);
  assert.throws(() => attachH2oRunner(owner, distinctRunnerWeights()), /not available/);
});

test('whole-file hash mismatch and abort after a range retain the owned runtime', async () => {
  for (const mode of ['whole', 'abort']) {
    const f = fixture(),
      owner = initializeH2oRuntime(f.runtime, f.gpu),
      bytes = Uint8Array.of(1, 2, 3, 4);
    const controller = new AbortController();
    const entry: H2oFileParameter = {
      kind: 'file',
      path: 'a.pending',
      bytes: 4,
      sha256: mode === 'whole' ? '0'.repeat(64) : digest(bytes),
      dtype: 'uint32',
      shape: [1]
    };
    const parameters: H2oParameter[] = Array.from({length: 680}, (_, i): H2oFileParameter => ({
      ...entry,
      path: i ? `t-${i}.pending` : 'a.pending'
    }));
    parameters.push(...rotaryInputs());
    const inventory: H2oInventory = {'a.pending': [{offset: 0, length: 4, sha256: digest(bytes)}]};
    const task = uploadFakeH2oParameters({
      fake: true,
      owner,
      plan: {parameters},
      inventory,
      makeHasher,
      signal: controller.signal,
      async readRange() {
        if (mode === 'abort') controller.abort();
        return rangeResponse(bytes.buffer);
      }
    });
    await assert.rejects(task, mode === 'abort' ? {name: 'AbortError'} : {code: 'INVALID_ASSET'});
    assert.equal(owner.phase, 'failed');
    assert.equal(owner.disposalGuard.blocked, true);
    assert.equal(f.events.includes('dispose'), false);
  }
});

test('direct runner rejects concurrent work while a delayed readback is pending', async () => {
  const delayed = deferred(),
    f = fixture();
  const owner = initializeH2oRuntime(f.runtime, f.gpu),
    runner = new H2oTvmRunner({...owner, weights: distinctRunnerWeights()});
  const empty = f.runtime.empty;
  f.runtime.empty = (shape, dtype, where) => {
    const tensor = empty(shape, dtype, where);
    if (where === 'cpu')
      tensor.copyFrom = () => {
        f.context.pendingGPUToCPUCopy = delayed.promise;
        tensor.value = new Float32Array(255);
        return tensor;
      };
    return tensor;
  };
  const first = runner.run(inputs(2));
  await new Promise((resolve) => setImmediate(resolve));
  await assert.rejects(runner.run(inputs(2)), {code: 'INVALID_INPUT'});
  await assert.rejects(runner.dispose(), {code: 'INVALID_INPUT'});
  assert.equal(f.graphCalls.length, 1);
  delayed.resolve();
  await first;
  await runner.dispose();
});

test('root scope release failure is retained and every dispose reports the same failure', async () => {
  const f = fixture(),
    owner = initializeH2oRuntime(f.runtime, f.gpu);
  const weights = distinctRunnerWeights();
  const runner = new H2oTvmRunner({...owner, weights});
  f.runtime.endScope = () => {
    f.events.push('end-failed');
    throw new Error('fake scope cleanup failed');
  };
  const first = runner.dispose();
  assert.equal(runner.dispose(), first);
  let reason: unknown;
  await first.catch((error) => {
    reason = error;
  });
  assert.ok(reason && typeof reason === 'object');
  assert.equal(Reflect.get(reason, 'code'), 'INFERENCE_ERROR');
  const cause = Reflect.get(reason, 'cause');
  assert.ok(cause instanceof Error);
  assert.equal(cause.message, 'fake scope cleanup failed');
  await assert.rejects(runner.dispose(), (error) => error === reason);
  assert.equal(f.events.includes('dispose'), false);
  assert.equal(owner.disposalGuard.blocked, true);
  assert.equal(runner.runtime, f.runtime);
  assert.equal(runner.weights.length, 682);
});

test('runtime dispose failure retains ownership and rejects repeated disposal', async () => {
  const f = fixture();
  f.runtime.dispose = () => {
    f.events.push('dispose-failed');
    throw new Error('fake runtime cleanup failed');
  };
  const owner = initializeH2oRuntime(f.runtime, f.gpu);
  const runner = new H2oTvmRunner({...owner, weights: distinctRunnerWeights()});
  const first = runner.dispose();
  await assert.rejects(first, {code: 'INFERENCE_ERROR'});
  assert.equal(runner.dispose(), first);
  await assert.rejects(runner.dispose(), {code: 'INFERENCE_ERROR'});
  assert.equal(owner.disposalGuard.blocked, true);
  assert.equal(runner.runtime, f.runtime);
  assert.equal(runner.weights.length, 682);
  assert.equal(f.events.filter((x) => x === 'dispose-failed').length, 1);
});

test('upload rejects duplicate files and misplaced rotary tensors before allocation', async () => {
  const f = fixture(),
    owner = initializeH2oRuntime(f.runtime, f.gpu),
    bytes = Uint8Array.of(1, 2, 3, 4);
  const entry: H2oFileParameter = {
    kind: 'file',
    path: 'a.pending',
    bytes: 4,
    sha256: digest(bytes),
    dtype: 'uint32',
    shape: [1]
  };
  const files: H2oFileParameter[] = Array.from({length: 680}, (_, i) => ({...entry, path: `t-${i}.pending`}));
  const good: H2oParameter[] = [...files, ...rotaryInputs()];
  const duplicate: H2oParameter[] = [...files];
  duplicate[4] = {...files[3]!};
  duplicate.push(...rotaryInputs());
  const reversed: H2oParameter[] = [...files, ...rotaryInputs().reverse()];
  const misplaced: H2oParameter[] = [...good];
  [misplaced[0], misplaced[680]] = [misplaced[680]!, misplaced[0]!];
  const onlyRotary: H2oRotaryParameter[] = Array.from({length: 682}, (_, i) => ({...rotaryInputs()[i % 2]!}));
  for (const parameters of [duplicate, reversed, misplaced, onlyRotary]) {
    await assert.rejects(
      uploadFakeH2oParameters({
        fake: true,
        owner,
        plan: {parameters},
        inventory: {},
        makeHasher,
        readRange: async () => {
          throw new Error('should not read');
        }
      }),
      {code: 'INVALID_ASSET'}
    );
    assert.equal(owner.phase, 'prepared');
    assert.equal(f.writes.length, 0);
  }
});
