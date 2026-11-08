import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {loadFakeH2oCandidate, PRELIMINARY_LIMITS} from '../src/h2o/loader.mjs';
import {createH2oSession} from '../src/h2o/session.mjs';
import {initializeH2oRuntime, attachH2oRunner} from '../src/h2o/owner.mjs';

const plan = JSON.parse(await readFile(new URL('./fixtures/h2o-labels.json', import.meta.url)));
function deferred() {
  let resolve, reject;
  const promise = new Promise((a, b) => {
    resolve = a;
    reject = b;
  });
  return {promise, resolve, reject};
}
const tick = () => new Promise((resolve) => setImmediate(resolve));
async function until(predicate) {
  for (let n = 0; n < 20 && !predicate(); n++) await tick();
  assert.ok(predicate());
}
function fixture() {
  const events = [],
    loss = deferred();
  const gpu = {
    lost: loss.promise,
    destroy() {
      events.push('gpu-destroy');
      loss.resolve({reason: 'destroyed'});
    },
    addEventListener() {},
    pushErrorScope() {},
    async popErrorScope() {
      return null;
    }
  };
  const context = {pendingGPUToCPUCopy: null, copyRawBytesToBuffer() {}};
  const device = {
    async sync() {
      const p = context.pendingGPUToCPUCopy;
      if (p) await p;
      context.pendingGPUToCPUCopy = null;
    }
  };
  const runtime = {
    lib: {webGPUContext: context},
    initWebGPU(g) {
      g.lost.then(
        () => this.dispose(),
        () => {}
      );
    },
    beginScope() {},
    endScope() {
      events.push('end');
    },
    dispose() {
      events.push('runtime-dispose');
      loss.resolve({reason: 'destroyed'});
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
          return () => ({dtype: 'float32', shape: [1, 1, 255]});
        }
      };
    },
    getGlobalFunc() {
      return () => {};
    },
    empty(shape, dtype, where) {
      return {
        shape,
        dtype,
        byteOffset: 0,
        getDataPtr() {
          return 0;
        },
        copyFrom() {
          if (where === 'cpu') context.pendingGPUToCPUCopy = Promise.resolve();
          return this;
        },
        toArray() {
          return new Float32Array(255);
        }
      };
    }
  };
  const source = {
    stage: Uint8Array.of(1),
    manifest: Uint8Array.of(1),
    wasm: Uint8Array.of(1),
    tokenizer: Uint8Array.of(1),
    tokenizerConfig: Uint8Array.of(1),
    tokenizerContract: Uint8Array.of(1),
    inventory: {},
    readRange() {},
    async close() {
      events.push('source-close');
    }
  };
  const adapter = {
    features: new Set(['shader-f16']),
    limits: {...PRELIMINARY_LIMITS},
    async requestDevice() {
      events.push('device');
      return gpu;
    }
  };
  const deps = {
    fake: true,
    gpu: {
      async requestAdapter() {
        return adapter;
      }
    },
    async openSource() {
      return source;
    },
    async preparePlan() {
      return plan;
    },
    async verifyWasm(bytes) {
      return bytes;
    },
    async prepareTokenizer() {
      return {tokenizer: {encode: () => ({ids: [1, 2, 3]})}, addedTokens: ['<reserved>']};
    },
    async createRuntime() {
      events.push('runtime');
      return runtime;
    },
    initialize(rt, g) {
      events.push('initialize');
      return initializeH2oRuntime(rt, g, {teardownTimeoutMs: 15});
    },
    async upload({owner}) {
      events.push('upload');
      const weights = Array(682).fill({});
      owner.uploadedWeights = weights;
      return weights;
    },
    attach(owner, weights) {
      events.push('attach');
      return attachH2oRunner(owner, weights);
    },
    async createSession(args) {
      events.push('session');
      return createH2oSession(args);
    },
    makeHasher() {}
  };
  return {events, loss, gpu, runtime, source, adapter, deps};
}
const opts = {assetsUrl: 'https://example.invalid/assets', cache: false};

test('actual loader fake seam returns real session with observer-backed teardown', async () => {
  const f = fixture(),
    session = await loadFakeH2oCandidate(opts, f.deps);
  assert.deepEqual(f.events, ['device', 'runtime', 'initialize', 'upload', 'attach', 'session', 'source-close']);
  const answer = await session.decide({
    state: 'record',
    questions: [{instructions: 'Choose', options: ['first', 'second']}]
  });
  assert.equal(answer.answers.length, 1);
  await session.dispose();
  assert.deepEqual(f.events.slice(-3), ['end', 'end', 'runtime-dispose']);
});

test('late device after cancellation is destroyed before runtime or observer exists', async () => {
  const f = fixture(),
    pending = deferred(),
    controller = new AbortController(),
    reason = Error('stop');
  f.adapter.requestDevice = () => {
    f.events.push('device');
    return pending.promise;
  };
  let settled = false;
  const task = loadFakeH2oCandidate({...opts, signal: controller.signal}, f.deps).finally(() => {
    settled = true;
  });
  await until(() => f.events.includes('device'));
  controller.abort(reason);
  await tick();
  assert.equal(settled, false);
  pending.resolve(f.gpu);
  await assert.rejects(task, (error) => error === reason);
  assert.deepEqual(f.events, ['device', 'source-close', 'gpu-destroy']);
});

for (const trigger of ['abort', 'loss'])
  test(`late uninitialized runtime after ${trigger} is disposed by existing loader`, async () => {
    const f = fixture(),
      pending = deferred(),
      controller = new AbortController(),
      reason = Error('stop');
    f.deps.createRuntime = () => {
      f.events.push('runtime');
      return pending.promise;
    };
    const task = loadFakeH2oCandidate({...opts, signal: controller.signal}, f.deps);
    await until(() => f.events.includes('runtime'));
    if (trigger === 'abort') controller.abort(reason);
    else f.loss.resolve({reason: 'unknown'});
    await assert.rejects(task, (error) =>
      trigger === 'abort' ? error === reason : error.code === 'MODEL_LOAD_FAILED'
    );
    assert.deepEqual(f.events, ['device', 'runtime', 'source-close', 'gpu-destroy']);
    pending.resolve(f.runtime);
    await until(() => f.events.includes('runtime-dispose'));
    assert.equal(f.events.includes('initialize'), false);
  });

test('observer installation failure retains uninitialized owner path and does not start TVM', async () => {
  const f = fixture();
  let reads = 0;
  Object.defineProperty(f.gpu, 'lost', {
    get() {
      if (++reads === 2) throw Error('observer failure');
      return f.loss.promise;
    }
  });
  await assert.rejects(loadFakeH2oCandidate(opts, f.deps), {code: 'MODEL_LOAD_FAILED'});
  assert.deepEqual(f.events, ['device', 'runtime', 'initialize', 'source-close']);
  assert.equal(reads, 2);
});

test('load watcher installation failure detaches its caller abort listener', async () => {
  const f = fixture(),
    controller = new AbortController();
  let adds = 0,
    removes = 0;
  const signal = controller.signal;
  const add = signal.addEventListener.bind(signal),
    remove = signal.removeEventListener.bind(signal);
  signal.addEventListener = (...args) => {
    adds++;
    return add(...args);
  };
  signal.removeEventListener = (...args) => {
    removes++;
    return remove(...args);
  };
  Object.defineProperty(f.gpu, 'lost', {
    get() {
      throw Error('watch failure');
    }
  });
  await assert.rejects(loadFakeH2oCandidate({...opts, signal}, f.deps), {code: 'MODEL_LOAD_FAILED'});
  assert.equal(adds - removes, 0);
  assert.deepEqual(f.events, ['device', 'source-close', 'gpu-destroy']);
});

test('upload failure and abort retain initialized owner without disposal', async () => {
  for (const mode of ['fail', 'abort']) {
    const f = fixture(),
      controller = new AbortController(),
      reason = Error('stop');
    let owner;
    f.deps.upload = async (args) => {
      owner = args.owner;
      f.events.push('upload');
      if (mode === 'abort') controller.abort(reason);
      throw reason;
    };
    await assert.rejects(loadFakeH2oCandidate({...opts, signal: controller.signal}, f.deps), (error) =>
      mode === 'abort' ? error === reason : error.code === 'MODEL_LOAD_FAILED'
    );
    assert.equal(owner.disposalGuard.blocked, true);
    assert.deepEqual(f.events, ['device', 'runtime', 'initialize', 'upload', 'source-close']);
  }
});

test('abort while async session creation is pending waits and then retains attached owner', async () => {
  const f = fixture(),
    pending = deferred(),
    controller = new AbortController(),
    reason = Error('stop');
  let captured, owner;
  const attach = f.deps.attach;
  f.deps.attach = (o, weights) => {
    owner = o;
    return attach(o, weights);
  };
  f.deps.createSession = (args) => {
    captured = args;
    f.events.push('session');
    return pending.promise;
  };
  let settled = false;
  const task = loadFakeH2oCandidate({...opts, signal: controller.signal}, f.deps).finally(() => {
    settled = true;
  });
  await until(() => f.events.includes('session'));
  controller.abort(reason);
  await tick();
  assert.equal(settled, false);
  pending.resolve(await createH2oSession(captured));
  await assert.rejects(task, (error) => error === reason);
  assert.equal(owner.disposalGuard.blocked, true);
  assert.deepEqual(f.events, ['device', 'runtime', 'initialize', 'upload', 'attach', 'session', 'source-close']);
});

test('failed source close after session attachment retains owner and preserves failure', async () => {
  const f = fixture(),
    reason = Error('source-close failure');
  let owner;
  const attach = f.deps.attach;
  f.deps.attach = (o, weights) => {
    owner = o;
    return attach(o, weights);
  };
  f.source.close = async () => {
    f.events.push('source-close');
    throw reason;
  };
  await assert.rejects(
    loadFakeH2oCandidate(opts, f.deps),
    (error) => error.code === 'MODEL_LOAD_FAILED' && error.cause === reason
  );
  assert.equal(owner.disposalGuard.blocked, true);
  assert.deepEqual(f.events, [
    'device',
    'runtime',
    'initialize',
    'upload',
    'attach',
    'session',
    'source-close',
    'source-close'
  ]);
});

test('invalid uploaded tensors fail actual attachment and retain owner', async () => {
  const f = fixture();
  let owner;
  f.deps.upload = async (args) => {
    owner = args.owner;
    f.events.push('upload');
    return [];
  };
  await assert.rejects(loadFakeH2oCandidate(opts, f.deps), {code: 'MODEL_LOAD_FAILED'});
  assert.equal(owner.disposalGuard.blocked, true);
  assert.deepEqual(f.events, ['device', 'runtime', 'initialize', 'upload', 'attach', 'source-close']);
});
